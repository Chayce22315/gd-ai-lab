"""Train one specialist Transformer from random initialization."""
from __future__ import annotations
import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .model import ModelConfig, TinyFromScratchTransformer
from .tokenizer import BOS_ID, EOS_ID, PAD_ID, RESPONSE_ID, TASK_ID, encode_bytes


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CompletionDataset(Dataset):
    def __init__(self, path: str | Path, max_seq_len: int):
        self.path = Path(path)
        self.max_seq_len = max_seq_len
        self.examples: list[tuple[list[int], int]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("prompt"), str) or not isinstance(row.get("response"), str):
                    raise ValueError(f"{self.path}:{line_number} needs string prompt and response fields")
                self.examples.append(self._encode(row["prompt"], row["response"]))
        if not self.examples:
            raise ValueError(f"no examples found in {self.path}")

    def _encode(self, prompt: str, response: str) -> tuple[list[int], int]:
        prompt_ids = encode_bytes(prompt)
        response_ids = encode_bytes(response) + [EOS_ID]
        if len(response_ids) > self.max_seq_len - 3:
            response_ids = response_ids[: self.max_seq_len - 4] + [EOS_ID]
        prompt_room = max(0, self.max_seq_len - len(response_ids) - 3)
        if len(prompt_ids) > prompt_room:
            prompt_ids = prompt_ids[-prompt_room:] if prompt_room else []
        prefix = [BOS_ID, TASK_ID] + prompt_ids + [RESPONSE_ID]
        return prefix + response_ids, len(prefix)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[list[int], int]:
        return self.examples[index]


def collate_examples(batch: list[tuple[list[int], int]]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(tokens) for tokens, _ in batch)
    max_len = max(max_len, 2)
    x = torch.full((len(batch), max_len - 1), PAD_ID, dtype=torch.long)
    y = torch.full((len(batch), max_len - 1), -100, dtype=torch.long)
    for row_idx, (tokens, response_start) in enumerate(batch):
        inputs = torch.tensor(tokens[:-1], dtype=torch.long)
        targets = torch.tensor(tokens[1:], dtype=torch.long)
        x[row_idx, :len(inputs)] = inputs
        y[row_idx, :len(targets)] = targets
        # targets[k] represents original sequence position k+1. Mask the
        # user prompt, but train the model to emit the full target response/EOS.
        mask_count = max(0, response_start - 1)
        y[row_idx, :mask_count] = -100
        y[row_idx, len(targets):] = -100
    return x, y


@torch.no_grad()
def evaluate_loss(model: TinyFromScratchTransformer, loader: DataLoader, device: torch.device, amp_dtype: torch.dtype | None) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        enabled = device.type == "cuda" and amp_dtype is not None
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
            logits = model(x)
            loss_sum = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum")
        token_count = int((y != -100).sum().item())
        total_loss += float(loss_sum.item())
        total_tokens += token_count
    return total_loss / max(1, total_tokens)


def _save_checkpoint(path: Path, model: TinyFromScratchTransformer, optimizer: torch.optim.Optimizer, cfg: ModelConfig, args: argparse.Namespace, epoch: int, val_loss: float, train_loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "gd-ai-lab-checkpoint-v1",
        "model_name": args.model,
        "from_scratch": True,
        "initialization": "random normal weights; no pretrained/base model loaded",
        "model_config": cfg.__dict__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "seed": args.seed,
        "training_args": vars(args),
    }, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["deconet", "gdcore"], required=True)
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--val-file", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=1536)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.08)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=0, help="debug-only cap; 0 means all batches")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        parser.error("epochs, batch-size, and gradient-accumulation must be positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    if device.type == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None
    train_ds = CompletionDataset(args.train_file, args.max_seq_len)
    val_ds = CompletionDataset(args.val_file, args.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_examples, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_examples, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    cfg = ModelConfig(max_seq_len=args.max_seq_len, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads, dropout=args.dropout)
    model = TinyFromScratchTransformer(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
    except (AttributeError, TypeError):  # compatibility with older PyTorch releases
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda" and amp_dtype == torch.float16)
    start_epoch = 0
    best_val = math.inf
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if not ckpt.get("from_scratch"):
            raise ValueError("refusing to load a checkpoint not explicitly marked from_scratch")
        if ckpt["model_config"] != cfg.__dict__:
            raise ValueError("resume checkpoint config does not match current model configuration")
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        best_val = float(ckpt.get("val_loss", math.inf))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    config_path.write_text(json.dumps({"model":args.model,"from_scratch":True,"device":str(device),"parameters":model.num_parameters(),"model_config":cfg.__dict__,"args":vars(args)}, default=str, indent=2), encoding="utf-8")
    metrics_path = args.output_dir / "training_history.jsonl"
    if start_epoch == 0:
        baseline = evaluate_loss(model,val_loader,device,amp_dtype)
        with metrics_path.open("w",encoding="utf-8") as h:
            h.write(json.dumps({"epoch":-1,"stage":"random_init_baseline","val_loss":baseline,"val_perplexity":math.exp(min(20,baseline))})+"\n")
        print(f"random-init baseline val_loss={baseline:.4f} | model={args.model} | parameters={model.num_parameters():,} | device={device}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sum_loss, sum_tokens, optimizer_steps = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            enabled = device.type == "cuda" and amp_dtype is not None
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=enabled):
                logits = model(x)
                loss_sum = nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100, reduction="sum")
                token_count = int((y != -100).sum().item())
                loss = loss_sum / max(1,token_count)
                loss_for_backward = loss / args.gradient_accumulation
            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            sum_loss += float(loss_sum.detach().float().item())
            sum_tokens += token_count
            last_batch = batch_idx == len(train_loader)-1 or (args.max_train_batches and batch_idx+1 >= args.max_train_batches)
            if (batch_idx + 1) % args.gradient_accumulation == 0 or last_batch:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        train_loss = sum_loss / max(1,sum_tokens)
        val_loss = evaluate_loss(model,val_loader,device,amp_dtype)
        elapsed = time.time()-t0
        metrics = {"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"train_perplexity":math.exp(min(20,train_loss)),"val_perplexity":math.exp(min(20,val_loss)),"seconds":round(elapsed,2),"optimizer_steps":optimizer_steps}
        with metrics_path.open("a",encoding="utf-8") as h:
            h.write(json.dumps(metrics)+"\n")
        _save_checkpoint(args.output_dir/"checkpoint_last.pt",model,optimizer,cfg,args,epoch,val_loss,train_loss)
        if val_loss < best_val:
            best_val = val_loss
            _save_checkpoint(args.output_dir/"checkpoint_best.pt",model,optimizer,cfg,args,epoch,val_loss,train_loss)
        print(json.dumps(metrics))
    print(f"done. best validation loss={best_val:.4f}")
    print(f"checkpoints: {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
