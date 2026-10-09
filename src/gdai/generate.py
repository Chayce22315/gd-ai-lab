"""Generate a sample plan with a trained from-scratch checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from .model import ModelConfig, TinyFromScratchTransformer
from .schemas import validate_output
from .tokenizer import BOS_ID, EOS_ID, RESPONSE_ID, TASK_ID, decode_bytes, encode_bytes


def prepare_prompt(model_name: str, task: str, request: str) -> str:
    constraints = (
        "Return compact JSON only. Use semantic object roles. Do not invent numeric object IDs; require verified catalog lookup."
        if model_name == "deconet" else
        "Return compact JSON only. Treat exact 2.2081 trigger fields and numeric IDs as unknown unless verified in the catalog."
    )
    return f"task={task}\nrequest={request}\nconstraints={constraints}"


def load_model(checkpoint: Path, device: torch.device) -> tuple[TinyFromScratchTransformer, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("format") != "gd-ai-lab-checkpoint-v1" or payload.get("from_scratch") is not True:
        raise ValueError("not a GD AI Lab from-scratch checkpoint")
    cfg = ModelConfig.from_dict(payload["model_config"])
    model = TinyFromScratchTransformer(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def generate_text(model: TinyFromScratchTransformer, prompt: str, device: torch.device, max_new_tokens: int, temperature: float, top_k: int) -> str:
    prefix = torch.tensor([[BOS_ID,TASK_ID,*encode_bytes(prompt),RESPONSE_ID]],dtype=torch.long,device=device)
    # Keep at least the task/control tokens if user input is unusually large.
    prefix = prefix[:, -model.config.max_seq_len:]
    with torch.no_grad():
        result = model.generate(prefix,max_new_tokens=max_new_tokens,temperature=temperature,top_k=top_k)
    generated = result[0, prefix.shape[1]:].tolist()
    return decode_bytes(generated)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=["deconet","gdcore"],required=True)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--task",default=None,help="For gdcore: platformer, classic_layout, trigger_plan, title_screen, or shop_ui")
    parser.add_argument("--prompt",required=True)
    parser.add_argument("--max-new-tokens",type=int,default=1200)
    parser.add_argument("--temperature",type=float,default=0.65)
    parser.add_argument("--top-k",type=int,default=24)
    parser.add_argument("--json-output",type=Path,default=None,help="write a valid parsed JSON plan to this path")
    parser.add_argument("--svg-output",type=Path,default=None,help="for decoNET, render the plan to a self-contained SVG concept preview")
    args=parser.parse_args()
    if args.svg_output is not None and args.model != "deconet":
        parser.error("--svg-output is only supported for --model deconet")
    device=torch.device("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu"))
    model,payload=load_model(args.checkpoint,device)
    task = "decoration_plan" if args.model=="deconet" else (args.task or "platformer")
    full_prompt=prepare_prompt(args.model,task,args.prompt)
    text=generate_text(model,full_prompt,device,args.max_new_tokens,args.temperature,args.top_k).strip()
    print(f"# model={args.model} | from_scratch={payload['from_scratch']} | epoch={payload.get('epoch')} | device={device}")
    try:
        parsed=json.loads(text)
        errors=validate_output(args.model,parsed)
        pretty=json.dumps(parsed,indent=2)
        print(pretty)
        if args.json_output:
            args.json_output.parent.mkdir(parents=True,exist_ok=True)
            args.json_output.write_text(pretty,encoding="utf-8")
            print(f"\nJSON plan written: {args.json_output.resolve()}")
        if args.svg_output:
            from .render_svg import render_deco_plan_to_svg
            svg=render_deco_plan_to_svg(parsed)
            args.svg_output.parent.mkdir(parents=True,exist_ok=True)
            args.svg_output.write_text(svg,encoding="utf-8")
            print(f"SVG concept preview written: {args.svg_output.resolve()}")
        if errors:
            print("\nSTRUCTURE CHECK: FAIL\n- " + "\n- ".join(errors))
        else:
            print("\nSTRUCTURE CHECK: PASS (not a gameplay or in-game validity guarantee)")
    except json.JSONDecodeError as exc:
        print(text)
        print(f"\nSTRUCTURE CHECK: INVALID JSON ({exc}). A tiny scratch model may need more training data/steps.")

if __name__ == "__main__":
    main()
