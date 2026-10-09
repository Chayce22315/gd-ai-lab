"""A tiny deterministic byte tokenizer. No downloaded/pretrained vocabulary.

All 256 byte values are ordinary tokens. Four IDs are reserved for sequence
control. This keeps the training stack dependency-light, at the cost of longer
sequences than a learned BPE tokenizer.
"""
from __future__ import annotations

PAD_ID = 256
BOS_ID = 257
TASK_ID = 258
RESPONSE_ID = 259
EOS_ID = 260
VOCAB_SIZE = 261
SPECIAL_TOKENS = {
    "pad": PAD_ID,
    "bos": BOS_ID,
    "task": TASK_ID,
    "response": RESPONSE_ID,
    "eos": EOS_ID,
}


def encode_bytes(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode_bytes(ids: list[int] | tuple[int, ...], *, stop_at_eos: bool = True) -> str:
    result = bytearray()
    for raw_id in ids:
        token_id = int(raw_id)
        if token_id == EOS_ID and stop_at_eos:
            break
        if 0 <= token_id <= 255:
            result.append(token_id)
        # Other control tokens are deliberately omitted from decoded text.
    return result.decode("utf-8", errors="replace")


def encode_example(prompt: str, response: str, max_seq_len: int) -> tuple[list[int], int]:
    """Encode prompt+response and return (tokens, response_start_index).

    The response and EOS are prioritized; if a record is too long, the prompt
    is shortened before the target text. Training loss can therefore focus on
    the complete target whenever it fits the configured context window.
    """
    target = encode_bytes(response) + [EOS_ID]
    if len(target) >= max_seq_len - 2:
        target = target[: max_seq_len - 3] + [EOS_ID]
    prompt_tokens = encode_bytes(prompt)
    available_prompt = max(0, max_seq_len - 2 - len(target))
    if len(prompt_tokens) > available_prompt:
        prompt_tokens = prompt_tokens[-available_prompt:] if available_prompt else []
    prefix = [BOS_ID, TASK_ID] + prompt_tokens + [RESPONSE_ID]
    tokens = prefix + target
    return tokens[:max_seq_len], len(prefix)
