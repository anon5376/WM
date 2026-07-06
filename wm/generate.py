from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from wm.adapters.byte_tokenizer import ByteTokenizer
from wm.train.config import load_config
from wm.train.trainer import Trainer


def generate_text(
    config_path: str | Path,
    checkpoint_path: str | Path,
    prompt: str,
    *,
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> dict[str, object]:
    config = load_config(config_path)
    trainer = Trainer(config)
    trainer.load_checkpoint(checkpoint_path)
    trainer.model.eval()
    tok = ByteTokenizer()
    ids = tok.encode(prompt)
    max_len = int(config["model"]["max_len"])
    generated: list[int] = []
    with torch.no_grad():
        for _ in range(max_tokens):
            window = ids[-max_len:]
            inputs = torch.tensor(window, dtype=torch.long, device=trainer.device).unsqueeze(0)
            logits = trainer.model.forward_textlike(inputs, "text", causal=True)[0, -1]
            if temperature <= 0.0:
                next_id = int(torch.argmax(logits).detach().cpu())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).detach().cpu())
            ids.append(next_id)
            generated.append(next_id)
            if next_id == 10:
                break
    prompt_bytes = tok.encode_bytes(prompt.encode("utf-8"))
    text = bytes(ids).decode("utf-8", errors="replace")
    continuation = bytes(generated).decode("utf-8", errors="replace")
    return {
        "prompt": prompt,
        "prompt_bytes": len(prompt_bytes),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "text": text,
        "continuation": continuation,
        "generated_bytes": generated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = generate_text(
        args.config,
        args.checkpoint,
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result["text"])


if __name__ == "__main__":
    main()
