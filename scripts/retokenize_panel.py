"""Write a token panel re-encoded for another model's tokenizer.

Each document is decoded with the source tokenizer and encoded with the target one.
A decode window's context boundary moves to the first target token that starts at
or after the source boundary; documents whose boundary cannot be recovered are
dropped and listed in the output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    """Re-encode every selected document and write the converted panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--documents-per-source", type=int, default=0)
    args = parser.parse_args()
    source = AutoTokenizer.from_pretrained(args.source_model, local_files_only=True)
    target = AutoTokenizer.from_pretrained(args.target_model, local_files_only=True)
    panel = json.loads(args.panel.read_text())
    if args.max_model_len <= args.window or args.window < 1:
        parser.error("Require a positive window smaller than --max-model-len")
    counts: dict[str, int] = {}
    converted = []
    source_boundary_mismatches = 0
    dropped_boundary_ids: list[str] = []
    for original in panel["documents"]:
        domain = original["source"]
        if (
            args.documents_per_source
            and counts.get(domain, 0) >= args.documents_per_source
        ):
            continue
        text = source.decode(
            original["tokens"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        encoding = target(text, add_special_tokens=True, return_offsets_mapping=True)
        tokens = encoding["input_ids"]
        document = {
            k: v
            for k, v in original.items()
            if k
            not in {
                "tokens",
                "original_tokens",
                "sampled_positions",
                "original_length_bin",
                "stored_prefix_tokens",
                "cap_truncated_tokens",
                "window_discarded_tokens",
                "retained_context_tokens",
                "continuation_tokens",
                "observable_decode_inputs",
            }
        }
        if "context_length" in original:
            prefix = source.decode(
                original["tokens"][: original["context_length"]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            boundary = len(prefix)
            if not text.startswith(prefix):
                source_boundary_mismatches += 1
                dropped_boundary_ids.append(original["id"])
                # A partial source byte sequence can make prefix decode differ
                # from full decode. Its character boundary is not recoverable.
                continue
            # A target token crossing the source character boundary remains in
            # stock prefill. Only tokens whose start is at or after it decode.
            context = next(
                (
                    index
                    for index, (start, _) in enumerate(encoding["offset_mapping"])
                    if start >= boundary
                ),
                len(tokens),
            )
            window = min(args.window, len(tokens) - context)
            if context < 1 or window < 1:
                continue
            end = context + window
            start = max(0, end - args.max_model_len)
            document["tokens"] = tokens[start:end]
            if (
                start
                and target.bos_token_id is not None
                and tokens[0] == target.bos_token_id
            ):
                document["tokens"][0] = target.bos_token_id
            document["context_length"] = context - start
            document["window_tokens"] = window
        else:
            document["tokens"] = tokens[: args.max_model_len]
            if len(document["tokens"]) < 2:
                continue
        converted.append(document)
        counts[domain] = counts.get(domain, 0) + 1
    result = {key: panel[key] for key in ("role", "data") if key in panel}
    result["documents"] = converted
    result["retokenization"] = {
        "source_panel": str(args.panel),
        "source_model": str(args.source_model),
        "target_model": str(args.target_model),
        "text_reconstruction": "source tokenizer decode, then target tokenizer encode",
        "max_model_len": args.max_model_len,
        "decode_window_cap": args.window,
        "documents_per_source": args.documents_per_source,
        "source_boundary_mismatches": source_boundary_mismatches,
        "dropped_boundary_ids": dropped_boundary_ids,
        "window_tokens_by_role": {
            role: sum(
                d.get("window_tokens", 0)
                for d in converted
                if d.get("split_role", "").lower() == role
            )
            for role in ("fit", "tune")
        },
        "scope": "documents-only; source generation entries omitted",
    }
    if "data" in result:
        result["data"]["tokenizer"] = {
            "id": str(args.target_model),
            "revision": args.target_model.name,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
