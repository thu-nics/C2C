"""
Offline smoke test for `tokenize_prefix_conversation_all_together`.

Default mode uses a tiny locally-constructed tokenizer + simple chat template,
so it does not require network/model downloads.

Run:
  python script/context/examples/test_tokenize_all_together.py

With a real tokenizer (requires cached files or network):
  python script/context/examples/test_tokenize_all_together.py --tokenizer_name Qwen/Qwen3-1.7B
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

import torch

from rosetta.context.attention import tokenize_prefix_conversation_all_together


def _build_dummy_chat_tokenizer() -> Any:
    """
    Build a minimal tokenizer that supports `apply_chat_template(..., tokenize=True)`.

    This avoids any `from_pretrained(...)` call so it works without network access.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Tell me a short joke."},
    ]

    # Render once to discover the token surface forms used in this test.
    rendered = (
        "system: You are a helpful assistant.\n"
        "user: Hello\n"
        "assistant: Hi there!\n"
        "user: Tell me a short joke.\n"
        "assistant: "
    )
    vocab: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for tok in rendered.replace("\n", " ").split():
        if tok not in vocab:
            vocab[tok] = len(vocab)

    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = WhitespaceSplit()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = chat_template

    # Return tokenizer + the messages used to build vocab.
    tokenizer._rosetta_test_messages = messages  # type: ignore[attr-defined]
    return tokenizer


@dataclass
class SmokeTestResult:
    ok: bool
    seq_len: int
    boundaries: List[tuple[int, int, str, int]]


def _default_test_messages() -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "Tell me a short joke."},
    ]


def smoke_test_tokenize_conversation_all_together(
    *,
    tokenizer: Any,
    messages: List[Dict[str, str]],
    verbose: bool = True,
    strict_roundtrip: bool = False,
    enable_thinking: bool = False,
) -> SmokeTestResult:

    input_ids, boundaries = tokenize_prefix_conversation_all_together(
        tokenizer, messages, enable_thinking=enable_thinking
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    expected_full = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    assert torch.equal(input_ids, expected_full), "Full prompt tokenization mismatch."
    assert len(boundaries) == len(messages) + 1, "Expected one boundary per message + dummy gen-prompt boundary."
    assert boundaries[0][0] == 0, "First boundary must start at 0."
    assert boundaries[-1][1] == int(input_ids.shape[1]), "Last boundary must end at seq_len."

    # Validate that boundaries match the LCP(prefix-with-dummy) rule used in
    # `tokenize_prefix_conversation_all_together`.
    prev_end = 0
    dummy_role = "user"
    dummy_user = {"role": dummy_role, "content": "<<ROSETTA_DUMMY_MARKER>>"}
    no_gen_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    no_gen_len = int(no_gen_ids.shape[1])

    def _lcp_len(a: torch.Tensor, b: torch.Tensor) -> int:
        a1 = a[0]
        b1 = b[0]
        n = min(int(a1.numel()), int(b1.numel()))
        if n == 0:
            return 0
        diff = (a1[:n] != b1[:n]).nonzero(as_tuple=False)
        return n if diff.numel() == 0 else int(diff[0].item())

    def _role_header_prefix(role: str) -> torch.Tensor:
        candidates = [
            ("Hello", "Tell"),
            ("A", "B"),
            ("0", "1"),
            ("x", "y"),
            ("<<<ROSETTA_A>>>", "<<<ROSETTA_B>>>"),
        ]
        prelude = {"role": "system", "content": ""}
        base = tokenizer.apply_chat_template(
            [prelude],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=False,
            enable_thinking=enable_thinking,
        )
        base_len = int(base.shape[1])
        for ca, cb in candidates:
            a = tokenizer.apply_chat_template(
                [prelude, {"role": role, "content": ca}],
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            b = tokenizer.apply_chat_template(
                [prelude, {"role": role, "content": cb}],
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            )
            hlen = _lcp_len(a, b)
            min_len = min(int(a.shape[1]), int(b.shape[1]))
            if hlen < min_len and hlen >= base_len:
                return a[:, base_len:hlen]
        return torch.empty((1, 0), dtype=torch.long)

    roles = {str(m["role"]) for m in messages}
    roles.add(dummy_role)
    role_headers = {r: _role_header_prefix(r) for r in roles}

    for boundary_idx, (start, end, role, msg_id) in enumerate(boundaries):
        assert msg_id == boundary_idx, f"Expected msg_id=={boundary_idx}, got {msg_id}."
        assert start == prev_end, f"Non-contiguous boundaries at msg_id={msg_id}."
        assert start <= end, f"Invalid boundary (start>end) at msg_id={msg_id}."

        if boundary_idx < len(messages):
            assert role == messages[boundary_idx]["role"], f"Role mismatch at msg_id={msg_id}."
            if boundary_idx < len(messages) - 1:
                a = tokenizer.apply_chat_template(
                    messages[: boundary_idx + 1] + [dummy_user],
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                )
                b = tokenizer.apply_chat_template(
                    messages[: boundary_idx + 2] + [dummy_user],
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=False,
                    enable_thinking=enable_thinking,
                )
                expected_end = _lcp_len(a, b)
                next_role = str(messages[boundary_idx + 1]["role"])
                common = _lcp_len(role_headers[dummy_role], role_headers.get(next_role, torch.empty((1, 0), dtype=torch.long)))
                if common > 0:
                    expected_end = max(0, expected_end - common)
                if expected_end > no_gen_len:
                    expected_end = no_gen_len
                assert (
                    end == expected_end
                ), f"Boundary end mismatch vs LCP(prefix+dummy) at msg_id={msg_id}."
            else:
                assert end == no_gen_len, "Last (user) boundary must end at no-gen seq_len."
        else:
            assert role == "assistant", "Dummy boundary role must be 'assistant'."
            assert start == no_gen_len, "Dummy boundary must start at no-gen seq_len."
            assert end == int(input_ids.shape[1]), "Dummy boundary must end at full seq_len."

        span_ids = input_ids[:, start:end]

        # Re-encode the token IDs within this boundary slice:
        # ids -> tokens -> string -> ids (string produced via the tokenizer's own helper).
        span_tokens = tokenizer.convert_ids_to_tokens(span_ids[0].tolist())
        decoded = tokenizer.convert_tokens_to_string(span_tokens)
        decoded_ids = tokenizer(decoded, return_tensors="pt", add_special_tokens=False).input_ids
        if strict_roundtrip:
            assert torch.equal(
                span_ids, decoded_ids
            ), f"Span token-id re-encode mismatch (ids->tokens->string->ids) at msg_id={msg_id}."
        elif verbose and not torch.equal(span_ids, decoded_ids):
            print(
                f"WARNING: ids->tokens->string->ids mismatch at msg_id={msg_id} "
                f"(span_len={span_ids.shape[1]} decoded_len={decoded_ids.shape[1]})"
            )
            print(f"  decoded={decoded!r}")

        if verbose:
            print(f"msg_id={msg_id:>2} role={role:<9} span=[{start},{end}) len={end-start}")
            print(f"  span_tokens={span_tokens}")

        prev_end = end

    last_prefix = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    assert full_text == last_prefix, "Rendered full text mismatch against re-rendered full prefix."

    if verbose:
        print(f"OK: seq_len={int(input_ids.shape[1])}")

    return SmokeTestResult(ok=True, seq_len=int(input_ids.shape[1]), boundaries=boundaries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--local_files_only", action="store_true", default=False)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--strict_roundtrip", action="store_true", default=False)
    args = parser.parse_args()

    if args.tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=bool(args.trust_remote_code),
            local_files_only=bool(args.local_files_only),
        )
        messages = _default_test_messages()
    else:
        tokenizer = _build_dummy_chat_tokenizer()
        messages = tokenizer._rosetta_test_messages  # type: ignore[attr-defined]

    smoke_test_tokenize_conversation_all_together(
        tokenizer=tokenizer,
        messages=messages,
        verbose=True,
        strict_roundtrip=bool(args.strict_roundtrip),
        enable_thinking=bool(args.enable_thinking),
    )
