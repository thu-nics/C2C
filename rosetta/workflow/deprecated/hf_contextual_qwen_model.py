"""
HuggingFace Qwen model backend with contextual KV-cache dropping.

This backend mirrors `rosetta.context.server`'s drop scheduling semantics while
plugging directly into CAMEL's `ChatAgent` as a `BaseModelBackend`.

Key differences vs the server:
- Uses real KV-cache dropping via `rosetta.context.contextual.ContextualModel`
  instead of attention-mask simulation.
- Supports prefix-hit reuse by keeping an internal KV-cache across calls.
- Uses an attention.py-style round-by-round prefill for cache rebuilds.

Drop schedule (`model_config_dict["drop_messages"]`):
    {drop_at_msg_id: [msg_ids_to_drop, ...], ...}

Semantics (per assistant generation after input message `input_id`):
- Prefill (input message + generation prompt) happens BEFORE applying drops at
  `drop_at_msg_id == input_id`.
- Dropped messages remain dropped for later rounds.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union

import torch
from pydantic import BaseModel

from camel.messages import OpenAIMessage
from camel.types import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)

from rosetta.context.contextual import ContextualModel, tokenize_conversation_round_by_round
from rosetta.workflow.hf_qwen_model import HFQwenModel


def _normalize_drop_messages(
    x: Optional[Dict[Union[int, str], List[int]]],
) -> Dict[int, List[int]]:
    if not x:
        return {}
    out: Dict[int, List[int]] = {}
    for k, v in x.items():
        out[int(k)] = [int(i) for i in (v or [])]
    return out


def _tools_fingerprint(tools: Optional[List[Dict[str, Any]]]) -> str:
    if not tools:
        return ""
    return json.dumps(tools, sort_keys=True, ensure_ascii=False, default=str)


def _as_dict(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump()
    if hasattr(message, "dict"):
        return message.dict()
    return {
        "role": getattr(message, "role", None),
        "content": getattr(message, "content", None),
    }


def _canonicalize_messages(messages: List[OpenAIMessage]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        m = _as_dict(msg)
        if m.get("role") == "assistant" and m.get("content") is None:
            m["content"] = ""

        tool_calls = m.get("tool_calls")
        if tool_calls is not None:
            canonical_calls: List[Dict[str, Any]] = []
            for tc in tool_calls:
                tc_dict = _as_dict(tc)
                fn = tc_dict.get("function")
                if fn is not None and not isinstance(fn, dict):
                    tc_dict["function"] = _as_dict(fn)
                canonical_calls.append(tc_dict)
            m["tool_calls"] = canonical_calls
        out.append(m)
    return out


def _common_prefix_len(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


class HFContextualQwenModel(HFQwenModel):
    """Contextual Qwen backend (KV-cache dropping) compatible with CAMEL."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ctx = ContextualModel(self.model, self.tokenizer, verbose=False)
        self._cached_messages: List[Dict[str, Any]] = []
        self._cached_tools_fingerprint = ""
        self._drop_schedule: Dict[int, List[int]] = {}
        self._applied_drop_keys: set[int] = set()

        self.cache_hits = 0
        self.cache_misses = 0
        self.last_prefix_len = 0
        self.last_cache_action: str = "init"

    def reset_cache(self) -> None:
        """Reset the internal KV cache and prefix state."""
        self._ctx.reset()
        self._cached_messages = []
        self._cached_tools_fingerprint = ""
        self._drop_schedule = {}
        self._applied_drop_keys = set()
        self.last_cache_action = "reset"

    def _apply_chat_template_ids(
        self,
        messages: List[OpenAIMessage],
        *,
        add_generation_prompt: bool,
        tools: Optional[List[Dict[str, Any]]],
    ) -> torch.Tensor:
        template_kwargs: Dict[str, Any] = {"enable_thinking": self.enable_thinking}
        if tools:
            template_kwargs["tools"] = tools

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
        return input_ids.to(self.model.device)

    def _prefill_from_scratch(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: Optional[List[Dict[str, Any]]],
        drop_messages: Dict[int, List[int]],
    ) -> None:
        self._ctx.reset()
        self._applied_drop_keys = set()

        # noqa: in standard transformers, tokenization is done in a single pass.
        # here, we tokenize round by round, which exactly matches the behavior of the contextual model. Note that is does not match the behavior of the standard model.
        input_ids, boundaries = tokenize_conversation_round_by_round(
            self.tokenizer,
            messages,
            enable_thinking=self.enable_thinking,
            tools=tools,
            add_generation_prompt_last=False,
        )
        input_ids = input_ids.to(self.model.device)

        for idx, (start, end, role, msg_id) in enumerate(boundaries):
            seg = input_ids[:, start:end]
            if seg.numel() > 0:
                self._ctx.append(seg, id=msg_id)

            if (
                role != "assistant"
                and idx < len(boundaries) - 1
                and boundaries[idx + 1][2] == "assistant"
            ):
                drop_ids = drop_messages.get(msg_id, [])
                if msg_id not in self._applied_drop_keys:
                    if drop_ids:
                        self._ctx.drop_context(drop_ids)
                    self._applied_drop_keys.add(msg_id)

    def _extend_prefill(
        self,
        messages: List[Dict[str, Any]],
        *,
        start_idx: int,
        tools: Optional[List[Dict[str, Any]]],
        drop_messages: Dict[int, List[int]],
    ) -> None:
        eos_id = self.tokenizer.eos_token_id

        for i in range(start_idx, len(messages)):
            msg = messages[i]
            role = msg.get("role")

            if role == "assistant" and not msg.get("tool_calls"):
                content = msg.get("content") or ""
                ids = self.tokenizer(
                    content, return_tensors="pt", add_special_tokens=False
                ).input_ids.to(self.model.device)
            else:
                full_no_gen = self._apply_chat_template_ids(
                    messages[: i + 1], add_generation_prompt=False, tools=tools
                )
                ids = full_no_gen[:, self._ctx.seq_length :]

                if role == "assistant" and msg.get("tool_calls") and ids.numel() > 0:
                    flat = ids[0]
                    eos_pos = (flat == int(eos_id)).nonzero(as_tuple=False)
                    if eos_pos.numel() > 0:
                        ids = ids[:, : int(eos_pos[0].item())]

            if ids.numel() > 0:
                self._ctx.append(ids, id=i)

            if (
                role != "assistant"
                and i < len(messages) - 1
                and messages[i + 1].get("role") == "assistant"
            ):
                full_with_gen = self._apply_chat_template_ids(
                    messages[: i + 1], add_generation_prompt=True, tools=tools
                )
                gen_prompt_ids = full_with_gen[:, self._ctx.seq_length :]
                if gen_prompt_ids.numel() > 0:
                    self._ctx.append(gen_prompt_ids, id=i)

                if i not in self._applied_drop_keys:
                    drop_ids = drop_messages.get(i, [])
                    if drop_ids:
                        self._ctx.drop_context(drop_ids)
                    self._applied_drop_keys.add(i)

    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        drop_messages = _normalize_drop_messages(self.model_config_dict.get("drop_messages"))
        tool_fp = _tools_fingerprint(tools)
        msgs_norm = _canonicalize_messages(messages)

        schedule_changed = any(
            self._drop_schedule.get(k, []) != drop_messages.get(k, [])
            for k in self._applied_drop_keys
        )
        tools_changed = tool_fp != self._cached_tools_fingerprint

        prefix_len = _common_prefix_len(self._cached_messages, msgs_norm)
        self.last_prefix_len = prefix_len

        can_reuse = (
            bool(self._cached_messages)
            and not tools_changed
            and not schedule_changed
            and prefix_len == len(self._cached_messages)
            and len(msgs_norm) >= len(self._cached_messages)
        )

        if not can_reuse:
            self.cache_misses += 1
            self.last_cache_action = "rebuild"
            self._prefill_from_scratch(msgs_norm, tools=tools, drop_messages=drop_messages)
        else:
            self.cache_hits += 1
            self.last_cache_action = "extend"
            self._extend_prefill(
                msgs_norm,
                start_idx=len(self._cached_messages),
                tools=tools,
                drop_messages=drop_messages,
            )

        self._cached_tools_fingerprint = tool_fp
        self._drop_schedule = drop_messages

        input_id = len(msgs_norm) - 1
        assistant_id = len(msgs_norm)

        full_with_gen = self._apply_chat_template_ids(
            msgs_norm, add_generation_prompt=True, tools=tools
        )
        gen_prompt_ids = full_with_gen[:, self._ctx.seq_length :]
        if gen_prompt_ids.numel() == 0:
            raise ValueError("Empty generation prompt tokens from chat template.")

        max_new_tokens = int(self.model_config_dict.get("max_tokens", 2048))
        temperature = float(self.model_config_dict.get("temperature", 0.7))
        top_p = float(self.model_config_dict.get("top_p", 1.0))
        top_k = int(self.model_config_dict.get("top_k", 0))

        prompt_tokens = int(full_with_gen.shape[1])

        output_text = self._ctx.generate_step(
            gen_prompt_ids,
            input_id=input_id,
            output_id=assistant_id,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            drop_ids=drop_messages.get(input_id, []),
        )
        self._applied_drop_keys.add(input_id)

        if not self.enable_thinking:
            output_text = re.sub(
                r"<think>.*?</think>", "", output_text, flags=re.DOTALL
            ).strip()

        tool_calls = self._parse_tool_calls(output_text)
        content = self._extract_content(output_text, bool(tool_calls))

        completion_tokens = int(self._ctx.seq_length - prompt_tokens)

        assistant_msg: Dict[str, Any] = {"role": "assistant", "content": content or ""}
        if tool_calls:
            assistant_msg["content"] = ""
            assistant_msg["tool_calls"] = [_as_dict(tc) for tc in (tool_calls or [])]
            assistant_msg = _canonicalize_messages([assistant_msg])[0]

        self._cached_messages = [*msgs_norm, assistant_msg]

        # Tool-call messages are stored as structured tool_calls in memory; rebuild
        # to keep tokenization aligned with the next step's message format.
        if tool_calls:
            self.last_cache_action = "rebuild_after_tool_calls"
            self._prefill_from_scratch(
                self._cached_messages,
                tools=tools,
                drop_messages=drop_messages,
            )

        return ChatCompletion(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            choices=[
                Choice(
                    finish_reason="tool_calls" if tool_calls else "stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content=content, role="assistant", tool_calls=tool_calls
                    ),
                    logprobs=None,
                )
            ],
            created=int(time.time()),
            model=self._model_name,
            object="chat.completion",
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
