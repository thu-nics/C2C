"""rosetta.context.server

OpenAI API-compatible server for contextual generation.

This server currently uses the attention-mask based implementation from
`rosetta.context.attention` (simulating KV dropping via contextual attention masks).

Implemented endpoints:
- POST /v1/chat/completions  (OpenAI compatible; streaming not supported)
- POST /generate             (simple non-OpenAI helper endpoint used by examples)
- GET  /health               (basic readiness)

Notes on drop scheduling:
- Message IDs are the indices in the provided `messages` list (0, 1, 2, ...).
- Preferred: provide `drop_messages={drop_at_user_msg_id: [msg_ids_to_drop...]}` so drops
  persist across later rounds (e.g. for round 5, prefill will respect drops from round 3).
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from rosetta.context.attention import generate as attention_generate
from rosetta.context.attention import prepare_context


# -------------------------
# Request / response models
# -------------------------


class SamplingParams(BaseModel):
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    temperature: float = Field(default=0.0, ge=0.0)
    max_new_tokens: int = Field(default=256, ge=1, le=32768)
    # Preferred: provide full drop schedule for the whole conversation.
    # Example: {3: [1, 2]} means message IDs 1 & 2 are considered dropped starting at user msg 3.
    drop_messages: Optional[Dict[Union[int, str], List[int]]] = None


class GenerateRequest(BaseModel):
    # Either provide `text` (single-turn) or `messages` (chat-style).
    text: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    system_prompt: str = "You are a helpful assistant."
    enable_thinking: bool = False
    sampling_params: SamplingParams = SamplingParams()


class GenerateResponse(BaseModel):
    text: str
    finish_reason: Literal["stop", "length"]
    segment_id_map: Dict[str, int]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionsRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=256, ge=1, le=32768)
    # Non-OpenAI parameter: full drop schedule for the conversation (preferred).
    drop_messages: Optional[Dict[Union[int, str], List[int]]] = None
    stream: bool = False
    enable_thinking: bool = False


# -------------------------
# App state
# -------------------------


app = FastAPI()

_MODEL = None
_TOKENIZER = None
_MODEL_NAME = "contextual-attention-model"
_LOCK = asyncio.Lock()


def load_model(*, model_name_or_path: str, dtype: torch.dtype) -> None:
    global _MODEL, _TOKENIZER, _MODEL_NAME

    _TOKENIZER = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if _TOKENIZER.pad_token_id is None:
        _TOKENIZER.pad_token_id = _TOKENIZER.eos_token_id

    _MODEL = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    _MODEL_NAME = model_name_or_path


# -------------------------
# Core generation
# -------------------------


def _normalize_drop_messages(
    x: Optional[Dict[Union[int, str], List[int]]],
) -> Dict[int, List[int]]:
    if not x:
        return {}
    out: Dict[int, List[int]] = {}
    for k, v in x.items():
        out[int(k)] = [int(i) for i in (v or [])]
    return out


def _segment_id_map_for_messages(messages: List[Dict[str, str]]) -> Dict[str, int]:
    # IDs are message indices; generated assistant is always the next ID.
    out: Dict[str, int] = {f"msg_{i}": i for i in range(len(messages))}
    out["assistant"] = len(messages)
    return out


def _generate_one_turn(
    *,
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    drop_messages: Dict[int, List[int]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    enable_thinking: bool,
) -> tuple[str, Literal["stop", "length"]]:
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("messages must be non-empty and end with a user message")

    ctx = prepare_context(
        tokenizer=tokenizer,
        messages=messages,
        drop_ids=drop_messages,
        enable_thinking=enable_thinking,
    )
    gen_ids: List[int] = attention_generate(
        model=model,
        input_ids=ctx["input_ids"],
        generation_config={"max_new_tokens": int(max_new_tokens), "eos_token_id": tokenizer.eos_token_id},
        left_padding=None,
        context_kwargs=ctx,
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
    )

    finish: Literal["stop", "length"] = "length" if len(gen_ids) >= int(max_new_tokens) else "stop"
    return tokenizer.decode(gen_ids, skip_special_tokens=True), finish


def _build_messages(req: GenerateRequest) -> List[Dict[str, str]]:
    if req.messages is not None:
        if len(req.messages) == 0:
            raise HTTPException(status_code=400, detail="messages cannot be empty")
        return req.messages
    if req.text is None:
        raise HTTPException(status_code=400, detail="Either `text` or `messages` must be provided")
    return [
        {"role": "system", "content": req.system_prompt},
        {"role": "user", "content": req.text},
    ]


# -------------------------
# Routes
# -------------------------


@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "ok": bool(_MODEL is not None and _TOKENIZER is not None),
        "model": _MODEL_NAME,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        raise HTTPException(status_code=500, detail="Model not loaded (server startup failed?)")

    messages = _build_messages(req)
    sp = req.sampling_params
    drop_messages = _normalize_drop_messages(sp.drop_messages)

    async with _LOCK:
        try:
            text, finish = await asyncio.to_thread(
                _generate_one_turn,
                model=_MODEL,
                tokenizer=_TOKENIZER,
                messages=messages,
                drop_messages=drop_messages,
                max_new_tokens=sp.max_new_tokens,
                temperature=sp.temperature,
                top_p=sp.top_p,
                top_k=sp.top_k,
                enable_thinking=req.enable_thinking,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return GenerateResponse(
        text=text,
        finish_reason=finish,
        segment_id_map=_segment_id_map_for_messages(messages),
    )


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsRequest) -> Dict[str, Any]:
    global _MODEL, _TOKENIZER
    if _MODEL is None or _TOKENIZER is None:
        raise HTTPException(status_code=500, detail="Model not loaded (server startup failed?)")
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not supported")

    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    drop_messages = _normalize_drop_messages(req.drop_messages)

    async with _LOCK:
        try:
            text, finish = await asyncio.to_thread(
                _generate_one_turn,
                model=_MODEL,
                tokenizer=_TOKENIZER,
                messages=messages,
                drop_messages=drop_messages,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                enable_thinking=req.enable_thinking,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or _MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish,
            }
        ],
        # non-standard metadata (safe for OpenAI clients to ignore)
        "segment_id_map": _segment_id_map_for_messages(messages),
    }


# -------------------------
# CLI
# -------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible contextual server")
    parser.add_argument("--model", type=str, required=True, help="HF model path or name")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32", "auto"],
    )
    args = parser.parse_args(argv)

    if args.dtype == "auto":
        dtype = torch.float16
    else:
        dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[args.dtype]

    load_model(model_name_or_path=args.model, dtype=dtype)

    import uvicorn  # local import so server deps are optional

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
