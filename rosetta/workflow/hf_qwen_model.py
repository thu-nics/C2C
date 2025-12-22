"""
HuggingFace Qwen3 model backend for CAMEL.

Uses the standard HuggingFace Transformers `generate` interface directly,
compatible with CAMEL's ChatAgent and tool calling framework.

Usage:
    from rosetta.workflow.hf_qwen_model import HFQwenModel
    from camel.agents import ChatAgent
    from camel.toolkits import FunctionTool
    
    model = HFQwenModel("Qwen/Qwen3-0.6B")
    agent = ChatAgent(model=model, tools=[FunctionTool(my_func)])
    response = agent.step("Hello!")
"""

import json
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Type, Union

import torch
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from camel.models.base_model import BaseModelBackend
from camel.messages import OpenAIMessage
from camel.types import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
)
from camel.utils import BaseTokenCounter

try:
    from openai.types.chat.chat_completion_message_tool_call import Function
except ImportError:
    Function = None

from rosetta.context.attention import prepare_context as attention_prepare_context
from rosetta.context.attention import generate as attention_generate


def _normalize_drop_messages(
    x: Optional[Dict[Union[int, str], List[int]]],
) -> Dict[int, List[int]]:
    if not x:
        return {}
    out: Dict[int, List[int]] = {}
    for k, v in x.items():
        out[int(k)] = [int(i) for i in (v or [])]
    return out


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
        "tool_calls": getattr(message, "tool_calls", None),
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


class HFQwenModel(BaseModelBackend):
    """HuggingFace Qwen model backend for CAMEL.
    
    Args:
        model_name_or_path: HuggingFace model name or local path.
        model_config_dict: Generation config (temperature, max_tokens, etc).
        device_map: Device map for model loading (default: "auto").
        device: Specific GPU(s) to use. Can be:
            - int: single GPU id (e.g., 4 for cuda:4)
            - list[int]: multiple GPU ids (e.g., [4,5,6] to distribute model)
            - str: device string (e.g., "cuda:4")
            - None: use device_map as-is (default)
        dtype: Model dtype (default: torch.bfloat16).
        enable_thinking: Whether to enable Qwen thinking mode.
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        model_config_dict: Optional[Dict[str, Any]] = None,
        device_map: str = "auto",
        device: Optional[Union[int, List[int], str]] = None,
        dtype: torch.dtype = torch.bfloat16,
        enable_thinking: bool = False,
        **kwargs,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Configure device placement
        load_kwargs = {"device_map": device_map, "torch_dtype": dtype}
        if device is not None:
            if isinstance(device, int):
                # Single GPU: put everything on that device
                load_kwargs["device_map"] = f"cuda:{device}"
            elif isinstance(device, list):
                # Multiple GPUs: distribute using max_memory
                load_kwargs["device_map"] = "auto"
                load_kwargs["max_memory"] = {gpu_id: "80GB" for gpu_id in device}
            elif isinstance(device, str):
                load_kwargs["device_map"] = device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, **load_kwargs
        )
        self.enable_thinking = enable_thinking
        self._model_name = model_name_or_path

        self.role = "main"
        
        super().__init__(
            model_type=model_name_or_path,
            model_config_dict=model_config_dict or {},
        )
    
    @property
    def token_counter(self) -> BaseTokenCounter:
        """Token counter using the model's tokenizer."""
        if self._token_counter is None:
            self._token_counter = _HFTokenCounter(self.tokenizer)
        return self._token_counter

    @property
    def token_limit(self) -> int:
        """Context window limit used by CAMEL's memory and auto-summarization.

        Note: CAMEL's BaseModelBackend uses `model_config_dict["max_tokens"]`
        for token_limit by default, but in OpenAI-style configs `max_tokens`
        usually means max *generation* tokens. Override to avoid unintended
        early summarization.
        """
        default = (
            getattr(self.model.config, "max_position_embeddings", None)
            or getattr(self.model.config, "max_seq_len", None)
            or getattr(self.model.config, "seq_length", None)
            or 32768
        )
        return int(self.model_config_dict.get("token_limit", default))
    
    def _run(
        self,
        messages: List[OpenAIMessage],
        response_format: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> ChatCompletion:
        """Run inference with the model."""
        output_text, prompt_tokens, completion_tokens = self._generate_output(
            messages=messages,
            tools=tools,
        )
        
        # Remove thinking content if disabled
        if not self.enable_thinking:
            output_text = re.sub(r'<think>.*?</think>', '', output_text, flags=re.DOTALL).strip()
        
        # Parse tool calls from Qwen format
        tool_calls = self._parse_tool_calls(output_text)
        content = self._extract_content(output_text, bool(tool_calls))
        
        return ChatCompletion(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            choices=[Choice(
                finish_reason="tool_calls" if tool_calls else "stop",
                index=0,
                message=ChatCompletionMessage(
                    content=content, role="assistant", tool_calls=tool_calls
                ),
                logprobs=None,
            )],
            created=int(time.time()),
            model=self._model_name,
            object="chat.completion",
            usage=CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
    
    async def _arun(self, *args, **kwargs) -> ChatCompletion:
        """Async version - runs sync in thread."""
        import asyncio
        return await asyncio.to_thread(self._run, *args, **kwargs)
    
    @property
    def stream(self) -> bool:
        return False
    
    def _parse_tool_calls(self, text: str) -> Optional[List[ChatCompletionMessageFunctionToolCall]]:
        """Parse Qwen tool call format: <tool_call>{"name": ..., "arguments": ...}</tool_call>"""
        calls = []
        for match in re.findall(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
            try:
                data = json.loads(match)
                args = data.get("arguments", {})
                func = Function(name=data["name"], arguments=json.dumps(args) if isinstance(args, dict) else args) if Function else {"name": data["name"], "arguments": json.dumps(args)}
                calls.append(ChatCompletionMessageFunctionToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}", type="function", function=func
                ))
            except (json.JSONDecodeError, KeyError):
                continue
        return calls or None
    
    def _extract_content(self, text: str, has_tools: bool) -> Optional[str]:
        """Extract content, removing tool call tags."""
        if has_tools:
            content = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()
            return content or None
        return text

    def _generate_output(
        self,
        *,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]],
    ) -> tuple[str, int, int]:
        """Generate raw assistant text and return (text, prompt_tokens, completion_tokens)."""
        template_kwargs = {"enable_thinking": self.enable_thinking}
        if tools:
            template_kwargs["tools"] = tools

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        temp = self.model_config_dict.get("temperature", 0.7)
        gen_kwargs = {
            "max_new_tokens": self.model_config_dict.get("max_tokens", 2048),
            "do_sample": temp > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if temp > 0:
            gen_kwargs["temperature"] = temp

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        prompt_tokens = int(inputs["input_ids"].shape[1])
        new_tokens = outputs[0][prompt_tokens:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_text, prompt_tokens, int(new_tokens.numel())


class HFContextAttentionQwenModel(HFQwenModel):
    """
    HF Qwen backend that simulates KV-dropping via contextual attention masks.

    Drop schedule (`model_config_dict["extra_body"]["drop_messages"]`):
        {drop_at_msg_id: [msg_ids_to_drop, ...], ...}
    """

    def _generate_output(
        self,
        *,
        messages: List[OpenAIMessage],
        tools: Optional[List[Dict[str, Any]]],
    ) -> tuple[str, int, int]:
        msgs_norm = _canonicalize_messages(messages)
        if not msgs_norm or msgs_norm[-1].get("role") == "assistant":
            raise ValueError("messages must be non-empty and must not end with an assistant message.")

        extra_body = self.model_config_dict.get("extra_body", {"main":{}, "search":{}})
        drop_messages = _normalize_drop_messages(extra_body[self.role].get("drop_messages"))

        max_new_tokens = int(self.model_config_dict.get("max_tokens", 2048))
        temperature = float(self.model_config_dict.get("temperature", 0.7))
        top_p = float(self.model_config_dict.get("top_p", 1.0))
        top_k = int(self.model_config_dict.get("top_k", 0))

        ctx = attention_prepare_context(
            tokenizer=self.tokenizer,
            messages=msgs_norm,
            drop_ids=drop_messages,
            enable_thinking=self.enable_thinking,
            tools=tools,
        )
        prompt_tokens = int(ctx["input_ids"].shape[1])

        gen_ids = attention_generate(
            model=self.model,
            input_ids=ctx["input_ids"],
            generation_config={
                "max_new_tokens": max_new_tokens,
                "eos_token_id": self.tokenizer.eos_token_id,
            },
            left_padding=None,
            context_kwargs=ctx,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        output_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return output_text, prompt_tokens, int(len(gen_ids))


class _HFTokenCounter(BaseTokenCounter):
    """Token counter using HuggingFace tokenizer."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def count_tokens_from_messages(self, messages: List[Dict]) -> int:
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return len(self.tokenizer.encode(text))
    
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)
    
    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)
