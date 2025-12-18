"""
Contextual attention mask utilities for simulating KV cache dropping.

This module provides functions to build custom 4D attention masks that simulate
dropping conversation rounds. The mask prevents assistant tokens from attending
to dropped rounds while allowing user tokens (prefill) to still see them.

Key insight: Position IDs must be preserved across drops to maintain correct
positional encoding. When we "drop" a round via attention mask, the tokens
still exist in the sequence, so position IDs continue monotonically.
"""

import torch
from typing import List, Tuple, Optional, Dict, Any

from rosetta.context.utils import top_k_top_p_filtering
from rosetta.context.utils import _print_eval_results, _log_to_wandb, HAS_WANDB


def tokenize_conversation_all_together(
    tokenizer,
    messages: List[dict],
    *,
    enable_thinking: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[torch.Tensor, List[Tuple[int, int, str, int]]]:
    """
    Tokenize the entire conversation in one pass and derive per-message boundaries.

    This matches practical usage where callers build a single prompt with
    `apply_chat_template(messages, add_generation_prompt=True)` and prefill once.

    Boundaries are computed from prefix lengths under the same chat-template rules:
    - The final user message includes the generation-prompt tokens.
    - Earlier messages use `add_generation_prompt=False`.
    """
    if not messages:
        return torch.empty((1, 0), dtype=torch.long), []

    template_kwargs: Dict[str, Any] = {"enable_thinking": enable_thinking}
    if tools:
        template_kwargs["tools"] = tools

    # Always build the full prompt in one pass.
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        **template_kwargs,
    )
    seq_len = int(input_ids.shape[1])

    # Try to derive boundaries directly from the fully-rendered prompt.
    # This is important for chat templates that are context-dependent (e.g., Qwen3
    # inserts different markup depending on whether an assistant message is final).
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,
    )

    boundaries: List[Tuple[int, int, str, int]] = []

    # Qwen-style template: messages are wrapped in <|im_start|> ... <|im_end|>\n,
    # and the generation prompt is a trailing "<|im_start|>assistant\n".
    if "<|im_start|>" in full_text and "<|im_end|>" in full_text:
        segments: List[str] = []
        cursor = 0
        for _ in messages:
            start = full_text.find("<|im_start|>", cursor)
            if start != cursor:
                raise ValueError(
                    "Failed to parse <|im_start|>/<|im_end|> blocks from chat template text "
                    f"(cursor={cursor}, start={start})."
                )
            end_marker = "<|im_end|>\n"
            end = full_text.find(end_marker, start)
            if end < 0:
                raise ValueError(
                    "Failed to find end marker '<|im_end|>\\n' while parsing chat template text."
                )
            end += len(end_marker)
            segments.append(full_text[start:end])
            cursor = end

        # Attach any remaining suffix (generation prompt) to the last message.
        suffix = full_text[cursor:]
        if suffix:
            segments[-1] = segments[-1] + suffix

        # Tokenize each segment text and ensure they concatenate to the full prompt IDs.
        all_seg_ids: List[torch.Tensor] = []
        for seg in segments:
            seg_ids = tokenizer(
                seg, return_tensors="pt", add_special_tokens=False
            ).input_ids
            all_seg_ids.append(seg_ids)
        seg_concat = torch.cat(all_seg_ids, dim=1) if all_seg_ids else torch.empty((1, 0), dtype=torch.long)

        if seg_concat.shape[1] != seq_len or not torch.equal(seg_concat, input_ids):
            raise ValueError(
                "Segment-based boundary derivation produced token IDs that do not match "
                "the full `apply_chat_template(..., tokenize=True)` tokenization."
            )

        current_pos = 0
        for msg_id, (msg, seg_ids) in enumerate(zip(messages, all_seg_ids)):
            end_pos = current_pos + int(seg_ids.shape[1])
            boundaries.append((current_pos, end_pos, str(msg.get("role")), msg_id))
            current_pos = end_pos

        if current_pos != seq_len:
            raise ValueError(
                f"Boundary computation did not consume full prompt: current_pos={current_pos} seq_len={seq_len}"
            )

        return input_ids, boundaries

    # Fallback: compute boundaries by prefix token lengths.
    # Note: This may fail for context-dependent templates; the Qwen-style branch above
    # covers the primary case in this codebase.
    current_pos = 0
    for msg_id, msg in enumerate(messages):
        prefix_add_gen = msg_id == (len(messages) - 1)
        prefix_ids = tokenizer.apply_chat_template(
            messages[: msg_id + 1],
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=prefix_add_gen,
            **template_kwargs,
        )
        end_pos = int(prefix_ids.shape[1])
        if end_pos < current_pos or end_pos > seq_len:
            raise ValueError(
                f"Inconsistent chat-template tokenization while computing boundaries: "
                f"msg_id={msg_id} current_pos={current_pos} end_pos={end_pos} seq_len={seq_len}"
            )
        boundaries.append((current_pos, end_pos, str(msg.get("role")), msg_id))
        current_pos = end_pos

    if current_pos != seq_len:
        raise ValueError(
            f"Boundary computation did not consume full prompt: current_pos={current_pos} seq_len={seq_len}"
        )

    return input_ids, boundaries


def tokenize_prefix_conversation_all_together(
    tokenizer,
    messages: List[dict],
    *,
    enable_thinking: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[torch.Tensor, List[Tuple[int, int, str, int]]]:
    """
    Tokenize the full prompt once and derive per-message boundaries by prefix-ID matching.

    Algorithm:
    - Compute the full prompt token IDs in one pass with `add_generation_prompt=True`.
    - To compute message boundaries robustly across chat templates (including templates
      whose rendering depends on whether a message is "final"), compute boundaries via
      prefix matching:
        - For each i < last: tokenize prefixes `messages[:i+1] + [dummy]` and
          `messages[:i+2] + [dummy]` (with `add_generation_prompt=False`) and take the
          longest common prefix length of those token ID sequences as the end boundary
          for message i.
        - Message (last) ends at the length of the full prompt with
          `add_generation_prompt=False` (i.e., before generation prompt tokens).
    - Finally, append one extra "dummy assistant" boundary spanning the generation
      prompt tokens, so the last user round is separated from the generation prompt.

    This approach does not rely on template-specific delimiters like <|im_start|>, and
    works across tokenizers as long as `apply_chat_template(tokenize=True)` is available.
    """
    if not messages:
        return torch.empty((1, 0), dtype=torch.long), []

    template_kwargs: Dict[str, Any] = {"enable_thinking": enable_thinking}
    if tools:
        template_kwargs["tools"] = tools

    # Dummy message used to stabilize chat-template rendering for intermediate prefixes.
    # We intentionally use a user-role message so templates that special-case "final assistant"
    # behavior (e.g., adding thinking blocks) render assistant messages consistently.
    dummy_role = "user"
    dummy_marker = "<<ROSETTA_DUMMY_MARKER>>"
    dummy_msg = {"role": dummy_role, "content": dummy_marker}

    def _apply_messages(msgs: List[dict], *, add_generation_prompt: bool) -> torch.Tensor:
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )

    def _lcp_len(a: torch.Tensor, b: torch.Tensor) -> int:
        a1 = a[0]
        b1 = b[0]
        n = min(int(a1.numel()), int(b1.numel()))
        if n == 0:
            return 0
        diff = (a1[:n] != b1[:n]).nonzero(as_tuple=False)
        return n if diff.numel() == 0 else int(diff[0].item())

    def _role_header_prefix(role: str) -> torch.Tensor:
        """
        Return a best-effort token-ID prefix for the role header emitted by the chat template.

        We compute this inside a fixed 1-message prelude so templates that emit global
        prefixes (e.g., tool preambles) don't contaminate the role-header estimate:
        - base = tokenize([system:""])
        - a = tokenize([system:"", role:ca])
        - b = tokenize([system:"", role:cb])
        - header = a[base_len : lcp(a,b)]
        """
        prelude = {"role": "system", "content": ""}
        base = _apply_messages([prelude], add_generation_prompt=False)
        base_len = int(base.shape[1])
        candidates = [
            ("Hello", "Tell"),
            ("A", "B"),
            ("0", "1"),
            ("x", "y"),
            ("<<<ROSETTA_A>>>", "<<<ROSETTA_B>>>"),
        ]
        for ca, cb in candidates:
            a = _apply_messages([prelude, {"role": role, "content": ca}], add_generation_prompt=False)
            b = _apply_messages([prelude, {"role": role, "content": cb}], add_generation_prompt=False)
            hlen = _lcp_len(a, b)
            min_len = min(int(a.shape[1]), int(b.shape[1]))
            if hlen < min_len and hlen >= base_len:
                return a[:, base_len:hlen]
        return torch.empty((1, 0), dtype=torch.long)

    roles = {str(m.get("role")) for m in messages}
    roles.add(dummy_role)
    role_headers: Dict[str, torch.Tensor] = {r: _role_header_prefix(r) for r in roles}

    # Full prompt (what we will actually prefill).
    input_ids = _apply_messages(messages, add_generation_prompt=True)
    seq_len = int(input_ids.shape[1])

    # Full prompt without generation prompt (used to split last user vs gen prompt).
    no_gen_ids = _apply_messages(messages, add_generation_prompt=False)
    no_gen_len = int(no_gen_ids.shape[1])
    if no_gen_len > seq_len or not torch.equal(input_ids[:, :no_gen_len], no_gen_ids):
        raise ValueError(
            "Expected `add_generation_prompt=True` tokenization to start with the "
            "`add_generation_prompt=False` tokenization, but it did not."
        )

    # Compute boundary endpoints for messages[0..last-1] via prefix matching with dummy.
    boundary_ends: List[int] = []
    for i in range(len(messages) - 1):
        a = _apply_messages(messages[: i + 1] + [dummy_msg], add_generation_prompt=False)
        b = _apply_messages(messages[: i + 2] + [dummy_msg], add_generation_prompt=False)
        end_pos = _lcp_len(a, b)

        # At the boundary, the sequences differ by (next message header) vs (dummy header).
        # Since many templates share common header tokens across roles (e.g., "<|im_start|>"),
        # the LCP will usually include that shared prefix; remove it so the boundary lands
        # at the end of message i.
        next_role = str(messages[i + 1].get("role"))
        common = _lcp_len(role_headers[dummy_role], role_headers.get(next_role, torch.empty((1, 0), dtype=torch.long)))
        if common > 0:
            end_pos = max(0, end_pos - common)

        if end_pos > no_gen_len:
            end_pos = no_gen_len
        boundary_ends.append(end_pos)
    boundary_ends.append(no_gen_len)

    boundaries: List[Tuple[int, int, str, int]] = []
    current_pos = 0
    for msg_id, end_pos in enumerate(boundary_ends):
        if end_pos < current_pos or end_pos > seq_len:
            raise ValueError(
                "Invalid boundary endpoints computed by prefix matching: "
                f"msg_id={msg_id} current_pos={current_pos} end_pos={end_pos} seq_len={seq_len}"
            )
        boundaries.append((current_pos, end_pos, str(messages[msg_id].get("role")), msg_id))
        current_pos = end_pos

    # Add a dummy assistant boundary for the generation prompt suffix (possibly empty).
    boundaries.append((current_pos, seq_len, "assistant", len(messages)))
    current_pos = seq_len

    if current_pos != seq_len:
        raise ValueError(f"Boundary computation did not consume full prompt: current_pos={current_pos} seq_len={seq_len}")

    return input_ids, boundaries


def get_round_boundaries(tokenizer, messages, max_length: int = 2048) -> List[Tuple[int, int, str, int]]:
    """
    Deprecated: use tokenize_conversation_round_by_round instead

    Get token boundaries for each message in the conversation.
    
    Args:
        tokenizer: The tokenizer to use
        messages: List of message dicts with 'role' and 'content'
        max_length: Maximum sequence length
        
    Returns:
        List of (start_idx, end_idx, role, msg_id) tuples.
        msg_id is the message index (0, 1, 2, ...) - each message gets its own ID.
        This matches the ID assignment in ContextualModel.generate_step().
    """
    boundaries = []
    current_pos = 0
    
    for i, msg in enumerate(messages):
        partial = messages[:i+1]
        text = tokenizer.apply_chat_template(
            partial,
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        tokens = tokenizer(text, return_tensors="pt", truncation=True, 
                          max_length=max_length, add_special_tokens=False)
        end_pos = tokens.input_ids.shape[1]
        msg_id = i  # Each message gets its own ID (matching generate_step behavior)
        boundaries.append((current_pos, end_pos, msg["role"], msg_id))
        current_pos = end_pos
    
    return boundaries
    
def build_contextual_attention_mask(
    seq_len: int,
    msg_boundaries: List[Tuple[int, int, str, int]],
    messages_to_drop: Optional[Dict[int, List[int]]] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Build a custom 4D attention mask that simulates KV dropping.
    
    This matches the behavior of ContextualModel where dropped messages are
    permanently removed from the cache.
    
    Rules (for messages in messages_to_drop[X] dropped at message X):
    - Messages with msg_id <= X CAN see dropped messages (before/at drop point)
    - Messages with msg_id > X CANNOT see dropped messages (after drop point)
    
    Args:
        seq_len: Total sequence length
        msg_boundaries: List of (start_idx, end_idx, role, msg_id)
        messages_to_drop: Dict mapping {msg_id_when_drop_happens: [msg_ids_to_drop]}.
                         E.g., {3: [1, 2]} means drop messages 1 and 2 at message 3.
                         Messages 0-3 can see 1,2; messages 4+ cannot see 1,2.
        device: Torch device
        dtype: Torch dtype
    
    Returns:
        attention_mask: (1, 1, seq_len, seq_len) mask
        Values: 0.0 = can attend, -inf = cannot attend
    """
    # Start with causal mask
    mask = torch.triu(
        torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype),
        diagonal=1
    )
    
    if messages_to_drop is None or len(messages_to_drop) == 0:
        return mask.unsqueeze(0).unsqueeze(0)
    
    # Build effective drop mapping: for each dropped message, at which message was it dropped?
    # drop_info[dropped_msg_id] = msg_id_when_it_was_dropped
    drop_info: Dict[int, int] = {}
    
    for drop_at_msg_id, dropped_msg_ids in messages_to_drop.items():
        for dm in dropped_msg_ids:
            # If a message is dropped at multiple points, use the earliest
            if dm not in drop_info or drop_at_msg_id < drop_info[dm]:
                drop_info[dm] = drop_at_msg_id
    
    # Apply masking for each dropped message
    for dropped_msg_id, dropped_at_msg_id in drop_info.items():
        # Find the token range of the dropped message
        drop_start, drop_end = None, None
        for start, end, role, msg_id in msg_boundaries:
            if msg_id == dropped_msg_id:
                drop_start = start
                drop_end = end
                break
        
        if drop_start is None:
            continue
        
        # Apply masking based on when the drop happened
        # The drop happens AT dropped_at_msg_id, meaning:
        # - msg_id <= dropped_at_msg_id: CAN see dropped messages (before/at drop point)
        # - msg_id > dropped_at_msg_id: CANNOT see dropped messages (after drop)
        for start, end, role, msg_id in msg_boundaries:
            if msg_id > dropped_at_msg_id:
                # All messages after the drop point cannot see dropped messages
                mask[start:end, drop_start:drop_end] = float('-inf')
    
    return mask.unsqueeze(0).unsqueeze(0)


def generate_with_contextual_mask(
    model,
    tokenizer,
    messages: List[dict],
    drop_ids: Optional[Dict[int, List[int]]] = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
) -> str:
    """
    Generate a response using contextual attention mask to simulate dropping.
    
    This is the unified generation function that should be used for both training
    evaluation (with dropping) and standard generation (without dropping) to ensure
    consistency.
    
    Behavior matches ContextualModel.generate_step():
    - User tokens (prefill) CAN see messages to be dropped
    - Assistant tokens (generated) CANNOT see dropped messages
    - Message IDs are assigned per-message (0, 1, 2, ...), matching generate_step
    
    Args:
        model: The model (unwrapped)
        tokenizer: Tokenizer
        messages: Conversation messages (ending with a user message)
        messages_to_drop: Dict mapping {msg_id_when_drop_happens: [msg_ids_to_drop]}.
                         E.g., {2: [0, 1]} means drop messages 0 and 1 when generating
                         the response after message 2.
                         If None or empty, standard generation.
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Generated response string
    """
    """
    One-shot prefill + decode generation that simulates KV-dropping via attention masks.

    This function is intentionally a *single-turn* primitive: it generates the assistant
    response for a conversation that ends with a non-assistant message (typically `user`,
    but `tool` is also supported). Multi-round behavior is achieved by calling this
    function repeatedly outside (as in the examples).

    Semantics (matching `ContextualModel.generate_step` / examples.yaml ID scheme):
    - Let `input_id` be the ID of the last user message (the last element in `messages`).
    - The generation prompt tokens are included in prefill; for boundary bookkeeping we
      treat them as a dummy assistant segment with `msg_id = output_id`.
    - Generated assistant tokens are treated as `output_id = input_id + 1`.
    - Drops at key `input_id` are applied AFTER prefill of the generation prompt:
        - Prefill uses drops with drop_at < input_id
        - Decode uses drops with drop_at <= input_id
    """
    ctx = prepare_context(
        tokenizer=tokenizer,
        messages=messages,
        drop_ids=drop_ids,
        enable_thinking=False,
    )
    gen_ids = generate(
        model=model,
        input_ids=ctx["input_ids"],
        generation_config={"max_new_tokens": max_new_tokens, "eos_token_id": tokenizer.eos_token_id},
        left_padding=None,
        context_kwargs=ctx,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def prepare_context(
    tokenizer,
    messages: List[dict],
    drop_ids: Optional[Dict[int, List[int]]] = None,
    enable_thinking: bool = False,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Prepare a single-turn generation context (prefill ids + boundaries + drop metadata).

    The conversation must end with a non-assistant message (typically `user`, but `tool`
    is also supported). The returned dict is intended to be passed as `context_kwargs`
    into `generate(...)`.
    """
    if drop_ids is None:
        drop_ids = {}

    if not messages or messages[-1].get("role") == "assistant":
        raise ValueError("messages must be non-empty and must not end with an assistant message.")

    # Message IDs are the message indices in `messages` (0, 1, 2, ...).
    input_id = len(messages) - 1

    # Tokenize the entire prompt once (matches practical generation), then derive boundaries
    # using prefix-ID matching (more robust across tokenizers/templates).
    # Note: boundaries include an extra dummy assistant segment that covers the
    # generation-prompt suffix tokens.
    input_ids, boundaries = tokenize_prefix_conversation_all_together(
        tokenizer, messages, enable_thinking=enable_thinking, tools=tools
    )

    # Drops are keyed by the user-message ID at which the drop happens.
    # Prefill happens before applying the drop-at-input_id; decode happens after.
    prefill_drop = {int(k): v for k, v in drop_ids.items() if int(k) < input_id} or None
    decode_drop = {int(k): v for k, v in drop_ids.items() if int(k) <= input_id} or None

    return {
        "messages": messages,
        "input_id": input_id,
        "drop_ids": drop_ids,
        "prefill_drop": prefill_drop,
        "decode_drop": decode_drop,
        "boundaries": boundaries,
        "input_ids": input_ids,
    }


def generate(
    model,
    input_ids,
    generation_config=None,
    left_padding=None,
    context_kwargs=None,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    **kwargs,
):
    """
    One-shot prefill + decode generation that simulates KV-dropping via attention masks.

    Args:
        model: HF causal LM (unwrapped).
        input_ids: Prefill tokens (1, seq_len).
        generation_config: dict-like or HF GenerationConfig. Supports max_new_tokens, eos_token_id.
        left_padding: reserved for future batching; must be None for now.
        context_kwargs: dict containing at least: boundaries, prefill_drop, decode_drop.
        **kwargs: reserved for future extensions.

    Returns:
        List[int] of generated token IDs (excluding the prompt).
    """
    if context_kwargs is None:
        raise ValueError("context_kwargs is required (needs boundaries/drop info).")
    if left_padding is not None:
        raise NotImplementedError("left_padding is not supported in this single-sample generator.")

    max_new_tokens = 256
    eos_token_id = getattr(generation_config, "eos_token_id", None) if generation_config is not None else None
    if isinstance(generation_config, dict):
        max_new_tokens = int(generation_config.get("max_new_tokens", max_new_tokens))
        eos_token_id = generation_config.get("eos_token_id", eos_token_id)
    elif generation_config is not None:
        max_new_tokens = int(getattr(generation_config, "max_new_tokens", max_new_tokens))

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]

    boundaries: List[Tuple[int, int, str, int]] = context_kwargs["boundaries"]
    prefill_drop = context_kwargs.get("prefill_drop")
    decode_drop = context_kwargs.get("decode_drop")

    # Prefill with a 4D contextual mask
    attn_mask_4d = build_contextual_attention_mask(
        seq_len=seq_len,
        msg_boundaries=boundaries,
        messages_to_drop=prefill_drop,
        device=device,
        dtype=dtype,
    )

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attn_mask_4d,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]

    # For decode, compute dropped message spans (columns) that the generated tokens must not attend to.
    dropped_msg_set = set()
    if decode_drop:
        for _, ids in decode_drop.items():
            dropped_msg_set.update(ids)

    dropped_spans: List[Tuple[int, int]] = []
    if dropped_msg_set:
        for start, end, _, mid in boundaries:
            if mid in dropped_msg_set:
                dropped_spans.append((start, end))

    generated_ids: List[int] = []
    current_pos = seq_len

    for _ in range(max_new_tokens):
        if float(temperature) > 0.0:
            logits = next_token_logits / float(temperature)
            logits = top_k_top_p_filtering(logits, top_k=int(top_k), top_p=float(top_p))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        token_id = int(next_token.item())
        if eos_token_id is not None and token_id == int(eos_token_id):
            break

        generated_ids.append(token_id)

        new_mask_row = torch.zeros(1, 1, 1, current_pos + 1, device=device, dtype=dtype)
        for start, end in dropped_spans:
            new_mask_row[:, :, :, start:end] = float("-inf")

        new_position_ids = torch.tensor([[current_pos]], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                position_ids=new_position_ids,
                attention_mask=new_mask_row,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

        current_pos += 1

    return generated_ids


@torch.no_grad()
def run_evaluation(
    model,
    tokenizer,
    eval_examples: List[Dict[str, Any]],
    accelerator,
    global_step: int,
    max_new_tokens: int = 256,
    use_drop: bool = False,
) -> List[Dict[str, Any]]:
    """
    Generate responses for evaluation examples with sequential multi-round generation.
    Each round's response is generated by the model and used as context for the next round.
    
    YAML format expected:
        - system_prompt: optional system prompt (ID 0 if present)
        - user_messages: list of user message strings
        - drop_messages: {msg_id_when_drop_happens: [msg_ids_to_drop]}
    
    Message IDs (matching generate_step):
        - ID 0 = system prompt (if present)
        - ID 1 = first user message
        - ID 2 = first assistant response
        - ID 3 = second user message
        - etc.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        eval_examples: List of evaluation examples from YAML
        accelerator: Accelerator instance
        global_step: Current training step
        max_new_tokens: Max tokens to generate per response
        use_drop: If True, apply drop_messages using contextual attention mask
        
    Returns:
        List of evaluation results
    """
    model.eval()
    unwrapped_model = accelerator.unwrap_model(model)
    
    eval_results = []
    
    for example in eval_examples:
        name = example.get("name", "unnamed")
        system_prompt = example.get("system_prompt", None)
        user_messages = example.get("user_messages", [])
        # drop_messages: {msg_id_when_drop_happens: [msg_ids_to_drop]}
        drop_messages = example.get("drop_messages", {})
        
        if not user_messages:
            continue
        
        # Generate responses sequentially, using model's own responses as context
        # Include system prompt if present (ID 0)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        generated_responses = []
        
        for turn_idx, user_msg in enumerate(user_messages):
            # Add user message
            messages.append({"role": "user", "content": user_msg})
            
            # Current message ID = len(messages) - 1 (0-indexed)
            # With system prompt: ID 0 = system, ID 1 = first user, etc.
            current_msg_id = len(messages) - 1
            
            # Determine which messages to drop for THIS turn
            # drop_messages format: {msg_id_when_drop_happens: [msg_ids_to_drop]}
            if use_drop and drop_messages:
                # Pass the full drop_messages dict; generate_with_contextual_mask
                # will handle accumulating drops based on current_msg_id
                effective_drop_messages = drop_messages
            else:
                effective_drop_messages = {}
            
            response = generate_with_contextual_mask(
                unwrapped_model, tokenizer, messages, effective_drop_messages, max_new_tokens
            )
            
            # Add model's response to conversation context
            messages.append({"role": "assistant", "content": response})
            generated_responses.append(response)
        
        eval_results.append({
            "name": name,
            "user_messages": user_messages,
            "generated_responses": generated_responses,
            "drop_messages": drop_messages,
            "full_conversation": messages,
        })
    
    # Log results
    if accelerator.is_main_process and eval_results:
        _print_eval_results(eval_results, global_step, use_drop)
        if HAS_WANDB:
            try:
                _log_to_wandb(accelerator, eval_results, global_step)
            except Exception as e:
                print(f"Failed to log to wandb: {e}")
    
    model.train()
    return eval_results
