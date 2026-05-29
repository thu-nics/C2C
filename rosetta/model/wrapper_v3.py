"""
RosettaModel wrapper for V3 hidden-state projectors.

This keeps the public RosettaModel interface but lets projectors marked with
`uses_source_hidden_states=True` consume sharer hidden states instead of sharer
KV tensors.
"""

from typing import List, Optional, Union

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from rosetta.model.wrapper import RosettaModel as BaseRosettaModel
from rosetta.model.wrapper import clone_kv_cache, hybrid_to_dynamic


class RosettaModel(BaseRosettaModel):
    @staticmethod
    def _uses_source_hidden_states(projector) -> bool:
        return bool(getattr(projector, "uses_source_hidden_states", False))

    def _source_requires_hidden_states(self, source_model_idx: int) -> bool:
        if self.base_model_idx not in self.projector_dict:
            return False
        if source_model_idx not in self.projector_dict[self.base_model_idx]:
            return False
        for entry in self.projector_dict[self.base_model_idx][source_model_idx].values():
            for _, projector_idx in entry:
                if self._uses_source_hidden_states(self.projector_list[projector_idx]):
                    return True
        return False

    @staticmethod
    def _get_source_hidden_for_layer(
        source_hidden_states,
        source_layer_idx: int,
        section_length: int,
    ) -> torch.Tensor:
        if source_hidden_states is None:
            raise ValueError("source_hidden_states is required for C2C V3 projection")

        # HF hidden_states[0] is the embedding / layer-0 input, and
        # hidden_states[i] is the input used to form layer i K/V.
        hidden_idx = source_layer_idx
        if hidden_idx >= len(source_hidden_states):
            hidden_idx = len(source_hidden_states) - 1

        source_hidden = source_hidden_states[hidden_idx]
        if source_hidden.size(1) != section_length:
            if source_hidden.size(1) < section_length:
                raise ValueError(
                    "Source hidden state is shorter than the target section: "
                    f"{source_hidden.size(1)} < {section_length}"
                )
            source_hidden = source_hidden[:, -section_length:, :]
        return source_hidden

    def _project_with_source(
        self,
        projector,
        source_model_layer_idx: int,
        source_kv_cache,
        source_hidden_states,
        target_kv,
        start: int,
        end: int,
    ):
        if self._uses_source_hidden_states(projector):
            source_hidden = self._get_source_hidden_for_layer(
                source_hidden_states,
                source_model_layer_idx,
                end - start,
            )
            return projector.forward(source_hidden, target_kv)

        source_key_cache, source_value_cache = source_kv_cache[source_model_layer_idx]
        new_source_key_cache = source_key_cache[:, :, start:end, :]
        new_source_value_cache = source_value_cache[:, :, start:end, :]
        return projector.forward((new_source_key_cache, new_source_value_cache), target_kv)

    def register_hooks(
        self,
        input_ids,
        attention_mask,
        position_ids,
        base_kv_cache,
        source_model_idx,
        source_kv_cache,
    ):
        base_kv_copy = clone_kv_cache(base_kv_cache)
        new_length = input_ids.shape[1]

        base_output_kv_cache = self.model_list[self.base_model_idx].forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=base_kv_copy,
            labels=None,
            use_cache=True,
        ).past_key_values

        source_kv_copy = clone_kv_cache(source_kv_cache)
        source_requires_hidden = self._source_requires_hidden_states(source_model_idx)
        source_output = self.model_list[source_model_idx].forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=source_kv_copy,
            labels=None,
            use_cache=True,
            output_hidden_states=source_requires_hidden,
            return_dict=True,
        )
        source_output_kv_cache = source_output.past_key_values
        source_hidden_states = source_output.hidden_states if source_requires_hidden else None

        fused_kv_cache = clone_kv_cache(base_output_kv_cache)

        for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
            base_key_cache, base_value_cache = base_output_kv_cache[target_layer_idx]
            new_base_key_cache = base_key_cache[:, :, -new_length:, :]
            new_base_value_cache = base_value_cache[:, :, -new_length:, :]
            new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

            projected_kv_list = []
            for source_model_layer_idx, projector_idx in entry:
                projector = self.projector_list[projector_idx]
                projected_key, projected_value = self._project_with_source(
                    projector=projector,
                    source_model_layer_idx=source_model_layer_idx,
                    source_kv_cache=source_output_kv_cache,
                    source_hidden_states=source_hidden_states,
                    target_kv=new_base_kv_cache,
                    start=base_key_cache.size(2) - new_length,
                    end=base_key_cache.size(2),
                )
                projected_kv_list.append((projected_key, projected_value))

            if not projected_kv_list:
                continue

            agg_key, agg_value = projected_kv_list[0]
            fused_kv_cache.key_cache[target_layer_idx][:, :, -new_length:, :] = agg_key
            fused_kv_cache.value_cache[target_layer_idx][:, :, -new_length:, :] = agg_value

        hook_handlers = []
        for i in range(self.model_list[self.base_model_idx].config.num_hidden_layers):
            attn = self.model_list[self.base_model_idx].model.layers[i].self_attn
            new_k = fused_kv_cache.key_cache[i][:, :, -new_length:, :]
            new_v = fused_kv_cache.value_cache[i][:, :, -new_length:, :]
            orig_forward = RosettaModel._monkeypatch_qwen3_attention_forward(attn, new_k, new_v)
            hook_handlers.append((attn, orig_forward))

        return hook_handlers, base_output_kv_cache, source_output_kv_cache

    def forward(
        self,
        kv_cache_index: Optional[List] = None,
        input_ids: Optional[Union[torch.LongTensor, List[torch.LongTensor]]] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx] if input_ids is not None else None
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
            _, seqlen = base_input_ids.size() if base_input_ids is not None else (0, 0)
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask
            _, seqlen = input_ids.size() if input_ids is not None else (0, 0)

        if seqlen > 1:
            self.kv_cache_dict = dict()

        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1
        section_lengths = [kv_cache_index[i].shape[1] for i in range(num_sections)] if kv_cache_index is not None else [seqlen]
        section_starts = [0]
        for length in section_lengths:
            section_starts.append(section_starts[-1] + length)

        curr_base_kv_cache = past_key_values
        output = None

        for i in range(num_sections):
            start = section_starts[i]
            end = section_starts[i + 1]
            prefill_input_ids = base_input_ids[:, start:end] if base_input_ids is not None else None
            prefill_attention_mask = base_attention_mask[:, :end] if base_attention_mask is not None else None
            prefill_position_ids = position_ids[:, start:end] if position_ids is not None else None
            prefill_labels = labels[:, start:end] if labels is not None else None

            if i == num_sections - 1:
                if self.include_response:
                    hook_handlers, base_output_kv_cache, source_output_kv_cache = self.register_hooks(
                        input_ids=prefill_input_ids,
                        attention_mask=prefill_attention_mask,
                        position_ids=prefill_position_ids,
                        base_kv_cache=self.kv_cache_dict[self.base_model_idx][self.base_model_idx],
                        source_model_idx=1,
                        source_kv_cache=self.kv_cache_dict[self.base_model_idx][1],
                    )

                output = self.model_list[self.base_model_idx].forward(
                    input_ids=prefill_input_ids,
                    attention_mask=prefill_attention_mask,
                    position_ids=prefill_position_ids,
                    past_key_values=curr_base_kv_cache,
                    labels=prefill_labels,
                    use_cache=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    *args,
                    **kwargs,
                )

                if self.include_response:
                    self.remove_hooks(hook_handlers)
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(base_output_kv_cache)
                    self.kv_cache_dict[self.base_model_idx][1] = clone_kv_cache(source_output_kv_cache)
                continue

            output = self.model_list[self.base_model_idx].forward(
                input_ids=prefill_input_ids,
                attention_mask=prefill_attention_mask,
                position_ids=prefill_position_ids,
                past_key_values=curr_base_kv_cache,
                labels=prefill_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                *args,
                **kwargs,
            )

            if self.base_model_idx not in self.kv_cache_dict:
                self.kv_cache_dict[self.base_model_idx] = {}
            if self.base_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = None
            self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = clone_kv_cache(output.past_key_values)

            curr_base_kv_cache: DynamicCache = output.past_key_values
            source_hidden_states_dict = {}

            for source_model_idx in range(1, len(self.model_list)):
                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if source_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                    self.kv_cache_dict[self.base_model_idx][source_model_idx] = None

                if isinstance(input_ids, list):
                    source_input_ids = input_ids[source_model_idx]
                    source_attention_mask = attention_mask[source_model_idx] if attention_mask is not None else None
                    source_prefill_input_ids = source_input_ids[:, start:end] if source_input_ids is not None else None
                    source_prefill_attention_mask = source_attention_mask[:, :end] if source_attention_mask is not None else None
                else:
                    source_prefill_input_ids = prefill_input_ids
                    source_prefill_attention_mask = prefill_attention_mask

                model = self.model_list[source_model_idx]
                was_training = model.training
                had_gc = getattr(model, "is_gradient_checkpointing", False)
                source_requires_hidden = self._source_requires_hidden_states(source_model_idx)

                try:
                    if was_training:
                        model.eval()
                    if had_gc:
                        model.gradient_checkpointing_disable()

                    with torch.no_grad():
                        out = model(
                            input_ids=source_prefill_input_ids,
                            attention_mask=source_prefill_attention_mask,
                            position_ids=prefill_position_ids,
                            past_key_values=self.kv_cache_dict[self.base_model_idx][source_model_idx],
                            use_cache=True,
                            output_hidden_states=source_requires_hidden,
                            return_dict=True,
                        )
                        curr_source_kv_cache = out.past_key_values
                        if source_requires_hidden:
                            source_hidden_states_dict[source_model_idx] = out.hidden_states
                finally:
                    if had_gc:
                        model.gradient_checkpointing_enable()
                    if was_training:
                        model.train()

                curr_source_kv_cache = hybrid_to_dynamic(curr_source_kv_cache)
                self.kv_cache_dict[self.base_model_idx][source_model_idx] = clone_kv_cache(curr_source_kv_cache)

            if self.base_model_idx in self.projector_dict:
                sharer_mask = kv_cache_index[i][0][0][0].item()
                if sharer_mask > 0:
                    base_cache = clone_kv_cache(curr_base_kv_cache)
                    parallel_delta_cache = {} if self.multi_source_fusion_mode == "parallel" else None

                    for source_model_idx in self.projector_dict[self.base_model_idx].keys():
                        if not (sharer_mask & (1 << (source_model_idx - 1))):
                            continue

                        if self.multi_source_fusion_mode == "sequential":
                            base_cache_ref = curr_base_kv_cache
                        else:
                            base_cache_ref = base_cache

                        for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
                            base_key_cache, base_value_cache = base_cache_ref[target_layer_idx]
                            new_base_key_cache = base_key_cache[:, :, start:end, :]
                            new_base_value_cache = base_value_cache[:, :, start:end, :]
                            new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

                            projected_kv_list = []
                            for source_model_layer_idx, projector_idx in entry:
                                projector = self.projector_list[projector_idx]
                                projected_key, projected_value = self._project_with_source(
                                    projector=projector,
                                    source_model_layer_idx=source_model_layer_idx,
                                    source_kv_cache=self.kv_cache_dict[self.base_model_idx][source_model_idx],
                                    source_hidden_states=source_hidden_states_dict.get(source_model_idx),
                                    target_kv=new_base_kv_cache,
                                    start=start,
                                    end=end,
                                )
                                projected_kv_list.append((projected_key, projected_value))

                            if not projected_kv_list:
                                continue

                            agg_key, agg_value = projected_kv_list[0]

                            if self.multi_source_fusion_mode == "sequential":
                                curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = agg_key
                                curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = agg_value
                            else:
                                if target_layer_idx not in parallel_delta_cache:
                                    parallel_delta_cache[target_layer_idx] = (
                                        torch.zeros_like(new_base_key_cache),
                                        torch.zeros_like(new_base_value_cache),
                                    )
                                delta_key, delta_value = parallel_delta_cache[target_layer_idx]
                                delta_key = delta_key + (agg_key - new_base_key_cache)
                                delta_value = delta_value + (agg_value - new_base_value_cache)
                                parallel_delta_cache[target_layer_idx] = (delta_key, delta_value)

                    if self.multi_source_fusion_mode == "parallel":
                        for target_layer_idx, (delta_key, delta_value) in parallel_delta_cache.items():
                            base_key_cache, base_value_cache = base_cache[target_layer_idx]
                            base_key_slice = base_key_cache[:, :, start:end, :]
                            base_value_slice = base_value_cache[:, :, start:end, :]
                            curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = base_key_slice + delta_key
                            curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = base_value_slice + delta_value

            output.past_key_values = curr_base_kv_cache

        return output
