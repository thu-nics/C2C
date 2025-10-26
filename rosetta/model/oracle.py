from typing import List, Optional, Union
import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
import json

from rosetta.model.projector import Projector
from rosetta.model.sampling import sample_token
from transformers.utils import ModelOutput
try:
    from transformers.generation.utils import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except Exception:
    GreedySearchDecoderOnlyOutput = None
    SampleDecoderOnlyOutput = None
from rosetta.model.wrapper import RosettaModel

class OracleRosettaModel(nn.Module):
    """
    Drop in replacement for the standard transformers LLM models, like Qwen3ForCausalLM
    """
    def __init__(self, model_list: List[PreTrainedModel], base_model_idx = 0, projector_list: List[Projector] = [], aggregator_list: List[nn.Module] = []):
        super().__init__()
        # model list: a list of model, model 0 by default is the base model
        # projector list: a list of projector
        # standard init with additional model list parameter
        # kv-cache dict: key (source_model_idx, target_model_idx), value (Cache), assume only convert at prefill with one type of model
        # projector dict: key (source_model_idx, target_model_idx) value dict(key (source_model_layer_idx, M_target value )

        self.base_model_idx = base_model_idx
        self.model_list = nn.ModuleList(model_list)

        device = model_list[base_model_idx].device
        dtype = model_list[base_model_idx].dtype
        self.projector_list = nn.ModuleList(projector_list).to(device=device, dtype=dtype)
        self.aggregator_list = nn.ModuleList(aggregator_list).to(device=device, dtype=dtype)

        self.projector_dict = {}
        self.aggregator_dict = {}
        self.kv_cache_dict = {}
        self._generation_hook_handlers = []

    @property
    def device(self):
        return self.model_list[self.base_model_idx].device
    
    def to(self, device):
        """
        Move the RosettaModel and all underlying models and projectors to the specified device.
        """
        super().to(device)
        for model in self.model_list:
            model.to(device)
        for projector in self.projector_list:
            projector.to(device)
        for aggregator in self.aggregator_list:
            aggregator.to(device)
        return self
        
    # set projector 
    def set_projector_config(self, 
                        source_model_idx: int, 
                        source_model_layer_idx: int, 
                        target_model_idx: int,
                        target_model_layer_idx: int, 
                        projector_idx: int):
        """
        Set the projector configuration
        Args:
            source_model_idx: int, the index of the source model
            source_model_layer_idx: int, the index of the source model layer
            target_model_idx: int, the index of the target model
            target_model_layer_idx: int, the index of the target model layer
            projector_idx: int, the index of the projector

        The projector dict structure supports multiple projectors per target layer.
        Structure:
        {
            target_model_idx: {
                source_model_idx: {
                    target_model_layer_idx: [(source_model_layer_idx, projector_idx), ...]
                }
            }
        }
        Repeated calls for the same (target, source, target_layer) append additional pairs.
        """

        if target_model_idx not in self.projector_dict.keys():
            self.projector_dict[target_model_idx] = {}
        if source_model_idx not in self.projector_dict[target_model_idx].keys():
            self.projector_dict[target_model_idx][source_model_idx] = {}
        # Accumulate list of (source_layer, projector_idx) for this target layer
        layer_entry = self.projector_dict[target_model_idx][source_model_idx].get(target_model_layer_idx)
        if layer_entry is None:
            self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx] = [(source_model_layer_idx, projector_idx)]
        else:
            layer_entry.append((source_model_layer_idx, projector_idx))


    def load_projector(self, projector_list):
        self.projector_list: List[Projector] = projector_list

    def load_aggregator(self, aggregator_list):
        self.aggregator_list: List[nn.Module] = aggregator_list


    def get_projector(self, 
                        source_model_idx, 
                        source_model_layer_idx, 
                        target_model_idx,
                        target_model_layer_idx):
        pair_list = self.projector_dict[target_model_idx][source_model_idx][target_model_layer_idx]
        if len(pair_list) == 0:
            raise ValueError("No projector configured for the given target layer")
        # Prefer exact source layer match
        for src_layer, projector_id in pair_list:
            if src_layer == source_model_layer_idx:
                return self.projector_list[projector_id]
        # Fallback: return the first projector
        return self.projector_list[pair_list[0][1]]

    def set_aggregator_idx(self,
                           source_model_idx: int,
                           target_model_idx: int,
                           target_model_layer_idx: int,
                           aggregator_idx: int):
        if target_model_idx not in self.aggregator_dict:
            self.aggregator_dict[target_model_idx] = {}
        if source_model_idx not in self.aggregator_dict[target_model_idx]:
            self.aggregator_dict[target_model_idx][source_model_idx] = {}
        self.aggregator_dict[target_model_idx][source_model_idx][target_model_layer_idx] = aggregator_idx


    @staticmethod
    def load_json(file_name):
        with open(file_name, "r") as f:
            result = json.load(f)
        return result
    
    @staticmethod
    def _convert_dict_keys_to_ints(obj):
        """
        Recursively convert dictionary keys that look like integers back to int.
        This reverses json.dump's coercion of dict keys to strings.
        """
        if isinstance(obj, dict):
            new_obj = {}
            for key, value in obj.items():
                if isinstance(key, str) and key.lstrip('-').isdigit():
                    new_key = int(key)
                else:
                    new_key = key
                new_obj[new_key] = RosettaModel._convert_dict_keys_to_ints(value)
            return new_obj
        if isinstance(obj, list):
            return [RosettaModel._convert_dict_keys_to_ints(v) for v in obj]
        return obj
    
    
    def save_projector_config(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.projector_dict, f)

    
    def load_projector_config(self, config_path):
        if config_path.endswith(".json"):
            loaded = RosettaModel.load_json(config_path)
            self.projector_dict = RosettaModel._convert_dict_keys_to_ints(loaded)

    def save_aggregator_config(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.aggregator_dict, f)

    def load_aggregator_config(self, config_path):
        if config_path.endswith(".json"):
            loaded = RosettaModel.load_json(config_path)
            self.aggregator_dict = RosettaModel._convert_dict_keys_to_ints(loaded)


    def set_kv_cache_dict(self, source_model_idx, target_model_idx, cache):
        if target_model_idx not in self.kv_cache_dict.keys():
            self.kv_cache_dict[target_model_idx] = {}
        if cache is None:
            # Initialize with a DynamicCache instead of RosettaCache for now
            self.kv_cache_dict[target_model_idx][source_model_idx] = DynamicCache() # noqa, maybe we should use RosettaCache here
        else:
            self.kv_cache_dict[target_model_idx][source_model_idx] = cache

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
        # **kwargs: Unpack[KwargsForCausalLM],
        identifier = -1,
        subject = None,
        *args,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass
        KVCache index is a list of tensors with shape (B, sec_seq_len, 2), indicating the source and target kv cache model index

        If input_ids is LongTensor, default to same input ids for different models
        If input_ids is Tuple, default to different input ids for different models.

        No Rosetta: (-1, 0)
        """
        
        # noqa
        self.kv_cache_dict = dict()

        # Handle different input formats: if input_ids is a list, use per-model inputs
        if isinstance(input_ids, list):
            # Use list format: different input_ids and attention_mask for each model
            base_input_ids = input_ids[self.base_model_idx] if input_ids is not None else None
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
            _, seqlen = base_input_ids.size() if base_input_ids is not None else (0, 0)
        else:
            # Use tensor format: same input_ids and attention_mask for all models (backward compatibility)
            base_input_ids = input_ids
            base_attention_mask = attention_mask
            _, seqlen = input_ids.size() if input_ids is not None else (0, 0)

        num_sections = len(kv_cache_index) if kv_cache_index is not None else 1

        section_lengths = [kv_cache_index[i].shape[1] for i in range(num_sections)] if kv_cache_index is not None else [seqlen]
        section_starts = [0]
        for l in section_lengths:
            section_starts.append(section_starts[-1] + l)
        
        curr_base_kv_cache = past_key_values

        if seqlen > 1:
            for i in range(num_sections):
                start = section_starts[i]
                end = section_starts[i + 1]
                prefill_input_ids = base_input_ids[:, start:end] if base_input_ids is not None else None
                prefill_attention_mask = base_attention_mask[:, :end] if base_attention_mask is not None else None
                prefill_position_ids = position_ids[:, start:end] if position_ids is not None else None
                prefill_labels = labels[:, start:end] if labels is not None else None

                # calculate target model kvcache
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
                    **kwargs
                )

                if self.base_model_idx not in self.kv_cache_dict:
                    self.kv_cache_dict[self.base_model_idx] = {}
                if self.base_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                    self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = None
                self.kv_cache_dict[self.base_model_idx][self.base_model_idx] = output.past_key_values

                curr_base_kv_cache: DynamicCache = output.past_key_values
                
                # if i != num_sections - 1:
                for source_model_idx in range(1, len(self.model_list)):
                    if self.base_model_idx not in self.kv_cache_dict:
                        self.kv_cache_dict[self.base_model_idx] = {}
                    if source_model_idx not in self.kv_cache_dict[self.base_model_idx]:
                        self.kv_cache_dict[self.base_model_idx][source_model_idx] = None

                    # Get model-specific input_ids and attention_mask
                    if isinstance(input_ids, list):
                        source_input_ids = input_ids[source_model_idx]
                        source_attention_mask = attention_mask[source_model_idx] if attention_mask is not None else None
                        source_prefill_input_ids = source_input_ids[:, start:end] if source_input_ids is not None else None
                        source_prefill_attention_mask = source_attention_mask[:, :end] if source_attention_mask is not None else None
                    else:
                        # Backward compatibility: use same input for all models
                        source_prefill_input_ids = prefill_input_ids
                        source_prefill_attention_mask = prefill_attention_mask

                    curr_source_kv_cache = self.model_list[source_model_idx].forward(
                        input_ids=source_prefill_input_ids,
                        attention_mask=source_prefill_attention_mask,
                        position_ids=prefill_position_ids,
                        past_key_values=self.kv_cache_dict[self.base_model_idx][source_model_idx],
                        use_cache=use_cache, 
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        *args,
                        **kwargs
                    ).past_key_values
                    self.kv_cache_dict[self.base_model_idx][source_model_idx] = curr_source_kv_cache

                # calculate source model kvcache and apply projections
                if self.base_model_idx in self.projector_dict:
                    source_model_idx = kv_cache_index[i][0][0][0].item()  # Get the source model index from the kv_cache_index
                    if source_model_idx != -1:
                        for target_layer_idx, entry in self.projector_dict[self.base_model_idx][source_model_idx].items():
                            base_key_cache, base_value_cache = curr_base_kv_cache[target_layer_idx]
                            new_base_key_cache = base_key_cache[:, :, start:end, :]
                            new_base_value_cache = base_value_cache[:, :, start:end, :]
                            new_base_kv_cache = (new_base_key_cache, new_base_value_cache)

                            pair_list = entry

                            projected_kv_list = []
                            source_kv_list = []
                            for source_model_layer_idx, projector_idx in pair_list:
                                source_key_cache, source_value_cache = self.kv_cache_dict[self.base_model_idx][source_model_idx][source_model_layer_idx]
                                new_source_key_cache = source_key_cache[:, :, start:end, :]
                                new_source_value_cache = source_value_cache[:, :, start:end, :]
                                new_source_kv_cache = (new_source_key_cache, new_source_value_cache)
                                projected_key, projected_value = self.projector_list[projector_idx].forward(
                                    new_source_kv_cache, # tuple of (key, value), each of shape (B, N, H, D)
                                    new_base_kv_cache
                                )
                                projected_kv_list.append((projected_key, projected_value))

                                # --------------
                                # save base and projected kv cache
                                torch.save((projected_key, projected_value), f"oracle/projected_kv/{subject}_{identifier}_{i}.pt")
                                torch.save(new_base_kv_cache, f"oracle/target_kv/{subject}_{identifier}_{i}.pt")
                                # --------------
                                source_kv_list.append(new_source_kv_cache)

                            # Aggregate (fallback to first projector if no aggregator is available)
                            use_aggregator = (
                                len(projected_kv_list) > 1 and
                                len(self.aggregator_list) > 0 and
                                self.base_model_idx in self.aggregator_dict and
                                source_model_idx in self.aggregator_dict[self.base_model_idx] and
                                target_layer_idx in self.aggregator_dict[self.base_model_idx][source_model_idx]
                            )

                            if use_aggregator:
                                aggregator_idx = self.aggregator_dict[self.base_model_idx][source_model_idx][target_layer_idx]
                                agg_key, agg_value = self.aggregator_list[aggregator_idx].forward(
                                    source_kv_list,
                                    new_base_kv_cache,
                                    projected_kv_list
                                )
                            else:
                                # Fallback to first projector result when no aggregator is available
                                agg_key, agg_value = projected_kv_list[0]

                            # Update cache with aggregated result
                            curr_base_kv_cache.key_cache[target_layer_idx][:, :, start:end, :] = agg_key
                            curr_base_kv_cache.value_cache[target_layer_idx][:, :, start:end, :] = agg_value
                        
                        output.past_key_values = curr_base_kv_cache
                                                                             
        # use base model for decode phase
        else:
            # Handle list input format for decode phase as well
            decode_input_ids = input_ids[self.base_model_idx] if isinstance(input_ids, list) else input_ids
            decode_attention_mask = attention_mask[self.base_model_idx] if isinstance(attention_mask, list) else attention_mask
            
            output = self.model_list[self.base_model_idx].forward(
                input_ids=decode_input_ids,
                attention_mask=decode_attention_mask,
                position_ids=position_ids,
                past_key_values=curr_base_kv_cache,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                *args,
                **kwargs
            )

        return output

    @torch.no_grad()
    def generate(
        self,
        kv_cache_index,
        input_ids,
        max_new_tokens: Optional[int] = None,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        do_sample: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        max_length: Optional[int] = None,
        use_cache: bool = True,
        *args,
        **kwargs,
    ):
        """
        New generation loop without using the base model's generate.
        - Uses this module's forward for prefill and per-token decode.
        - Samples tokens via rosetta.model.sampling.sample_token.
        Returns a tensor of shape [batch, prompt_len + generated_len] for the base model stream.
        """
        # Derive number of tokens to generate
        # If max_new_tokens not provided, infer from max_length
        if isinstance(input_ids, list):
            base_input_ids_for_len = input_ids[self.base_model_idx]
        else:
            base_input_ids_for_len = input_ids
        prompt_len = base_input_ids_for_len.size(1)

        # Default eos/pad from base model tokenizer/config if not provided
        base_model = self.model_list[self.base_model_idx]
        gen_cfg = getattr(base_model, "generation_config", None)
        cfg_obj = gen_cfg if gen_cfg is not None else getattr(base_model, "config", None)
        if eos_token_id is None and cfg_obj is not None:
            eos_token_id = getattr(cfg_obj, "eos_token_id", None)
        if pad_token_id is None and cfg_obj is not None:
            pad_token_id = getattr(cfg_obj, "pad_token_id", None)
        if pad_token_id is None and eos_token_id is not None:
            pad_token_id = eos_token_id if isinstance(eos_token_id, int) else eos_token_id[0]

        if max_new_tokens is None:
            if max_length is not None:
                if max_length <= prompt_len:
                    max_new_tokens = 0
                else:
                    max_new_tokens = max_length - prompt_len
            else:
                raise ValueError("Provide max_new_tokens or max_length")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        # Resolve base inputs
        if isinstance(input_ids, list):
            base_input_ids = input_ids[self.base_model_idx]
            base_attention_mask = attention_mask[self.base_model_idx] if attention_mask is not None else None
        else:
            base_input_ids = input_ids
            base_attention_mask = attention_mask

        if base_attention_mask is None:
            base_attention_mask = torch.ones_like(base_input_ids, dtype=torch.long, device=base_input_ids.device)

        batch_size = base_input_ids.size(0)

        # Prefill to build caches and obtain initial logits
        prefill_output = self.forward(
            kv_cache_index=kv_cache_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            *args,
            **kwargs,
        )

        current_past = prefill_output.past_key_values
        all_input_ids = base_input_ids
        current_attention_mask = base_attention_mask

        # EOS handling setup
        eos_set = None
        if eos_token_id is not None:
            eos_set = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
        finished = torch.zeros(batch_size, dtype=torch.bool, device=all_input_ids.device)

        # Start from last prefill logits
        last_logits = prefill_output.logits[:, -1, :]

        # Determine sampling mode
        if do_sample is None:
            do_sample = False
        effective_temperature = temperature if do_sample else 0.0

        # Optional scores collection
        collect_scores = bool(return_dict_in_generate) and bool(output_scores)
        scores = []

        for _ in range(max_new_tokens):
            if collect_scores:
                scores.append(last_logits)
            # Sample next token
            next_token = sample_token(last_logits, temperature=effective_temperature, top_p=top_p, top_k=top_k)
            if not isinstance(next_token, torch.Tensor):
                next_token = torch.tensor([next_token], device=all_input_ids.device, dtype=torch.long).repeat(batch_size)

            # Apply EOS logic
            if eos_set is not None:
                just_finished = torch.zeros_like(finished)
                for eid in eos_set:
                    just_finished |= (next_token == eid)
                finished = finished | just_finished
                if pad_token_id is not None:
                    next_token = torch.where(
                        finished,
                        torch.tensor(pad_token_id, device=next_token.device, dtype=next_token.dtype),
                        next_token,
                    )

            # Append sampled token
            next_token_unsqueezed = next_token.unsqueeze(1)
            all_input_ids = torch.cat([all_input_ids, next_token_unsqueezed], dim=1)
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((batch_size, 1), device=current_attention_mask.device, dtype=current_attention_mask.dtype),
                ],
                dim=1,
            )

            # Early stop if all sequences finished
            if eos_set is not None and torch.all(finished):
                break

            # Decode one step using cached states; pass base-stream tensors
            kv_cache_index = [torch.tensor([-1, 0], dtype=torch.long).repeat(1, 1).unsqueeze(0).to(all_input_ids.device)]

            decode_output = self.forward(
                kv_cache_index=kv_cache_index,
                input_ids=next_token_unsqueezed,
                attention_mask=current_attention_mask,
                position_ids=None,
                past_key_values=current_past,
                use_cache=True,
                *args,
                **kwargs,
            )
            current_past = decode_output.past_key_values
            last_logits = decode_output.logits[:, -1, :]

        # Return style compatible with HF generate
        if return_dict_in_generate:
            if GreedySearchDecoderOnlyOutput is not None and SampleDecoderOnlyOutput is not None:
                if do_sample:
                    return SampleDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
                else:
                    return GreedySearchDecoderOnlyOutput(
                        sequences=all_input_ids,
                        scores=scores if collect_scores else None,
                    )
            # Fallback to generic ModelOutput
            result = {"sequences": all_input_ids}
            if collect_scores:
                result["scores"] = scores
            return ModelOutput(**result)
        return all_input_ids