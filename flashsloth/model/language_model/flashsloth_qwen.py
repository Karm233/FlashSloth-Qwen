from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

from .qwen2.modeling_qwen2 import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


class FlashslothQwen2Config(Qwen2Config):
    model_type = "flashsloth_qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class FlashslothQwen2Model(LlavaMetaModel, Qwen2Model):
    config_class = FlashslothQwen2Config

    def __init__(self, config: FlashslothQwen2Config):
        super(FlashslothQwen2Model, self).__init__(config)


class FlashslothQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = FlashslothQwen2Config

    def __init__(self, config):
        super(FlashslothQwen2ForCausalLM, self).__init__(config)
        self.model = FlashslothQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.need_clear_cache = False
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        learnable_tokens = self.model.get_learnabletoken()
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                insert_place, 
                image_features, 
                learnable_token_len,
                modal,
                question_token_ranges
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                learnable_tokens,
                'qwen2.5'
            )
    # loss function
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            insert_place=insert_place,
            image_features=image_features,
            learnable_token_len=learnable_token_len,
            modal = modal,
            question_token_ranges = question_token_ranges
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("flashsloth_qwen2", FlashslothQwen2Config)
AutoModelForCausalLM.register(FlashslothQwen2Config, FlashslothQwen2ForCausalLM)