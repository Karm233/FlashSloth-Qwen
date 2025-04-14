import argparse
import time
from PIL import Image

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Tuple, Any
import re

import requests
from threading import Thread

ERROR_CODE = [260, 1794, 11440]
ERROR_MESSAGE = [1, 7423, 29892, 474, 508, 29915, 29873, 1234, 445, 1139, 29889, 2]


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = ""
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False
def expand2square(pil_img, background_color=(0, 0, 0)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

CONV_VISION = Conversation(
    system="",
    roles=("Instruction", "Responese"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
)

# conv_llava_siyuan = Conversation(
#     system= "你是思源，一个由厦门大学、北京大学深圳研究生院、合肥综合性国家科学中心人工智能研究院（安徽省人工智能实验室）、安徽淘云科技股份有限公司合作研发的人工智能助手。在保证安全的前提下，回答问题要尽可能有帮助。你的答案不应该包含任何有害的、不道德的、种族主义的、性别歧视的、有毒的、危险的或非法的内容。请确保你的回答在社会上是公正和积极的。如果一个问题没有任何意义，或者与事实不一致，解释为什么，而不是回答不正确的问题。如果你不知道问题的答案，请不要分享虚假信息。",
#     roles=("USER", "ASSISTANT"),
#     version="llama_v2",
#     messages=(),
#     offset=0,
#     sep_style=SeparatorStyle.LLAMA_2,
#     sep="<s>",
#     sep2="</s>",
# )


class Chat:
    def __init__(self, vis_processor):
        # self.device = device
        # self.lavin = model
        self.vis_processor = vis_processor
        self.worker0 = 'http://127.0.0.1:7680/predict'
        self.worker1 = 'http://127.0.0.1:7681/predict'
        # self.worker1 = 'http://127.0.0.1:7680/predict'

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000,n_feats=256):
        conv.append_message(conv.roles[1], None)

        prompt, indicator, img = self.get_context_emb(conv, img_list)

        # print(prompt)

        current_max_len = len(prompt) + max_new_tokens+n_feats
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        data = {
            'begin_idx':begin_idx + 1,
            'prompt':prompt,
            'indicator':indicator,
            'max_length':max_length,
            'n_feats':n_feats
        }

        # mthread = Thread(target=requests.get, args=(self.worker1, data)) 
        # mthread.start()
        response = requests.post(self.worker0, data)

        # prompt = prompt[begin_idx:]
        # CODE=self.lavin.tokenizer.encode(prompt, bos=False, eos=False)
        # if ERROR_CODE in [CODE[i:i+len(ERROR_CODE)] for i in range(len(CODE)-len(ERROR_CODE)+1)]:
        #     output_text=self.lavin.tokenizer.decode(ERROR_MESSAGE).split('Responese:')[-1].strip()
        # else:
        #     outputs = self.lavin.generate(
        #         prompts= [prompt],
        #         images= img,
        #         indicators=[indicator],
        #         max_gen_len=max_length,
        #         n_feats=n_feats,
        #         temperature = 0.1,
        #         top_p = 0.75,
        #     )

        #     output_text = outputs[0].split('Responese:')[-1].strip()

        output_text = response.text[2:-3]

        conv.messages[-1][1] = output_text
        return output_text

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            raw_image = expand2square(raw_image)
            image = self.vis_processor(raw_image).unsqueeze(0)#.to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            raw_image = expand2square(raw_image)
            image = self.vis_processor(raw_image).unsqueeze(0)#.to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image#.to(self.device)

        raw_image.save('./tmp.png')

        # image_emb, _ = self.lavin.backbone.encode_img(image)
        img_list.append(image)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg, img_list

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()

        if '<Img><ImageHere></Img>' in prompt:
            indicator=1
            prompt=prompt.replace('<Img><ImageHere></Img>','')
        else:
            indicator=0
        assert img_list is None or len(img_list) <=  1

        return prompt, indicator, img_list[0] if indicator==1 else torch.Tensor(torch.zeros(1,3, 224, 224).float())