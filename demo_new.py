import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import gradio as gr
from flashsloth.constants import *
from flashsloth.conversation import conv_templates
from flashsloth.model.builder import load_pretrained_model
from flashsloth.utils import disable_torch_init
from flashsloth.mm_utils import *
from PIL import Image
from transformers import GenerationConfig
from threading import Thread
from transformers import TextIteratorStreamer

def deal_logits_processor(generation_config):
    generation_config.guidance_scale = None
    generation_config.sequence_bias = None
    generation_config.diversity_penalty = None
    generation_config.encoder_repetition_penalty = None
    generation_config.repetition_penalty = None
    generation_config.no_repeat_ngram_size = None
    generation_config.encoder_no_repeat_ngram_size = None
    generation_config.bad_words_ids = None
    generation_config.min_length = None
    generation_config.min_new_tokens = None
    generation_config.forced_bos_token_id = None
    generation_config.forced_eos_token_id = None
    generation_config.remove_invalid_values = False
    generation_config.exponential_decay_length_penalty = None
    generation_config.suppress_tokens = None
    generation_config.begin_suppress_tokens = None
    generation_config.forced_decoder_ids = None
    generation_config.renormalize_logits = False
    return generation_config

# 模型加载函数
def load_models():
    disable_torch_init()
    model_path = "/mnt/82_store/luogen/tb/rebuttal/FlashSloth_qwen/checkpoints/flashslothqwen-cn_HD"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, None, model_name)
    model = model.to('cuda').to(torch.bfloat16)
    
    # 准备生成配置
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config = deal_logits_processor(generation_config)
    print('-----all model are loaded-----')
    return tokenizer, model, image_processor, generation_config

tokenizer, model, image_processor, generation_config = load_models()



def process_inputs(image_path, input_text, chat_history):
    try:
        
        text = input_text or ""

        # 处理图片
        image = Image.open(image_path).convert('RGB')
        if model.config.image_hd:
            image_tensor = process_images_hd_inference([image], image_processor, model.config)[0]
        else:
            image_tensor = process_images([image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.bfloat16, device='cuda')

        text = DEFAULT_IMAGE_TOKEN + '\n' + text + LEARNABLE_TOKEN
        conv = conv_templates["qwen2"].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        stopping_criteria = KeywordsStoppingCriteria(['<|endoftext|>','<|im_end|>'], tokenizer, input_ids)
        generation_kwargs = dict(
            input_ids=input_ids,
            images=image_tensor,
            max_new_tokens=2048,
            use_cache=True,
            do_sample=False,
            eos_token_id=[151643,151645],
            stopping_criteria=[stopping_criteria],
            generation_config=generation_config,
            streamer=streamer
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        # 初始化响应内容
        outputs = ""
        chat_history.append((text, outputs))
        yield chat_history

        # 流式输出文本
        for new_text in streamer:
            outputs += new_text
            chat_history[-1] = (chat_history[-1][0], outputs)
            yield chat_history

        # 生成完成后合成语音
        chat_history[-1] = (chat_history[-1][0], outputs.strip())
        yield chat_history

    except Exception as e:
        chat_history.append((text, f"Error: {str(e)}"))
        yield chat_history

# CSS和界面部分保持不变...
css = """
    .custom-chatbot {
        height: 400px;
    }
    .custom-audio-output {
        height: 80px;
    }
    .custom-audio-input {
        height: 230px;
    }
    .custom-image {
        height: 500px;
    }
    .custom-textbox {
        height: 250px;
    }
"""

# 创建Gradio界面
with gr.Blocks(css=css,title="多模态交互系统") as demo:
    gr.Markdown("## 多模态交互系统（图文语音交互）")
    
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="对话历史",elem_classes="custom-chatbot")
            
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="上传图片", sources=["webcam","upload","clipboard"],elem_classes="custom-image")
        with gr.Column():
            text_input = gr.Textbox(label="文本输入", placeholder="或在此输入文本...",lines=9, elem_classes="custom-textbox")
    with gr.Row():
        submit_btn = gr.Button("提交", variant="primary")

    submit_btn.click(
        fn=process_inputs,
        inputs=[image_input, text_input, chatbot],
        outputs=[chatbot]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)