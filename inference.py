import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from flashsloth.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, LEARNABLE_TOKEN, LEARNABLE_TOKEN_INDEX
from flashsloth.conversation import conv_templates, SeparatorStyle
from flashsloth.model.builder import load_pretrained_model
from flashsloth.utils import disable_torch_init
from flashsloth.mm_utils import tokenizer_image_token, process_images, process_images_hd_inference, get_model_name_from_path, KeywordsStoppingCriteria
from PIL import Image
from transformers import GenerationConfig

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

def main():
    image_path = "/data/tb/rebuttal/FlashSloth/images/1733320415394.jpg"
    text = "Describe this photo in detail."
    model_path = "/data/tb/rebuttal/FlashSloth/checkpoints/flashslothqwen-fft_llava665k"
    generation_config = GenerationConfig.from_pretrained(model_path)
    generation_config = deal_logits_processor(generation_config)
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.to('cuda').to(torch.bfloat16)
    torch.set_printoptions(threshold=torch.inf)
    keywords = ['<|endoftext|>']
    text = DEFAULT_IMAGE_TOKEN + '\n' + text
    text = text + LEARNABLE_TOKEN
    image = Image.open(image_path).convert('RGB')
    if model.config.image_hd:
        image_tensor = process_images_hd_inference([image], image_processor, model.config)[0]
    else:
        image_tensor = process_images([image], image_processor, model.config)[0]
    image_tensor = image_tensor.unsqueeze(0)
    conv = conv_templates["qwen2"].copy()
    conv.append_message(conv.roles[0], text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    print(prompt)
    # Tokenize text
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device='cuda', non_blocking=True)
    print(input_ids)
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.bfloat16, device='cuda', non_blocking=True),
            max_new_tokens=1024,
            use_cache=True,
            do_sample=False,
            temperature=None, 
            top_p=None, 
            eos_token_id=151645,
            stopping_criteria=[stopping_criteria],
            generation_config=generation_config
        )
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    print('output_ids:',output_ids[:, input_token_len:])
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    print(outputs)

if __name__ == "__main__":
    main()
