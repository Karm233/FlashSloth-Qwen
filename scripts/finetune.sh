#!/bin/bash

# uncomment the following lines to shutoff the internet access
# export HF_HUB_OFFLINE=True
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
export FlashSloth_SILIENT_OTHERS=true
# if not use all GPUs 
# deepspeed --include localhost:0,1,2,3 --master_port 29600

deepspeed  flashsloth/train/train_finetune.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /mnt/82_store/luogen/tb/rebuttal/FlashSloth_qwen/checkpoints/flashslothqwen-cn-stage1 \
    --version qwen2 \
    --data_path /mnt/82_store/luogen/tb/data_cn/Instruct_all_9213k.json \
    --image_folder /mnt/82_store/luogen/images_aug \
    --vision_tower /mnt/82_store/luogen/tb/rebuttal/FlashSloth_siglip/checkpoints/siglip \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --group_by_modality_length True \
    --bf16 False \
    --fp16 True \
    --output_dir ./checkpoints/flashslothqwen-cn_HD \
    --num_train_epochs 1 \
    --per_device_train_batch_size 10 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 6000\
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --image_hd True