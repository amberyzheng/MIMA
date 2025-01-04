#!/bin/bash

instance=$1

export MODEL_NAME="CompVis/stable-diffusion-v1-4"

export OUTPUT_DIR="results/${instance}"
accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance=$instance \
  --full_concepts_list assets/full_concepts_list.json \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --num_class_images=2 \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate_inner=5e-6  \
  --learning_rate_outer=2e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --scale_lr \
  --hflip  \
  --save_steps 100 \
  --output_dir=$OUTPUT_DIR \
  --param_names_to_optmize unet \
  --imma_param_names_to_optmize xattn