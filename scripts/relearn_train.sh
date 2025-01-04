
dataset=$1
mode=$2 # obj or art

INNER_PARAMs="unet"
OUTER_PARAMs="xattn"
export MODEL_NAME="CompVis/stable-diffusion-v1-4"



DELTA_CKPT="results/${dataset}/delta_ckpt.pt"
if [ $mode = "obj" ]
then
    if [ ! -e "$DELTA_CKPT" ]; then
        echo "Erased ckpt does not exist: $DELTA_CKPT. Running erase.py"
        python mima_utils/erase.py --concepts $dataset --device '0' --concept_type 'obj'
    fi
elif [ $mode = "art" ]
then
    if [ ! -e "$DELTA_CKPT" ]; then
        echo "Erased ckpt does not exist: $DELTA_CKPT. Running erase.py"
        python mima_utils/erase.py --concepts $dataset --device '0' --concept_type 'art' --preserve_number 10
    fi
fi


OUTPUT_DIR="results/${dataset}"
accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance="${dataset}" \
  --full_concepts_list "assets/${mode}_concepts_list.json" \
  --with_prior_preservation \
  --prior_loss_weight=1.0 \
  --num_class_images=200 \
  --resolution=512  \
  --train_batch_size=1  \
  --learning_rate_inner=5e-6  \
  --learning_rate_outer=3e-5  \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --scale_lr \
  --hflip  \
  --save_steps 100 \
  --output_dir=$OUTPUT_DIR \
  --param_names_to_optmize $INNER_PARAMs \
  --imma_param_names_to_optmize $OUTER_PARAMs \
  --delta_ckpt "${DELTA_CKPT}" \
  --max_train_samples 20 
