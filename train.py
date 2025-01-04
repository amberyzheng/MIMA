import sys
import argparse
import warnings
import hashlib
import itertools
import logging
import math
import os
from pathlib import Path
from typing import Optional
import torch
import json
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging import version

import transformers
import diffusers
from accelerate.logging import get_logger
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, DPMSolverMultistepScheduler, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
from diffusers.models.attention import Attention
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version, is_wandb_available

# import retrieve
from model_pipeline import MIMAAttnProcessor, MIMAPipeline, set_use_memory_efficient_attention_xformers
from data_pipeline import MIMADataset_Compose, PromptDataset, collate_fn_compose
from mima_utils import *
from composenW import compose_in_train

from memory_profiler import profile


logger = get_logger(__name__)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="IMMA-Pro training script.")
    parser.add_argument(
        "--delta_ckpt",
        type=str,
        default=None,
        help=(
            "The model ckpt with target concept erased."
        ),
    )
    parser.add_argument(
        "--param_names_to_optmize",
        type=str,
        default='xattn_kv',
        help=(
            "The parameters to train in the inner loop."
        ),
    )
    parser.add_argument(
        "--imma_param_names_to_optmize",
        type=str,
        default='xattn',
        help=(
            "The parameters to train in the outer loop."
        ),
    )
    parser.add_argument(
        "--instance",
        type=str,
        default='pet_cat1',
        help=(
            "The target instance name (image dir name) to defend."
        ),
    )
    parser.add_argument(
        "--full_concepts_list",
        type=str,
        default='data/full_concepts_list.json',
        help=(
            "The file containing all concepts."
        ),
    )
    parser.add_argument(
        "--sam_type",
        type=str,
        default=None,
        required=False,
        help="Algorithm of SAM to use in outer loop. If None, then use original SGD",
    )
    parser.add_argument("--sam_rho", type=float, default=0.05, help="The rho parameter for the SAM/ESAM optimizer.")
    parser.add_argument("--esam_beta", type=float, default=0.9, help="The beta parameter for the ESAM optimizer.")
    parser.add_argument("--esam_gamma", type=float, default=0.9, help="The gamma parameter for the ESAM optimizer.")
    parser.add_argument(
        "--is_asam",
        default=False,
        action="store_true",
        help="use adaptive ESAM",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--instance_prompt_file",
        type=str,
        default=None,
        help="The file containing prompts specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--validation_prompt_file",
        type=str,
        default=None,
        help="A file of prompts that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=2,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument(
        "--real_prior",
        default=False,
        action="store_true",
        help="real images as prior.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=200,
        help=(
            "Minimal class images for prior preservation loss. If there are not enough images already present in"
            " class_data_dir, additional images will be sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments",
        help="The output directory where the model predictions, checkpoints and validation generated images will be saved.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.5,
        help=(
            "Lambda in model merging optimization function."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=250,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--learning_rate_inner",
        type=float,
        default=1e-5,
        help="Initial learning rate for inner fine-tuning training.",
    )
    parser.add_argument(
        "--learning_rate_outer",
        type=float,
        default=1e-5,
        help="Initial learning rate for immunization training.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--prior_generation_precision",
        type=str,
        default=None,
        choices=["no", "fp32", "fp16", "bf16"],
        help=(
            "Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32."
        ),
    )
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument(
        "--inner_loop_steps", type=int, default=1, help=("Number of training steps for inner loop before doing outer loop training")
    )
    parser.add_argument(
        "--outer_loop_steps", type=int, default=1, help=("Number of training steps for outer loop after doing inner loop training")
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--modifier_token",
        type=str,
        default=None,
        help="A token to use as a modifier for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default="ktn+pll+ucd", help="A token to use as initializer word."
    )
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument(
        "--noaug",
        action="store_true",
        help="Dont apply augmentation during data augmentation when this flag is enabled.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.with_prior_preservation:
        if args.concepts_list is None and args.full_concepts_list is None:
            if args.class_data_dir is None:
                raise ValueError("You must specify a data directory for class images.")
            if args.class_prompt is None:
                raise ValueError("You must specify prompt for class images.")
    else:
        # logger is not available yet
        if args.class_data_dir is not None:
            warnings.warn("You need not use --class_data_dir without --with_prior_preservation.")
        if args.class_prompt is not None:
            warnings.warn("You need not use --class_prompt without --with_prior_preservation.")

    return args



def freeze_params(params):
    for param in params:
        param.requires_grad_(False)

def enable_params(params):
    for param in params:
        param.requires_grad_(True)



##### for debug only ######
def check_alive_param(params):
    param_list = []
    for name, param in params:
        if param.requires_grad == True:
            param_list.append(name)
    return param_list

def check_param_with_grad(params):
    param_list = []
    for name, param in params:
        if param.grad:
            param_list.append(name)
    return param_list


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")

@profile
def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=logging_dir
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Currently, it's not possible to do gradient accumulation when training two models with accelerate.accumulate
    # This will be enabled soon in accelerate. For now, we don't allow gradient accumulation when training two models.
    # TODO (patil-suraj): Remove this check when gradient accumulation with two models is enabled in accelerate.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)
    if args.full_concepts_list is not None and args.instance is not None:
        with open(args.full_concepts_list, "r") as f:
            full_concepts_list = json.load(f)
        instances = args.instance.split('+')
        args.concepts_list = []
        for instance in instances:
            args.concepts_list.extend([concept_dict for concept_dict in full_concepts_list if instance in concept_dict['instance_data_dir']])
    else:
        if args.concepts_list is None:
            args.concepts_list = [
                {
                    "instance_prompt": args.instance_prompt,
                    "class_prompt": args.class_prompt,
                    "instance_data_dir": args.instance_data_dir,
                    "class_data_dir": args.class_data_dir
                }
            ]
        else:
            with open(args.concepts_list, "r") as f:
                args.concepts_list = json.load(f)

    if args.with_prior_preservation:
        for i, concept in enumerate(args.concepts_list):
            class_images_dir = Path(concept['class_data_dir'])
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True, exist_ok=True)

            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < args.num_class_images:
                torch_dtype = torch.float16 if accelerator.device.type == "cuda" else torch.float32
                if args.prior_generation_precision == "fp32":
                    torch_dtype = torch.float32
                elif args.prior_generation_precision == "fp16":
                    torch_dtype = torch.float16
                elif args.prior_generation_precision == "bf16":
                    torch_dtype = torch.bfloat16
                pipeline = DiffusionPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    revision=args.revision,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = args.num_class_images - cur_class_images
                logger.info(f"Number of class images to sample: {num_new_images}.")

                sample_dataset = PromptDataset(concept['class_prompt'], num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

                sample_dataloader = accelerator.prepare(sample_dataloader)
                pipeline.to(accelerator.device)

                for example in tqdm(
                    sample_dataloader, desc="Generating class images", disable=not accelerator.is_local_main_process
                ):
                    images = pipeline(example["prompt"], num_inference_steps=50, guidance_scale=6., eta=1.).images

                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            revision=args.revision,
            use_fast=False,
        )
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
    )

    # add erasing checkpoint
    if args.delta_ckpt is not None:
        print("restroting from erased model from pre-training")
        unet.load_state_dict(torch.load(args.delta_ckpt, map_location=accelerator.device))

    # Get a list of parameter group names for training
    param_names_to_optmize = args.param_names_to_optmize.split('+')

    vae.requires_grad_(False)
    if 'text_encoder' not in param_names_to_optmize and args.modifier_token is None:
        text_encoder.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    if accelerator.mixed_precision != "fp16":
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    ## check this##
    
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder or args.modifier_token is not None:
            text_encoder.gradient_checkpointing_enable()
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate_inner = (
            args.learning_rate_inner * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
        if args.with_prior_preservation:
            args.learning_rate_inner = args.learning_rate_inner*2.

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Adding a modifier token which is optimized ####
    # Code taken from https://github.com/huggingface/diffusers/blob/main/examples/textual_inversion/textual_inversion.py
    modifier_token_id = []
    initializer_token_id = []
    params_to_optimize_inner_dict = {}
    if args.modifier_token is not None:
        args.modifier_token = args.modifier_token.split('+')
        args.initializer_token = args.initializer_token.split('+')
        if len(args.modifier_token) > len(args.initializer_token):
            raise ValueError("You must specify + separated initializer token for each modifier token.")
        for modifier_token, initializer_token in zip(args.modifier_token, args.initializer_token[:len(args.modifier_token)]):
            # Add the placeholder token in tokenizer
            num_added_tokens = tokenizer.add_tokens(modifier_token)
            if num_added_tokens == 0:
                raise ValueError(
                    f"The tokenizer already contains the token {modifier_token}. Please pass a different"
                    " `modifier_token` that is not already in the tokenizer."
                )

            # Convert the initializer_token, placeholder_token to ids
            token_ids = tokenizer.encode([initializer_token], add_special_tokens=False)
            print(token_ids)
            # Check if initializer_token is a single token or a sequence of tokens
            if len(token_ids) > 1:
                raise ValueError("The initializer token must be a single token.")

            initializer_token_id.append(token_ids[0])
            modifier_token_id.append(tokenizer.convert_tokens_to_ids(modifier_token))

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for (x,y) in zip(modifier_token_id,initializer_token_id):
            token_embeds[x] = token_embeds[y]

        # Freeze all parameters except for the token embeddings in text encoder
        params_to_freeze = itertools.chain(
            text_encoder.text_model.encoder.parameters(),
            text_encoder.text_model.final_layer_norm.parameters(),
            text_encoder.text_model.embeddings.position_embedding.parameters(),
        )
        freeze_params(params_to_freeze)

        for name, param in text_encoder.named_parameters():
                if 'embeddings' in name:
                    params_to_optimize_inner_dict[name] = param
    else:
        params_to_optimize_inner_dict = {}
        for name, param in unet.named_parameters():
            if 'attn2.to_k' in name or 'attn2.to_v' in name:
                params_to_optimize_inner_dict[name] = param
        if 'xattn' in param_names_to_optmize:
            params_to_optimize_inner = [x[1] for x in unet.named_parameters() if ('attn2.to_q' in x[0] or 'attn2.to_out' in x[0])]
            
        elif 'unet' in param_names_to_optmize:
            params_to_optimize_inner = [x[1] for x in unet.named_parameters() if ('attn2.to_k' not in x[0] and 'attn2.to_v' not in x[0])] 
        else:
            ValueError

        if 'text_encoder' in param_names_to_optmize:
            params_to_optimize_inner_dict = {x[0]:x[1] for x in text_encoder.named_parameters()}
    
    if args.imma_param_names_to_optmize == 'xattn':
        params_to_optimize_outer = [x[1] for x in unet.named_parameters() if ('attn2' in x[0])]
    elif args.imma_param_names_to_optmize == 'xattn_kv':
        params_to_optimize_outer = [x[1] for x in unet.named_parameters() if ('attn2.to_k' in x[0] or 'attn2.to_v' in x[0])]

    optimizer_inner = optimizer_class(
        params_to_optimize_inner,
        lr=args.learning_rate_inner,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    optimizer_outer = optimizer_class(
            params_to_optimize_outer,
            lr=args.learning_rate_outer,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )


    train_dataset = MIMADataset_Compose(
        concepts_list=args.concepts_list,
        tokenizer=tokenizer,
        with_prior_preservation=args.with_prior_preservation,
        size=args.resolution,
        center_crop=args.center_crop,
        num_class_images=args.num_class_images,
        hflip=args.hflip,
        max_train_samples=args.max_train_samples
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        collate_fn=lambda examples: collate_fn_compose(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler_inner = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_inner,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    lr_scheduler_outer = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_outer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder or args.modifier_token is not None:
        unet, text_encoder, optimizer_inner, train_dataloader, lr_scheduler_inner = accelerator.prepare(
            unet, text_encoder, optimizer_inner, train_dataloader, lr_scheduler_inner
        )
    else:
        unet, optimizer_inner, train_dataloader, lr_scheduler_inner = accelerator.prepare(
            unet, optimizer_inner, train_dataloader, lr_scheduler_inner
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("custom-diffusion")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder or args.modifier_token is not None:
            text_encoder.train()
        for step, batches in enumerate(train_dataloader):
            
            # batches is a dict of multiple concept dicts
            composed_weight_list = []
            categories = []
            is_inner = (step + epoch * len(train_dataloader)) % (args.inner_loop_steps + args.outer_loop_steps) < args.inner_loop_steps
            outer_loss = 0.0
            num_concepts = len(batches)
            inner_other_grad = [torch.zeros_like(param) for param in params_to_optimize_inner] # initialize the inner_other_grad, then aggregate the grad for each concept
            # for outer loop, run separate updates for each concept, i.e. no weights optimization
            for class_name, batch in batches.items():
                if is_inner:
                    freeze_params(params_to_optimize_outer)
                    enable_params(params_to_optimize_inner_dict.values())
                    enable_params(params_to_optimize_inner)
                else:
                    freeze_params(params_to_optimize_inner)
                    freeze_params(params_to_optimize_inner_dict.values())
                    enable_params(params_to_optimize_outer)
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    instance_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                    if is_inner:
                        # Add the prior loss to the instance loss.
                        loss = instance_loss.mean() + args.prior_loss_weight * prior_loss
                    else:
                        loss =  - instance_loss.mean() + args.prior_loss_weight * prior_loss

                    # Zero out the gradients for all token embeddings except the newly added
                    # embeddings for the concept, as we only want to optimize the concept embeddings
                    if args.modifier_token is not None:
                        if accelerator.num_processes > 1:
                            grads_text_encoder = text_encoder.module.get_input_embeddings().weight.grad
                        else:
                            grads_text_encoder = text_encoder.get_input_embeddings().weight.grad
                        # Get the index for tokens that we want to zero the grads for
                        index_grads_to_zero = torch.arange(len(tokenizer)) != modifier_token_id[0]
                        for i in range(len(modifier_token_id[1:])):
                            index_grads_to_zero = index_grads_to_zero & (torch.arange(len(tokenizer)) != modifier_token_id[i])
                        grads_text_encoder.data[index_grads_to_zero, :] = grads_text_encoder.data[index_grads_to_zero, :].fill_(0)

                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain([x[1] for x in unet.named_parameters() if ('attn2' in x[0])], text_encoder.parameters())
                            if (args.train_text_encoder or args.modifier_token is not None)
                            else itertools.chain([x[1] for x in unet.named_parameters() if ('attn2' in x[0])]) 
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    
                    # Do real update
                    if is_inner:
                        # Unroll inner loss manually
                        grads = torch.autograd.grad(loss, list(params_to_optimize_inner_dict.values()) + params_to_optimize_inner)
                        # Split the gradients
                        inner_grad = grads[:len(params_to_optimize_inner_dict)]
                        inner_other_g = grads[len(params_to_optimize_inner_dict):]
                        
                        new_weights =  {key: param - args.learning_rate_inner * g for g, param, key in zip(inner_grad, params_to_optimize_inner_dict.values(), params_to_optimize_inner_dict.keys())}
                        composed_weight_list.append(new_weights)
                        categories.append(class_name)
                        # grad for other params
                        for i, param in enumerate(params_to_optimize_inner):
                            inner_other_grad[i] += inner_other_g[i].detach() / num_concepts
                    else:
                        outer_loss += loss / num_concepts
                        

            # compose weights and update
            if is_inner:
                # update xattn key and value
                composed_weight = compose_in_train(unet, text_encoder, tokenizer, composed_weight_list, categories, float(args.lam))
                for name, params in unet.named_parameters():
                    if 'attn2.to_k' in name or 'attn2.to_v' in name:
                        new_param = composed_weight[name].to(accelerator.device)
                        with torch.no_grad():
                            params.copy_(new_param)
                # update the other params
                for i, param in enumerate(params_to_optimize_inner):
                    with torch.no_grad():
                        param.grad = inner_other_grad[i]
                optimizer_inner.step()
                lr_scheduler_inner.step()
                optimizer_inner.zero_grad(set_to_none=args.set_grads_to_none)
            else:
                accelerator.backward(outer_loss)
                accelerator.clip_grad_norm_(params_to_optimize_outer, args.max_grad_norm)
                optimizer_outer.step()
                lr_scheduler_outer.step()
                optimizer_outer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        pipeline = MIMAPipeline.from_pretrained(
                            args.pretrained_model_name_or_path,
                            unet=accelerator.unwrap_model(unet),
                            text_encoder=accelerator.unwrap_model(text_encoder),
                            tokenizer=tokenizer,
                            revision=args.revision,
                            modifier_token=args.modifier_token,
                            modifier_token_id=modifier_token_id,
                        )
                        save_dir = os.path.join(args.output_dir, 'weights', args.param_names_to_optmize+"_"+args.imma_param_names_to_optmize, "adam" if args.lam == 0.5 else "lam_"+str(args.lam))
                        os.makedirs(save_dir, exist_ok=True)
                        save_path = os.path.join(save_dir, f"imma_pro_ckpt-lr{args.learning_rate_outer}-{global_step}.pt")
                        pipeline.save_pretrained(save_path)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler_inner.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)