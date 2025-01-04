import importlib
import torch
from PIL import Image

import torchvision
from torchvision import transforms

from diffusers import DiffusionPipeline

def load_imma_pro_ckpt(unet, imma_ckpt):
    print("restroting from trained IMMA pro weights")
    model_dict = unet.state_dict()
    model_dict.update(torch.load(imma_ckpt))
    unet.load_state_dict(model_dict)

def log_validation(
    text_encoder,
    tokenizer,
    unet,
    vae,
    args,
    accelerator,
    weight_dtype,
    global_step,
    prompt_embeds=None,
    negative_prompt_embeds=None,
):
    
    image_save_transform = transforms.Compose([transforms.ToTensor()])
  

    pipeline_args = {}

    if vae is not None:
        pipeline_args["vae"] = vae

    if text_encoder is not None:
        text_encoder = accelerator.unwrap_model(text_encoder)

    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        unet=accelerator.unwrap_model(unet),
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        safety_checker = None,
        **pipeline_args,
    )

    # We train on the simplified learning objective. If we were previously predicting a variance, we need the scheduler to ignore it
    scheduler_args = {}

    if "variance_type" in pipeline.scheduler.config:
        variance_type = pipeline.scheduler.config.variance_type

        if variance_type in ["learned", "learned_range"]:
            variance_type = "fixed_small"

        scheduler_args["variance_type"] = variance_type

    module = importlib.import_module("diffusers")
    scheduler_class = getattr(module, args.validation_scheduler)
    pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config, **scheduler_args)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.pre_compute_text_embeddings:
        pipeline_args = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }
    else:
        pipeline_args = {"prompt": args.validation_prompt}

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(**pipeline_args, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    for idx, image in enumerate(images):
        image_save_name = f"{global_step}_{idx}.png".zfill(11)
        torchvision.utils.save_image(image_save_transform(image), f"{args.output_dir}/{image_save_name}")

    del pipeline
    torch.cuda.empty_cache()

    return images