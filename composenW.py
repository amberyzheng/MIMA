import sys
import os
import argparse
import torch
from torch.linalg import lu_factor, lu_solve

from pathlib import Path
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.append('./')
sys.path.append('../')
from diffusers import StableDiffusionPipeline, AutoencoderKL
from sample import sample

from datasets import load_dataset
import random

import pdb


def gdupdateWexact(K, Ktarget1, Vtarget1, W, lam=None, device='cuda'):
    input_ = K # C_reg
    C = input_.T@input_ # C_reg.T @ C_reg
    d = []
    lu, piv = lu_factor(C)
    for i in range(Ktarget1.size(0)):
        sol = lu_solve(lu, piv, Ktarget1[i].reshape(-1, 1))
        d.append(sol.to(K.device))

    d = torch.cat(d, 1).T # C @ (C_reg.T @ C_reg)^-1

    e2 = d@Ktarget1.T
    e1 = (Vtarget1.T - W@Ktarget1.T)
    delta = e1@torch.linalg.inv(e2) # value of inverse is too large when size of Ktarget1 is large (numerical problem!!)

    Wnew = W + delta@d

    return Wnew


def get_image_embedding(image_paths, pipe, vae):
    size=128
    conv_in = pipe.unet.conv_in
    norm_in = pipe.unet.down_blocks[0].attentions[0].norm
    proj_in = pipe.unet.down_blocks[0].attentions[0].proj_in
    
    image_transforms = transforms.Compose(
        [   
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    with torch.no_grad():
        uc = []
        for image_path in tqdm(image_paths):
            image = Image.open(image_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            image_feature = norm_in(conv_in(vae.encode(image_transforms(image).to(pipe.device).unsqueeze(0)).latent_dist.sample()))
            batch, inner_dim, height, width = image_feature.shape
            uc.append(proj_in(image_feature.permute(0, 2, 3, 1).reshape(batch * height * width, inner_dim)))
    return torch.cat(uc, 0).float()


def get_text_embedding(prompts, pipe):
    tokenizer = pipe.tokenizer
    pipe.text_encoder.resize_token_embeddings(len(tokenizer))

    uc = []
    for text in prompts:
        tokens = tokenizer(text,
                            truncation=True,
                            max_length=tokenizer.model_max_length,
                            return_length=True,
                            return_overflowing_tokens=False,
                            padding="do_not_pad",
                            ).input_ids
        if 'photo of a' in text[:15]:
            uc.append(pipe.text_encoder(torch.cuda.LongTensor(tokens).reshape(1,-1))[0][:, 4:].reshape(-1, 1024))
        else:
            uc.append(pipe.text_encoder(torch.cuda.LongTensor(tokens).reshape(1,-1))[0][:, 1:].reshape(-1, 1024))
    return torch.cat(uc, 0).float()


# function for composition during inference
def compose(paths, category, target_image_path, outpath, pretrained_model_path, prompts, regularization_image_path, save_path, device='cuda:0'):
    model_id = pretrained_model_path
    outpath = os.path.join(outpath, *paths.split('+')[0].split('/')[2:-1])
    ckpt_name = paths.split('+')[0].split('/')[-1]
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    # parts for image embeddings
    

    text_layers_modified = []
    image_layers_modified = []
    for name, _ in pipe.unet.named_parameters():
        if 'attn2.to_k' in name or 'attn2.to_v' in name:
            text_layers_modified.append(name)
    
    tokenizer = pipe.tokenizer

    def _get_text_embedding(prompts):
        with torch.no_grad():
            uc = []
            for text in prompts:
                tokens = tokenizer(text,
                                   truncation=True,
                                   max_length=tokenizer.model_max_length,
                                   return_length=True,
                                   return_overflowing_tokens=False,
                                   padding="do_not_pad",
                                   ).input_ids
                if 'photo of a' in text[:15]:
                    uc.append(pipe.text_encoder(torch.cuda.LongTensor(tokens).reshape(1,-1))[0][:, 4:].reshape(-1, 1024))
                else:
                    uc.append(pipe.text_encoder(torch.cuda.LongTensor(tokens).reshape(1,-1))[0][:, 1:].reshape(-1, 1024))
        return torch.cat(uc, 0).float()
    

    count = 1
    model2_sts = []
    categories = []
    target_image_paths = []
    for path1, cat1, img_path1 in zip(paths.split('+'), category.split('+'), target_image_path.split('+')):
        model2_st = torch.load(path1)
        # composition of models with individual concept only
        model2_sts.append(model2_st)
        categories.append(cat1)
        target_image_paths.append(img_path1)
        count += 1
    
    # get image_layers_modified
    for name in model2_st.keys():
        if 'attn2.to_q' in name:
            image_layers_modified.append(name)

    pipe.text_encoder.resize_token_embeddings(len(tokenizer))


    # load regularization prompts
    prompt = random.sample(load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS")['train']['caption'], 200)
    uc = _get_text_embedding(prompt) # C_reg

    # load regularization images
    uf = []
    for concept in categories:
        class_data_root = Path(os.path.join(regularization_image_path, "samples_{}".format(concept)))
        class_images_path = list(class_data_root.iterdir())[:50]
        uf.extend(class_images_path)
    print("Preparing regularization images for concept: {}".format(categories))
    uf = get_image_embedding(uf, pipe, vae)


    uc_targets = []
    uf_targets = []
    from collections import defaultdict
    uc_values = defaultdict(list)
    uf_values = defaultdict(list)
    for composing_model_count in range(len(model2_sts)):
        category = categories[composing_model_count]
        string1 = f'{category}'
        if 'art' in string1:
            prompt = [string1] + [f"painting in the style of {string1}"]
        else:
            prompt = [string1] + [f"photo of a {string1}"]
        uc_targets.append(_get_text_embedding(prompt)) # C

        image_path = Path(target_image_paths[composing_model_count])
        uf_targets.append(get_image_embedding(list(image_path.iterdir()), pipe, vae))

        for each in text_layers_modified:
            uc_values[each].append((model2_sts[composing_model_count][each].to(device)@uc_targets[-1].T).T) # W @ C.T = V
            # print(model2_sts[composing_model_count][each].shape)
        for each in image_layers_modified:
            uf_values[each].append((model2_sts[composing_model_count][each].to(device)@uf_targets[-1].T).T) # Q

            
    uc_targets = torch.cat(uc_targets, 0)
    uf_targets = torch.cat(uf_targets, 0)

    removal_indices = []
    removal_indices2 = []
    for i in range(uc_targets.size(0)):
        for j in range(i+1, uc_targets.size(0)):
            if (uc_targets[i]-uc_targets[j]).abs().mean() == 0:
                removal_indices.append(j)
    
    # for i in range(uf_targets.size(0)):
    #     for j in range(i+1, uf_targets.size(0)):
    #         if (uf_targets[i]-uf_targets[j]).abs().mean() == 0:
    #             removal_indices2.append(j)

    removal_indices = list(set(removal_indices))
    removal_indices2 = list(set(removal_indices2))
    uc_targets = torch.stack([uc_targets[i] for i in range(uc_targets.size(0)) if i not in removal_indices], 0)
    uf_targets = torch.stack([uf_targets[i] for i in range(uf_targets.size(0)) if i not in removal_indices2], 0)
    for each in text_layers_modified:
        uc_values[each] = torch.cat(uc_values[each], 0)
        uc_values[each] = torch.stack([uc_values[each][i] for i in range(uc_values[each].size(0)) if i not in removal_indices], 0)
        print(uc_values[each].size(), each)
    for each in image_layers_modified:
        uf_values[each] = torch.cat(uf_values[each], 0)
        uf_values[each] = torch.stack([uf_values[each][i] for i in range(uf_values[each].size(0)) if i not in removal_indices2], 0)
        print(uf_values[each].size(), each)

    new_weights = {}
    for each in text_layers_modified:
        W = pipe.unet.state_dict()[each].float()
        # values = (W@uc.T).T # W_0 @ C_reg.T
        input_target = uc_targets # C
        output_target = uc_values[each] # W @ C.T

        Wnew = gdupdateWexact(uc,
                              input_target,
                              output_target,
                              W.clone(),
                              )

        new_weights[each] = Wnew
    
    for each in image_layers_modified:
        W = pipe.unet.state_dict()[each].float()
        # values = uf@W.T
        input_target = uf_targets
        output_target = uf_values[each]

        Wnew = gdupdateWexact(uf,
                              input_target,
                              output_target,
                              W.clone(),
                              )

        new_weights[each] = Wnew


    os.makedirs(f'{save_path}/{outpath}', exist_ok=True)
    torch.save(new_weights, f'{save_path}/{outpath}/{ckpt_name}')
    print(f'Optimized weights saved at {save_path}/{outpath}/{ckpt_name}')

    if prompts is not None:
        if os.path.exists(prompts):
            sample(model_id, f'{save_path}/{outpath}/{ckpt_name}', prompts, prompt=None, compress=False, freeze_model='crossattn_kv', batch_size=1)
        else:
            sample(model_id, f'{save_path}/{outpath}/{ckpt_name}', from_file=None, prompt=prompts, compress=False, freeze_model='crossattn_kv', batch_size=1)



# prepare regularization prompts and images embeddings
def compose_prepare(pipe, vae, regularization_image_path, categories, device='cuda:0'):
    
    tokenizer = pipe.tokenizer
    pipe.text_encoder.resize_token_embeddings(len(tokenizer))

    # load regularization prompts
    prompt = random.sample(load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS")['train']['caption'], 200)
    uc = get_text_embedding(prompt, pipe, vae)

    # load regularization images
    uf = []
    for concept in categories:
        class_data_root = Path(os.path.join(regularization_image_path, "samples_{}".format(concept)))
        class_images_path = list(class_data_root.iterdir())[:50]
        uf.extend(class_images_path)
    print("Preparing regularization images for concept: {}".format(categories))
    uf = get_image_embedding(uf, pipe, vae)

    return uc, uf


# compose kv only in train
def compose_in_train(unet, text_encoder, tokenizer, model2_sts, categories, lam=0.5, device='cuda:0'):
    text_layers_modified = []
    for name, _ in unet.named_parameters():
        if 'attn2.to_k' in name or 'attn2.to_v' in name:
            text_layers_modified.append(name)
    
    def get_text_embedding(prompts):
        uc = []
        for text in prompts:
            tokens = tokenizer(text,
                                truncation=True,
                                max_length=tokenizer.model_max_length,
                                return_length=True,
                                return_overflowing_tokens=False,
                                padding="do_not_pad",
                                ).input_ids
            if 'photo of a' in text[:15]:
                uc.append(text_encoder(torch.cuda.LongTensor(tokens).reshape(1,-1))[0][:, 4:].reshape(-1, 768))
            else:
                uc.append(text_encoder(torch.cuda.LongTensor(tokens).reshape(1,-1))[0][:, 1:].reshape(-1, 768))
        return torch.cat(uc, 0).float()

    with torch.no_grad():
        text_encoder.resize_token_embeddings(len(tokenizer))


    # load regularization prompts
    prompt = random.sample(load_dataset("laion/220k-GPT4Vision-captions-from-LIVIS")['train']['caption'], 200)
    with torch.no_grad():
        uc = get_text_embedding(prompt) 


    uc_targets = []
    from collections import defaultdict
    uc_values = defaultdict(list)
    for composing_model_count in range(len(model2_sts)):
        category = categories[composing_model_count]
        string1 = f'{category}'
        if 'art' in string1:
            prompt = [string1] + [f"painting in the style of {string1}"]
        else:
            prompt = [string1] + [f"photo of a {string1}"]
        uc_targets.append(get_text_embedding(prompt))
        
        for each in text_layers_modified:
            uc_values[each].append((model2_sts[composing_model_count][each].to(device)@uc_targets[-1].T).T)

    uc_targets = torch.cat(uc_targets, 0)

    removal_indices = []
    for i in range(uc_targets.size(0)):
        for j in range(i+1, uc_targets.size(0)):
            if (uc_targets[i]-uc_targets[j]).abs().mean() == 0:
                removal_indices.append(j)

    removal_indices = list(set(removal_indices))
    uc_targets = torch.stack([uc_targets[i] for i in range(uc_targets.size(0)) if i not in removal_indices], 0)
    for each in text_layers_modified:
        uc_values[each] = torch.cat(uc_values[each], 0)
        uc_values[each] = torch.stack([uc_values[each][i] for i in range(uc_values[each].size(0)) if i not in removal_indices], 0)

    new_weights = {}
    for each in text_layers_modified:
        W = unet.state_dict()[each].float()
        # values = (W@uc.T).T # W_0 @ C_reg.T
        input_target = uc_targets # C
        output_target = uc_values[each] # W @ C.T

        Wnew = gdupdateWexact(uc,
                              input_target,
                              output_target,
                              W.clone(),
                              lam
                              )

        new_weights[each] = Wnew

    return new_weights



def parse_args():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--paths', help='+ separated list of checkpoints', required=True,
                        type=str)
    parser.add_argument('--save_path', help='folder name to save  optimized weights', default='results',
                        type=str)
    parser.add_argument('--categories', help='+ separated list of categories of the models', required=True,
                        type=str)
    parser.add_argument('--prompts', help='prompts for composition model (can be a file or string)', default=None,
                        type=str)
    parser.add_argument('--ckpt', required=True,
                        type=str)
    parser.add_argument('--regularization_image_path', default='./real_reg',
                        type=str)
    parser.add_argument('--target_image_path', default='data/benchmark_dataset/pet_cat1',
                        type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = args.paths
    categories = args.categories
    if ' ' in categories:
        temp = categories.replace(' ', '_')
    else:
        temp = categories
    outpath = '_'.join(['optimized', temp])
    compose(paths, categories, args.target_image_path, outpath, args.ckpt, args.prompts, args.regularization_image_path, args.save_path)
