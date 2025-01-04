import os
import random
from pathlib import Path
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import pdb


def preprocess(image, scale, resample):
    image = image.resize((scale, scale), resample=resample)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    return image


def collate_fn(examples, with_prior_preservation):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    mask = [example["mask"] for example in examples]
    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if with_prior_preservation:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]
        mask += [example["class_mask"] for example in examples]

    input_ids = torch.cat(input_ids, dim=0)
    pixel_values = torch.cat(pixel_values, dim=0) if len(pixel_values[0].shape) == 4 else torch.stack(pixel_values)
    mask = torch.cat(mask, dim=0) if len(mask[0].shape) == 4 else torch.stack(mask)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    mask = mask.to(memory_format=torch.contiguous_format).float()

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "mask": mask.unsqueeze(1)
    }
    return batch


class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class MIMADataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        max_train_samples=None
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = PIL.Image.BILINEAR
        self.concept_list = concepts_list

        self.instance_images_path = {}
        self.class_images_path = {}
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            concept_name = concept["instance_data_dir"].split("/")[-1]
            inst_img_path = [(x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if (x.is_file() and "metadata" not in x.stem)] # ignore metadata file
            if max_train_samples:
                inst_img_path = inst_img_path[:max_train_samples]
            self.instance_images_path[concept_name] = inst_img_path

            if with_prior_preservation:
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path[concept_name] = class_img_path[:num_class_images]
        
        self.num_instance_images = sum([len(x) for x in self.instance_images_path.values()])
        self.num_class_images = sum([len(x) for x in self.class_images_path.values()])
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        instance_images = []
        instance_prompt_ids = []
        masks = []
        if self.with_prior_preservation:
            class_images = []
            class_prompt_ids = []
            class_masks = []
        for concept in self.concept_list:
            concept_name = concept["instance_data_dir"].split("/")[-1]
            instance_image, instance_prompt = self.instance_images_path[concept_name][index % len(self.instance_images_path[concept_name])]
            instance_image = Image.open(instance_image)
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            instance_image = self.flip(instance_image)        

            ##############################################################################
            #### apply resize augmentation and create a valid image region mask ##########
            ##############################################################################
            if np.random.randint(0, 3) < 2:
                random_scale = np.random.randint(self.size // 3, self.size+1)
            else:
                random_scale = np.random.randint(int(1.2*self.size), int(1.4*self.size))

            if random_scale % 2 == 1:
                random_scale += 1

            if random_scale < 0.6*self.size:
                add_to_caption = np.random.choice(["a far away ", "very small "])
                instance_prompt = add_to_caption + instance_prompt
                cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
                cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
                instance_image1 = preprocess(instance_image, random_scale, self.interpolation)
                instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
                instance_image[cx - random_scale // 2: cx + random_scale // 2, cy - random_scale // 2: cy + random_scale // 2, :] = instance_image1

                mask = np.zeros((self.size // 8, self.size // 8))
                mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1, (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.
            elif random_scale > self.size:
                add_to_caption = np.random.choice(["zoomed in ", "close up "])
                instance_prompt = add_to_caption + instance_prompt
                cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
                cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

                instance_image = preprocess(instance_image, random_scale, self.interpolation)
                instance_image = instance_image[cx - self.size // 2: cx + self.size // 2, cy - self.size // 2: cy + self.size // 2, :]
                mask = np.ones((self.size // 8, self.size // 8))
            else:
                instance_image = preprocess(instance_image, self.size, self.interpolation)
                mask = np.ones((self.size // 8, self.size // 8))
            ########################################################################
                
            instance_images.append(torch.from_numpy(instance_image).permute(2, 0, 1))
            instance_prompt_ids.append(
                self.tokenizer(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
                ).input_ids
                )
            masks.append(torch.from_numpy(mask))

            if self.with_prior_preservation:
                class_image, class_prompt = self.class_images_path[concept_name][index % len(self.class_images_path[concept_name])]
                class_image = Image.open(class_image)
                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                class_images.append(self.image_transforms(class_image))
                class_prompt_ids.append(
                    self.tokenizer(
                    class_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                    ).input_ids
                    )
                class_masks.append(torch.ones_like(torch.from_numpy(mask)))
                


        example["instance_images"] = torch.stack(instance_images, dim=0)
        example["mask"] = torch.stack(masks, dim=0)
        example["instance_prompt_ids"] = torch.stack(instance_prompt_ids, dim=0)

        example["class_images"] = torch.stack(class_images, dim=0)
        example["class_mask"] = torch.stack(class_masks, dim=0)
        example["class_prompt_ids"] = torch.stack(class_prompt_ids, dim=0)

        return example

def merge_dict(dict1, dict2):
    for key in dict2.keys():
        if key not in dict1.keys():
            dict1[key] = dict2[key]
            continue
        for key2 in dict2[key].keys():
            dict1[key][key2] = torch.cat([dict1[key][key2], dict2[key][key2]], dim=0)
    return dict1


def collate_fn_compose(examples, with_prior_preservation):
    all_one_batch = {}
    for batch in examples:
        current_example = {}
        for concept_example in batch:
            current_class = concept_example["class_name"]
            if current_class not in current_example:
                current_example[current_class] = {}
            input_ids = [concept_example["instance_prompt_ids"]]
            pixel_values = [concept_example["instance_images"]]
            mask = [concept_example["mask"]]
            # Concat class and instance examples for prior preservation.
            # We do this to avoid doing two forward passes.
            if with_prior_preservation:
                input_ids_reg = [concept_example["class_prompt_ids"]]
                pixel_values_reg = [concept_example["class_images"]]
                mask_reg = [concept_example["class_mask"]]

            input_ids = torch.cat(input_ids, dim=0)
            pixel_values = torch.stack(pixel_values)
            mask = torch.stack(mask)
            input_ids_reg = torch.cat(input_ids_reg, dim=0)
            pixel_values_reg = torch.stack(pixel_values_reg)
            mask_reg = torch.stack(mask_reg)
            # class_name = torch.stack(class_name)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
            mask = mask.to(memory_format=torch.contiguous_format).float()

            current_example[current_class]["input_ids"] = input_ids
            current_example[current_class]["pixel_values"] = pixel_values
            current_example[current_class]["mask"] = mask.unsqueeze(1)
            current_example[current_class]["input_ids_reg"] = input_ids_reg
            current_example[current_class]["pixel_values_reg"] = pixel_values_reg
            current_example[current_class]["mask_reg"] = mask_reg.unsqueeze(1)
        
        all_one_batch = merge_dict(all_one_batch, current_example)
        
    for key, value in all_one_batch.items():
        for key2 in ["input_ids", "pixel_values", "mask"]:
            all_one_batch[key][key2] = torch.cat([value[key2], value[key2+"_reg"]], dim=0)
            del all_one_batch[key][key2+"_reg"]
    return all_one_batch


class MIMADataset_Compose(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    For composition during training.
    """

    def __init__(
        self,
        concepts_list,
        tokenizer,
        size=512,
        center_crop=False,
        with_prior_preservation=False,
        num_class_images=200,
        hflip=False,
        max_train_samples=None
    ):
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer
        self.interpolation = PIL.Image.BILINEAR

        self.instance_images_path = {}
        self.class_images_path = {}
        self.concept_class_pairs = []
        self.with_prior_preservation = with_prior_preservation
        for concept in concepts_list:
            concept_name = concept["instance_data_dir"].split("/")[-1]
            inst_img_path = [(x, concept["instance_prompt"]) for x in Path(concept["instance_data_dir"]).iterdir() if (x.is_file() and "metadata" not in x.stem)] # ignore metadata file
            if max_train_samples:
                inst_img_path = inst_img_path[:max_train_samples]
            self.instance_images_path[concept_name] = inst_img_path

            if with_prior_preservation:
                class_name = concept["class_data_dir"].split("/")[-1].replace(" ", "")
                class_data_root = Path(concept["class_data_dir"])
                if os.path.isdir(class_data_root):
                    class_images_path = list(class_data_root.iterdir())
                    class_prompt = [concept["class_prompt"] for _ in range(len(class_images_path))]
                else:
                    with open(class_data_root, "r") as f:
                        class_images_path = f.read().splitlines()
                    with open(concept["class_prompt"], "r") as f:
                        class_prompt = f.read().splitlines()

                class_img_path = [(x, y) for (x, y) in zip(class_images_path, class_prompt)]
                self.class_images_path[class_name] = class_img_path[:num_class_images]
                self.concept_class_pairs.append((concept_name, class_name))
            else:
                self.concept_class_pairs.append((concept_name, concept_name))

        for path_list in self.instance_images_path.values():
            random.shuffle(path_list)
        self.num_instance_images = sum([len(x) for x in self.instance_images_path.values()])
        self.num_class_images = sum([len(x) for x in self.class_images_path.values()])
        self._length = max(self.num_class_images, self.num_instance_images)
        self.flip = transforms.RandomHorizontalFlip(0.5 * hflip)

        self.image_transforms = transforms.Compose(
            [
                self.flip,
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __len__(self):
        return self._length

    def __getitem__(self, index):
        examples = []
        for concept_name, class_name in self.concept_class_pairs:
            example = {}
            instance_image, instance_prompt = self.instance_images_path[concept_name][index % len(self.instance_images_path[concept_name])]
            instance_image = Image.open(instance_image)
            if not instance_image.mode == "RGB":
                instance_image = instance_image.convert("RGB")
            instance_image = self.flip(instance_image)

            ##############################################################################
            #### apply resize augmentation and create a valid image region mask ##########
            ##############################################################################
            if np.random.randint(0, 3) < 2:
                random_scale = np.random.randint(self.size // 3, self.size+1)
            else:
                random_scale = np.random.randint(int(1.2*self.size), int(1.4*self.size))

            if random_scale % 2 == 1:
                random_scale += 1

            if random_scale < 0.6*self.size:
                add_to_caption = np.random.choice(["a far away ", "very small "])
                instance_prompt = add_to_caption + instance_prompt
                cx = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
                cy = np.random.randint(random_scale // 2, self.size - random_scale // 2 + 1)
                instance_image1 = preprocess(instance_image, random_scale, self.interpolation)
                instance_image = np.zeros((self.size, self.size, 3), dtype=np.float32)
                instance_image[cx - random_scale // 2: cx + random_scale // 2, cy - random_scale // 2: cy + random_scale // 2, :] = instance_image1

                mask = np.zeros((self.size // 8, self.size // 8))
                mask[(cx - random_scale // 2) // 8 + 1: (cx + random_scale // 2) // 8 - 1, (cy - random_scale // 2) // 8 + 1: (cy + random_scale // 2) // 8 - 1] = 1.
            elif random_scale > self.size:
                add_to_caption = np.random.choice(["zoomed in ", "close up "])
                instance_prompt = add_to_caption + instance_prompt
                cx = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)
                cy = np.random.randint(self.size // 2, random_scale - self.size // 2 + 1)

                instance_image = preprocess(instance_image, random_scale, self.interpolation)
                instance_image = instance_image[cx - self.size // 2: cx + self.size // 2, cy - self.size // 2: cy + self.size // 2, :]
                mask = np.ones((self.size // 8, self.size // 8))
            else:
                instance_image = preprocess(instance_image, self.size, self.interpolation)
                mask = np.ones((self.size // 8, self.size // 8))
            ########################################################################

            example["instance_images"] = torch.from_numpy(instance_image).permute(2, 0, 1)
            example["mask"] = torch.from_numpy(mask)
            example["instance_prompt_ids"] = self.tokenizer(
                instance_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids

            if self.with_prior_preservation:
                class_image, class_prompt = self.class_images_path[class_name][index % len(self.class_images_path[class_name])]
                class_image = Image.open(class_image)
                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                example["class_images"] = self.image_transforms(class_image)
                example["class_mask"] = torch.ones_like(example["mask"])
                example["class_prompt_ids"] = self.tokenizer(
                    class_prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=self.tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids
                example["class_name"] = class_prompt

            examples.append(example)

        return examples