#  Multi-concept Model Immunization through Differentiable Model Merging
### [Project Page](https://www.amberyzheng.com/mima/) | [Paper](https://arxiv.org/abs/2412.15320)

**[Amber Yijia Zheng](https://amberyzheng.com/), [Raymond A. Yeh](https://raymond-yeh.com/)**

Department of Computer Science, Purdue University

In AAAI 2025.

---

## Setup

This code was tested with **Python 3.10** and **PyTorch 2.1.2**. It supports [Stable Diffusion v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) via Hugging Face. To set up the environment:
```
git clone git@github.com:amberyzheng/MIMA.git
cd MIMA
conda create --name mima python=3.10
conda activate mima
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/huggingface/transformers.git
```

---


## Usage

###  Re-learning Immunization

**Training data:** 
- Place the training images in a folder, ensuring a `metadata.csv` file exists in the same folder. This file should list image filenames and their corresponding prompts.
- For multi-concept immunization, provide a JSON file specifying the data details. Examples are provided in `assets/art_concepts_list.json` and `assets/obj_concepts_list.json`.

**Immunize a pre-trained model:**

Run the following command:
```{bash}
bash scripts/relearn_train.sh  <dataset_1>+<dataset_2>+...+<dataset_k>  <'art'/'obj'>
```
**Additional Details**:
- The code checks if an erased model checkpoint exists. If not, it will use [UCE](https://github.com/rohitgandikota/unified-concept-editing) to erase target concepts, sampling images from [LAION5B](https://huggingface.co/datasets/laion/220k-GPT4Vision-captions-from-LIVIS) for concept preservation during erasure.
- Class images for prior preservation will be generated and saved in the `regularization/` directory.


### Personalization Immunization

**Training data:**
- Sample data is available in the `data` folder.
- A JSON file specifying concept details is required. An example is provided in `assets/full_concepts_list.json`.


**Immunize a pre-trained model:**

Run the following command:
```{bash}
bash scripts/personalize_train.sh  <dataset_1>+<dataset_2>+...+<dataset_k>
```

---


## Citation

If you find our work or any of our materials useful, please cite our paper:

```
@inproceedings{zheng2025multi,
  title={Multi-concept Model Immunization through Differentiable Model Merging},
  author={Zheng, Amber Yijia and Yeh, Raymond A},
  booktitle={In Proc. AAAI},
  year={2025}
}
```