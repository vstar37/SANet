<h1 align="center">Incremental Structural Adaptation for Camouflaged Object Detection</h1>

## 🗞️ News
* **`Nov 21, 2024`:** First release.

## 💡 Overview
This repository provides a PyTorch implementation of **SANet**, a **Structure-Adaptive Network** designed for **Camouflaged Object Detection** (COD).  
SANet addresses the challenges of detecting camouflaged objects by incorporating an innovative incremental structural adaptation mechanism, which enhances the model's ability to refine segmentation and improve localization in complex environments.  
The key feature of SANet is its ability to adaptively integrate high-resolution structural information, enabling fine-grained detection of camouflaged objects that closely resemble their backgrounds.

---

## 💻 Environment

- **OS:** Ubuntu 20.04  
- **CUDA:** 11.8  
- **Python:** 3.10  
- **PyTorch:** 2.4.0
- Other required packages: see [`requirements.txt`](./requirements.txt)

---

## 💡 Usage

### Installation
To use this repository, follow the steps below to set up the environment:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vstar37/SANet.git
   cd SANet

2.	**Install the required dependencies:**
    It is recommended to create a virtual environment first, then install the dependencies.
    ```bash
    # PyTorch==2.4.0 is used for faster training with compilation.
    conda create -n sanet python=3.10 -y && conda activate sanet
    pip install -r requirements.txt

3. **Download the datasets:**
   [Download datasets](https://drive.google.com/drive/folders/1ehBdZcQWRVshFxR2u7-E1Uv-fwhkdOiE?usp=drive_link)
    After setting up the environment, you can download the training and test datasets from the provided links. Once downloaded, please unzip the datasets into the data folder under the root directory of the project. The folder structure should look like this:
   ```bash
   SANet/
   ├── datasets/
   │   ├── CAMO_TestingDataset/
   │   ├── CHAMELEON_TestingDataset/
   │   ├── COD10K_TestingDataset/
   │   ├── NC4K_TestingDataset/
   │   └── COD10K_CAMO_CHAMELEON_TrainingDataset/
   ├── requirements.txt
   └── … (other project files)

### Run
```shell
# Train & Test & Evaluation
./sub.sh RUN_NAME GPU_NUMBERS_FOR_TRAINING GPU_NUMBERS_FOR_TEST
# Example: ./sub.sh  0,1,2,3,4,5,6,7 0

# See train.sh / test.sh for only training / test-evaluation.
# After the evaluation, run `gen_best_ep.py` to select the best ckpt from a specific metric (you choose it from Sm, wFm).
```


## 💡 Results

### 1. Model Weight
| Name | Backbone | Params | Weight |
|  :---: |  :---:    | :---:   |  :---:   |
| SANet-S |  Swin-S    |  87.3   |  [[Geogle Drive](https://drive.google.com/drive/folders/1ehBdZcQWRVshFxR2u7-E1Uv-fwhkdOiE?usp=drive_link)]|
| SANet-L |  Swin-L    |  241.209  |  [[Geogle Drive](https://drive.google.com/drive/folders/1ehBdZcQWRVshFxR2u7-E1Uv-fwhkdOiE?usp=drive_link)]|

- **Model Weights:** The model weights should be placed in the `ckpt/COD` directory.
- **Backbone Weights:** The backbone weights should be placed in the `lib/weights/backbones` directory.
After downloading the datasets, you will need to download the pretrained weights for the model and the backbone. These weights are required to initialize the model for training and inference.

### 2. Prediction Maps
We offer the prediction maps of **SANet-S** [[baidu](https://pan.baidu.com/s/13MKOObYH6afYzF7P-2vjeQ),PIN:gsvf] and **SANet-L** [[Geogle Drive](https://drive.google.com/file/d/17q1poRj1FagWDoVSSX1712Wl9P32xRHO/view?usp=share_link)] at this time.

### 3. Polyp and SOD Prediction Maps
We offer the prediction maps of **SANet-L** [[Polyp](https://drive.google.com/file/d/1YGrEHNHIYh9Y9iSXR-8CB3fq5-OjYQbF/view?usp=share_link)], [[SOD](https://drive.google.com/file/d/1Nl2yjuWZb-cF5vWX8rREaeQ7RyvasWgk/view?usp=share_link)] at this time.




   
   

## 📄 Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@article{WANG2025105565,
  title = {Incremental structural adaptation for camouflaged object detection},
  journal = {Image and Vision Computing},
  volume = {159},
  pages = {105565},
  year = {2025},
  issn = {0262-8856},
  doi = {https://doi.org/10.1016/j.imavis.2025.105565},
  url = {https://www.sciencedirect.com/science/article/pii/S0262885625001532},
  author = {Qingzheng Wang and Jiazhi Xie and Ning Li and Xingqin Wang and Wenhui Liu and Zengwei Mai},
  keywords = {Camouflaged object detection, Image segmentation, Computer vision, Structural information},
  abstract = {Camouflaged Object Detection (COD) is a challenging task due to the similarity between camouflaged objects and their backgrounds. Recent approaches predominantly utilize structural cues but often struggle with misinterpretations and noise, particularly for small objects. To address these issues, we propose the Structure-Adaptive Network (SANet), which incrementally supplements structural information from points to surfaces. Our method includes the Key Point Structural Information Prompting Module (KSIP) to enhance point-level structural information, Mixed-Resolution Attention (MRA) to incorporate high-resolution details, and the Structural Adaptation Patch (SAP) to selectively integrate high-resolution patches based on the shape of the camouflaged object. Experimental results on three widely used COD datasets demonstrate that SANet significantly outperforms state-of-the-art methods, achieving more accurate localization and finer edge segmentation, while minimizing background noise. Our code is available at https://github.com/vstar37/SANet/.}
}
```
