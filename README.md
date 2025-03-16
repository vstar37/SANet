<h1 align="center">Incremental Structural Adaptation for Camouflaged Object Detection</h1>


## News :newspaper:
* **`Nov 21, 2024`:** First release.

## 💡 Overview
This repository provides a PyTorch implementation of **SANet**, a **Structure-Adaptive Network** designed for **Camouflaged Object Detection** (COD). SANet addresses the challenges of detecting camouflaged objects by incorporating an innovative incremental structural adaptation mechanism, which enhances the model's ability to refine segmentation and improve localization in complex environments. The key feature of SANet is its ability to adaptively integrate high-resolution structural information, enabling fine-grained detection of camouflaged objects that closely resemble their backgrounds.

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
    # PyTorch==2.0.1 is used for faster training with compilation.
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
| SANet-S |  Swin-S    |  85.31   |  [[Geogle Drive](https://drive.google.com/drive/folders/1ehBdZcQWRVshFxR2u7-E1Uv-fwhkdOiE?usp=drive_link)]|
| SANet-L |  Swin-L    |  187.26  |  [[Geogle Drive](https://drive.google.com/drive/folders/1ehBdZcQWRVshFxR2u7-E1Uv-fwhkdOiE?usp=drive_link)]|

- **Model Weights:** The model weights should be placed in the `ckpt/COD` directory.
- **Backbone Weights:** The backbone weights should be placed in the `lib/weights/backbones` directory.
After downloading the datasets, you will need to download the pretrained weights for the model and the backbone. These weights are required to initialize the model for training and inference.

### 2. Prediction Maps
We offer the prediction maps of **SANet-S** [[baidu](https://pan.baidu.com/s/13MKOObYH6afYzF7P-2vjeQ),PIN:gsvf] and **SANet-L** [[Geogle Drive](https://drive.google.com/file/d/17q1poRj1FagWDoVSSX1712Wl9P32xRHO/view?usp=share_link)] at this time.




   
