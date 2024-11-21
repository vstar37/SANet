<h1 align="center">Incremental Structural Adaptation for Camouflaged Object Detection</h1>

<div align='center'>
    <strong>Qingzheng Wang</strong>,&thinsp;
    <strong>Jiazhi Xie</strong><sup> *</sup>,&thinsp;
    <strong>Ning Li</strong>,&thinsp;
    <strong>Xingqin Wang</strong>,&thinsp;
    <strong>Wenhui Liu</strong>,&thinsp;
    <strong>Zhengwei Mai</strong>
</div>


## News :newspaper:
* **`Nov 21, 2024`:** First release.

## Overview
This repository provides a PyTorch implementation of **SANet**, a **Structure-Adaptive Network** designed for **Camouflaged Object Detection** (COD). SANet addresses the challenges of detecting camouflaged objects by incorporating an innovative incremental structural adaptation mechanism, which enhances the model's ability to refine segmentation and improve localization in complex environments. The key feature of SANet is its ability to adaptively integrate high-resolution structural information, enabling fine-grained detection of camouflaged objects that closely resemble their backgrounds.

## Usage

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
       

   
