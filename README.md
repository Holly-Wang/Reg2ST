# Reg2ST: Recognizing potential patterns from gene expression for spatial transcriptomics prediction

## Overview
Reg2ST is a deep learning framework designed to predict
spatial transcriptomics from histology images by integrating
gene expression patterns using contrastive learning method. It
contains feature extraction module, feature fusion module and
dynamic graph neural network.


## Framework of Reg2ST
![model](https://github.com/Holly-Wang/Reg2ST/blob/main/model.png)

## Dataset
1. the human HER2-positive breast cancer dataset (HER2+): containing 36 sections from 8 patients.
2. the human cutaneous squamous cell carcinoma (cSCC) dataset: containing 12 sections from 4 patients.

## Results
![her2st result](https://github.com/Holly-Wang/Reg2ST/blob/main/res_her2st.png)

![cscc result](https://github.com/Holly-Wang/Reg2ST/blob/main/res_cscc.png)

## Usage
1. Clone the repository

```shell
git clone https://github.com/Holly-Wang/Reg2ST.git
cd Reg2ST
```

2. Install the required dependencies, run:
```shell
conda create -n reg2st python=3.9
pip install -r requirements.txt
```

3. Prepare the dataset

Download datasets mentioned above.

4. Train the model

```shell
python train.py --fold=$i  --device_id=0  --epochs=<EPOCHS> --dataset='cscc2'
```



