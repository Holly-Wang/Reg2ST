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
1. the human HER2-positive breast cancer (HER2+) dataset: containing 36 sections from 8 patients.
2. the human cutaneous squamous cell carcinoma (cSCC) dataset: containing 12 sections from 4 patients.

## Results
![her2st result](https://github.com/Holly-Wang/Reg2ST/blob/main/result/res_her2st.png)

![cscc result](https://github.com/Holly-Wang/Reg2ST/blob/main/result/res_cscc.png)


| Model       | Mean PCC (HER2+) | Mean PCC (cSCC) | Median PCC (HER2+) | Median PCC (cSCC) |
|-------------|------------------|------------------|---------------------|---------------------|
| HisToGene   | 0.0831           | 0.0775           | 0.0770              | 0.0785              |
| Hist2ST     | 0.1504           | 0.1819           | 0.1353              | 0.1780              |
| THIToGene   | 0.1390           | 0.1810           | 0.1249              | 0.1746              |
| Reg2ST      | **0.1741**       |  **0.2021**      |  **0.1616**         |  **0.1911**         |

Reuslts of ablation study and parameter sensitivity are in  folder `results`.

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
```shell
# Download Reg2ST dataset, which contains spatial transcriptomics and phikonv2 embedding of HER2+ and cSCC datasets.
cd data
bash download.sh
```

4. Train the model
```shell
cd code
python train.py --fold=$i  --device_id=0  --epochs=<EPOCHS> --dataset='her2st'
```
5. Predict
```shell
cd code
python predict.py --fold=0  --device_id=0 --dataset='her2st'
```

