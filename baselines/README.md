We use three baselins: [HisToGene](https://github.com/maxpmx/HisToGene), [Hist2ST](https://github.com/biomed-AI/Hist2ST), [THItoGene](https://github.com/yrjia1015/THItoGene).

### HisToGene
HisToGene is a deep learning method that predicts super-resolution gene expression from histology images in tumors. Trained in a spatial transcriptomics dataset, HisToGene models the spatial dependency in gene expression and histological features among spots through a modified Vision Transformer model.
![HisToGene](https://github.com/maxpmx/HisToGene/blob/main/Workflow.PNG)

### Hist2ST
Hist2ST is a deep learning-based model using histology images to predict RNA-seq expression. At each sequenced spot, the corre-sponding histology image is cropped into an image patch, from which 2D vision features are learned through convolutional operations. Meanwhile, the spatial relations with the whole image and neighbored patches are captured through Transformer and graph neural network modules, respectively. These learned features are then used to predict the gene expression by following the zero-inflated negative binomial (ZINB) distribution. To alleviate the impact by the small spatial transcriptomics data, a self-distillation mechanism is employed for efficient learning of the model. Hist2ST was tested on the HER2-positive breast cancer and the cutaneous squamous cell carcinoma datasets, and shown to outperform existing methods in terms of both gene expression prediction and following spatial region identification.
![Hist2ST](https://github.com/biomed-AI/Hist2ST/blob/main/Workflow.png)
### THItoGene
THItoGene is a hybrid neural network that leverages dynamic convolution and capsule networks to adaptively perceive latent molecular signals from histological images, for the systematic analysis of spatial gene expression within tissue pathology. THItoGene integrates gene expression, spatial locations, and histological images to explore and analyze the relationship between high-resolution pathological image phenotypes and tumor genetic morphology.

![THItoGene](https://github.com/yrjia1015/THItoGene/blob/main/workflow.png)