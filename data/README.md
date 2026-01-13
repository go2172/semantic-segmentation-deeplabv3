# Semantic Segmentation using DeepLabV3 + ResNet101

This repository contains the implementation of a semantic segmentation pipeline using **DeepLabV3 with a ResNet-101 backbone**, fine-tuned on a **custom classroom dataset**.

## Project Overview
The goal of this project is to perform pixel-wise segmentation of everyday classroom objects such as **notice boards** and **electrical sockets**, using a fully custom dataset and an end-to-end deep learning workflow.

## Model
- Architecture: DeepLabV3
- Backbone: ResNet-101 (pretrained)
- Framework: PyTorch
- Loss Function: Cross Entropy + Dice Loss
- Optimization: Adam / SGD
- Mixed Precision Training: Enabled (AMP)

## Dataset
- Source: Custom-collected classroom images
- Annotation Tool: Roboflow
- Mask Type: PNG semantic masks
- Split: 80% Train / 10% Validation / 10% Test

- ### Dataset Structure
- images/
├── train/
├── val/
└── test/

masks/
├── train/
├── val/
└── test/

📂 Dataset Link:  
👉 < https://drive.google.com/drive/folders/1M-aQB8ZvJExJjwN-xRw
zPgScbwCQhd0?usp=sharing >

## Training & Evaluation
- Metrics: Pixel Accuracy, Mean IoU, Precision, Recall, mAP
- Hyperparameter Tuning:
  - Learning Rate: {1e-3, 5e-4, 1e-4}
  - Batch Size: {2, 4, 8}
  - Weight Decay: {0, 1e-4, 1e-5}

## Results
The optimized model significantly improves segmentation quality and boundary accuracy compared to the baseline model.

## How to Run
1. Open the notebook in Google Colab
2. Download the dataset using the link above
3. Update dataset path in the notebook
4. Run all cells


## References
- DeepLabV3 Paper: https://arxiv.org/abs/1706.05587
- ResNet: https://arxiv.org/abs/1512.03385
- PyTorch DeepLabV3: https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html

- 
### Notes
- Masks are class-index encoded.
- This dataset was created and annotated manually for academic and research purposes.
