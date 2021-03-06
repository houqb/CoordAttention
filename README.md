# Coordinate Attention for Efficient Mobile Network Design ([preprint](https://arxiv.org/abs/2103.02907))

This repository is a PyTorch implementation of our coordinate attention (will appear in CVPR2021).

Our coordinate attention can be easily plugged into any classic building blocks as a feature representation augmentation tool. Here ([pytorch-image-models](https://github.com/rwightman/pytorch-image-models)) is a code base that you might want to train a classification model on ImageNet.


### Comparison to Squeeze-and-Excitation block and CBAM

![diagram](diagram.png)

(a) Squeeze-and-Excitation block      (b) CBAM      (C) Coordinate attention block


### How to plug the proposed CA block in the [inverted residual block](https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf) and the [sandglass block](https://arxiv.org/pdf/2007.02269.pdf)

![wheretoplug](attpos.png)
