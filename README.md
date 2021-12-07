# Interpreting-MDETR-Using-Attribution-Guided-Factorization

This repository contains our implementation of Attribution Guided Factorization, a model interpretability method, applied upon a state-of-the-art query-modulated object detection model MDETR.

The code should be able to run out of the box in Google's Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RU-CS535-ADOX/Interpreting-MDETR-Using-Attribution-Guided-Factorization/blob/main/demo.ipynb).

## Building Blocks

The implementation contains the following two building blocks:

### MDETR: Modulated Detection for End-to-End Multi-Modal Understanding

[Website](https://ashkamath.github.io/mdetr_page/) • [Paper](https://arxiv.org/abs/2104.12763)

#### Model checkpoint paths

The model checkpoint for MDETR and dataset annotations can be found at: <https://zenodo.org/record/4721981#.YawS29DMKUk>

- ResNet-101: <https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth>
- EfficientNet-B3: <https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth>
- EfficientNet-B5: <https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth>
- CLEVR: <https://zenodo.org/record/4721981/files/clevr_checkpoint.pth>
- CLEVR-Humans: <https://zenodo.org/record/4721981/files/clevr_humans_checkpoint.pth>
- ResNet-101-GQA: <https://zenodo.org/record/4721981/files/gqa_resnet101_checkpoint.pth>
- EfficientNet-B5-GQA: <https://zenodo.org/record/4721981/files/gqa_EB5_checkpoint.pth>
- ResNet-101-PhraseCut: <https://zenodo.org/record/4721981/files/phrasecut_resnet101_checkpoint.pth>
- EfficientNet-B3-PhraseCut: <https://zenodo.org/record/4721981/files/phrasecut_EB3_checkpoint.pth>
- ResNet-101-RefCOCO: <https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth>
- EfficientNet-B3-RefCOCO: <https://zenodo.org/record/4721981/files/refcoco_EB3_checkpoint.pth>
- ResNet-101-RefCOCO+: <https://zenodo.org/record/4721981/files/refcoco%2B_resnet101_checkpoint.pth>
- EfficientNet-B3-RefCOCO+: <https://zenodo.org/record/4721981/files/refcoco%2B_EB3_checkpoint.pth>
- ResNet-101-RefCOCOg: <https://zenodo.org/record/4721981/files/refcocog_resnet101_checkpoint.pth>
- EfficientNet-B3-RefCOCOg: <https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth>

### AGF: Attribution Guided Factorization

[Conference Recording](https://slideslive.com/38949126/visualization-of-supervised-and-selfsupervised-neural-networks-via-attribution-guided-factorization) • [Paper](https://arxiv.org/abs/2012.02166)
