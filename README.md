# Interpreting-MDETR-Using-Attribution-Guided-Factorization

This repository contains our implementation of Attribution Guided Factorization, a model interpretability method, applied upon a state-of-the-art query-modulated object detection model MDETR.

The code should be able to run out of the box in Google's Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RU-CS535-ADOX/Interpreting-MDETR-Using-Attribution-Guided-Factorization/blob/main/demo.ipynb).

## Building Blocks

The implementation contains the following two building blocks:

### MDETR: Modulated Detection for End-to-End Multi-Modal Understanding

[Website](https://ashkamath.github.io/mdetr_page/) • [Paper](https://arxiv.org/abs/2104.12763)

#### Input format
- MDETR expects two inputs - batch of images of shape [batch_size x 3 x H x W] where batch_size corresponds to the number of input samples and a list of captions of length batch_size.  

#### Output format

After propagating an image sample or a batch of image samples into the model, we get the output as a dictionary containing keys - pred_logits, pred_boxes, proj_queries, proj_tokens and tokenized.

- 'pred_logits' - This is the probability of each predicted class for all the N object queries across all images in the batch. Shape: [batch_size x num_queries x (num_classes + 1)]
- 'pred_boxes' - This is the normalized coordinate values for prediction boxes represented as (center_x, center_y, height, width)
- 'tokenized'  - This contains the tokenized input caption

#### Visualizing output

- Predictions - The softmax function is applied on pred_logits to get the probabilities to sum to 1. And then only the predictions with the confidence of 0.7 are taken. This gives us the final predictions for the given input caption.
- Bounding Box coordinates - Then corresponding pred_boxes are scaled according to the size of the image to get the box coordinates. 
- Labels - ‘tokenized’ is then used to decode the input labels for each prediction. 

The above three data is then used to plot the final predictions on the given input image sample. 


#### Model checkpoint paths

The model checkpoint for MDETR and dataset annotations can be found at: <https://zenodo.org/record/4721981#.YawS29DMKUk>.

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
