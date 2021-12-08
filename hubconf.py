# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from models.backbone import Backbone, Joiner, TimmBackbone
from models.mdetr import MDETR
from models.position_encoding import PositionEmbeddingSine
from models.postprocessors import PostProcess, PostProcessSegm
from models.segmentation import DETRsegm
from models.transformer import Transformer

dependencies = ["torch", "torchvision"]

cfg = { 
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.clone = Clone()
        self.add = Add()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=False)
        self.relu2 = ReLU(inplace=False)

        self.register_forward_hook(forward_hook)

    def forward(self, x):
        x1, x2 = self.clone(x, 2)

        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            x2 = self.downsample(x2)

        out = self.add([out, x2])
        out = self.relu2(out)

        return out

    def AGF(self, cam, grad_outputs, **kwargs):
        cam, grad_outputs = self.relu2.AGF(cam, grad_outputs, **kwargs)

        (cam, cam2), (grad_outputs, grad_outputs2) = self.add.AGF(cam, grad_outputs, **kwargs)

        if self.downsample is not None:
            cam2, grad_outputs2 = self.downsample.AGF(cam2, grad_outputs2, flat=True, **kwargs)

        cam, grad_outputs = self.bn2.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.conv2.AGF(cam, grad_outputs, flat=True, **kwargs)

        cam, grad_outputs = self.relu1.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.bn1.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.conv1.AGF(cam, grad_outputs, **kwargs)

        return self.clone.AGF((cam, cam2), grad_outputs + grad_outputs2)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.clone = Clone()
        self.add = Add()

        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

        self.relu1 = ReLU(inplace=False)
        self.relu2 = ReLU(inplace=False)
        self.relu3 = ReLU(inplace=False)

        self.register_forward_hook(forward_hook)

    def forward(self, x):
        x1, x2 = self.clone(x, 2)

        out = self.conv1(x1)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x2 = self.downsample(x2)

        out = self.add([out, x2])
        out = self.relu3(out)

        return out

    def AGF(self, cam, grad_outputs, **kwargs):
        cam, grad_outputs = self.relu3.AGF(cam, grad_outputs, **kwargs)

        (cam, cam2), (grad_outputs, grad_outputs2) = self.add.AGF(cam, grad_outputs, **kwargs)
        grad_outputs = grad_outputs if type(grad_outputs) is tuple else (grad_outputs,)
        grad_outputs2 = grad_outputs2 if type(grad_outputs2) is tuple else (grad_outputs2,)

        if self.downsample is not None:
            cam2, grad_outputs2 = self.downsample.AGF(cam2, grad_outputs2, flat=True, **kwargs)

        cam, grad_outputs = self.bn3.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.conv3.AGF(cam, grad_outputs, flat=True, **kwargs)

        cam, grad_outputs = self.relu2.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.bn2.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.conv2.AGF(cam, grad_outputs, **kwargs)

        cam, grad_outputs = self.relu1.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.bn1.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.conv1.AGF(cam, grad_outputs, flat=True, **kwargs)

        return self.clone.AGF((cam, cam2), grad_outputs + grad_outputs2)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU(inplace=False)
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def AGF(self, **kwargs):
        cam, grad_outputs = self.fc.AGF(**kwargs)
        cam = cam.reshape_as(self.avgpool.Y)
        grad_outputs = (grad_outputs[0].reshape_as(self.avgpool.Y),)
        cam, grad_outputs = self.avgpool.AGF(cam, grad_outputs, **kwargs)

        cam, grad_outputs = self.layer4.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.layer3.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.layer2.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.layer1.AGF(cam, grad_outputs, **kwargs)

        cam, grad_outputs = self.maxpool.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.relu.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.bn1.AGF(cam, grad_outputs, **kwargs)
        cam, grad_outputs = self.conv1.AGF(cam, grad_outputs, **kwargs)

        cam = cam / minmax_dims(cam, 'max')
        return cam.sum(1, keepdim=True)





class VGG(nn.Module):

    def __init__(self, vgg_name, batch_norm=False, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = self.make_layers(cfg[vgg_name], batch_norm=batch_norm)
        self.name = vgg_name
        self.bn = batch_norm
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

        self.get_biases = False
        self.get_features = False

        self.biases = []
        self.feature_list = []


    def getBiases(self, device):
        """
        Returns the explicit biases arising 
        from BatchNorm or convolution layers.
        """

        self.get_biases = True
        self.biases = [0]

        x = torch.zeros(1,3,224,224).to(device) #put in GPU
        _ = self.forward(x)
        self.get_biases = False
        return self.biases


    def getFeatures(self, x):
        """
        Returns features at every layer before
        the application of ReLU.
        """
        
        self.get_features = True
        self.feature_list = [x]

        x = self.forward(x)
        self.get_features = False
        return x, self.feature_list


    def _classify(self, x):
        for m in self.classifier: 
            x = m(x)
            if isinstance(m, nn.Linear):
                if self.get_biases:
                    self.biases.append(m.bias)
                if self.get_features:
                    self.feature_list.append(x)
        return x

    def forward(self, x):
        x = self.organize_features(x)
        x = x.view(x.size(0), -1)
        x = self._classify(x)
        return x

    def organize_features(self, x):
        in_channels = 3
        count = 0
        x_feat = None
        for i in cfg[self.name]:

            if (i == 'M'):
                x = self.features[count](x)
            else:
                if self.get_biases:
                    input_bias = torch.zeros(x.size()).to(x.device)
                    input_bias, _ = self._linear_block(input_bias, count)
                    self.biases.append(input_bias.detach())

                x, count = self._linear_block(x, count)    
                if self.get_features:
                    self.feature_list.append(x)

                x = self.features[count](x)

            count = count + 1

        return x

    def _linear_block(self, x, count):
        if self.bn:
            x = self.features[count](x)
            count = count + 1
        x = self.features[count](x)
        count = count + 1
        return x, count

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


    def make_layers(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        index = 0
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=False)]
                in_channels = v
        return nn.ModuleList(layers)


def _make_backbone(backbone_name: str, mask: bool = False):
    if backbone_name[: len("timm_")] == "timm_":
        backbone = TimmBackbone(
            backbone_name[len("timm_") :],
            mask,
            main_layer=-1,
            group_norm=True,
        )
    else:
        backbone = Backbone(backbone_name, train_backbone=True, return_interm_layers=mask, dilation=False)

    hidden_dim = 256
    pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
    backbone_with_pos_enc = Joiner(backbone, pos_enc)
    backbone_with_pos_enc.num_channels = backbone.num_channels
    return backbone_with_pos_enc


def _make_detr(
    backbone_name: str,
    num_queries=100,
    mask=False,
    qa_dataset=None,
    predict_final=False,
    text_encoder="roberta-base",
    contrastive_align_loss=True,
):
    hidden_dim = 256
    backbone = _make_backbone(backbone_name, mask)
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True, text_encoder_type=text_encoder)
    detr = MDETR(
        backbone,
        transformer,
        num_classes=255,
        num_queries=num_queries,
        qa_dataset=qa_dataset,
        predict_final=predict_final,
        contrastive_align_loss=contrastive_align_loss,
        contrastive_hdim=64,
    )
    if mask:
        return DETRsegm(detr)
    return detr


def mdetr_resnet101(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Pretrained on our combined aligned dataset of 1.3 million images paired with text.
    """

    model = _make_detr("resnet101")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB3(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Pretrained on our combined aligned dataset of 1.3 million images paired with text.
    """

    model = _make_detr("timm_tf_efficientnet_b3_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/pretrained_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB5(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB5 with 6 encoder and 6 decoder layers.
    Pretrained on our combined aligned dataset of 1.3 million images paired with text.
    """

    model = _make_detr("timm_tf_efficientnet_b5_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/pretrained_EB5_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_clevr(pretrained=False, return_postprocessor=False):
    """
    MDETR R18 with 6 encoder and 6 decoder layers.
    Trained on CLEVR, achieves 99.7% accuracy
    """

    model = _make_detr("resnet18", num_queries=25, qa_dataset="clevr", text_encoder="distilroberta-base")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/clevr_checkpoint.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_clevr_humans(pretrained=False, return_postprocessor=False):
    """
    MDETR R18 with 6 encoder and 6 decoder layers.
    Trained on CLEVR-Humans, achieves 81.7% accuracy
    """

    model = _make_detr("resnet18", num_queries=25, qa_dataset="clevr", text_encoder="distilroberta-base")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/clevr_humans_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_resnet101_gqa(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on GQA, achieves 61.99 on test-std
    """

    model = _make_detr("resnet101", qa_dataset="gqa", contrastive_align_loss=False)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/gqa_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB5_gqa(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB5 with 6 encoder and 6 decoder layers.
    Trained on GQA, achieves 61.99 on test-std
    """

    model = _make_detr("timm_tf_efficientnet_b5_ns", qa_dataset="gqa", contrastive_align_loss=False)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/gqa_EB5_checkpoint.pth", map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_resnet101_phrasecut(pretrained=False, threshold=0.5, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on Phrasecut, achieves 53.1 M-IoU on the test set
    """
    model = _make_detr("resnet101", mask=True, contrastive_align_loss=False)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/phrasecut_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, [PostProcess(), PostProcessSegm(threshold=threshold)]
    return model


def mdetr_efficientnetB3_phrasecut(pretrained=False, threshold=0.5, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Trained on Phrasecut, achieves 53.7 M-IoU on the test set
    """
    model = _make_detr("timm_tf_efficientnet_b3_ns", mask=True, contrastive_align_loss=False)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/phrasecut_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, [PostProcess(), PostProcessSegm(threshold=threshold)]
    return model


def mdetr_resnet101_refcoco(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on refcoco, achieves 86.75 val accuracy
    """
    model = _make_detr("resnet101")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcoco_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB3_refcoco(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Trained on refcoco, achieves 86.75 val accuracy
    """
    model = _make_detr("timm_tf_efficientnet_b3_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcoco_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_resnet101_refcocoplus(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on refcoco+, achieves 79.52 val accuracy
    """
    model = _make_detr("resnet101")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcoco%2B_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB3_refcocoplus(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Trained on refcoco+, achieves 81.13 val accuracy
    """
    model = _make_detr("timm_tf_efficientnet_b3_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcoco%2B_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_resnet101_refcocog(pretrained=False, return_postprocessor=False):
    """
    MDETR R101 with 6 encoder and 6 decoder layers.
    Trained on refcocog, achieves 81.64 val accuracy
    """
    model = _make_detr("resnet101")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcocog_resnet101_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model


def mdetr_efficientnetB3_refcocog(pretrained=False, return_postprocessor=False):
    """
    MDETR ENB3 with 6 encoder and 6 decoder layers.
    Trained on refcocog, achieves 83.35 val accuracy
    """
    model = _make_detr("timm_tf_efficientnet_b3_ns")
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    if return_postprocessor:
        return model, PostProcess()
    return model

def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('A', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']), strict=False)
    return model

def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('A', batch_norm=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']), strict=False)
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('B', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']), strict=False)
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('B', batch_norm=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']), strict=False)
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('D', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']), strict=False)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('D', batch_norm=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']), strict=False)
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('E', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG('E', batch_norm=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']), strict=False)
    return model


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


