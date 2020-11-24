import argparse
import torch as nn
import torch.onnx
import sys
import torchvision.models
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    models= {'resnet18':224,
             'resnet34':224,
             'resnet50':224,
             'resnet101':224,
             'resnet152':224,
             'resnext50_32x4d':224,
             'resnext101_32x8d':224,
             'wide_resnet50_2':224,
             'wide_resnet101_2':224,
             'alexnet':227,
             'googlenet':224,
             'inception_v3':299,
             'mnasnet0_5':224,
             'mnasnet0_75':224,
             'mnasnet1_0':224,
             'mnasnet1_3':224,
             'squeezenet1_0':227,
             'squeezenet1_1':227,
             'densenet121':224,
             'densenet161':224,
             'densenet169':224,
             'densenet201':224,
             'mobilenet_v2':224,
             'vgg11':224,
             'vgg11_bn':224,
             'vgg13':224,
             'vgg13_bn':224,
             'vgg16':224,
             'vgg16_bn':224,
             'vgg19':224,
             'vgg19_bn':224,
             'shufflenet_v2_x0_5':224,
             'shufflenet_v2_x1_0':224,
             'shufflenet_v2_x1_5':224,
             'shufflenet_v2_x2_0':224,
             'segmentation.deeplabv3_resnet101':513,
             #'segmentation.deeplabv3_resnet50':513
    }

    parser.add_argument('model', choices=sorted(models.keys()))
    
    args = parser.parse_args()
    modelFunc = getattr(torchvision.models,args.model)
    model = modelFunc(pretrained=True)
    if hasattr(model,'transform_input'):
        model.transform_input = False
    input_size = models[args.model]
    x = torch.randn(1,3,input_size,input_size)
    torch.onnx.export(model, x, args.model+'.onnx')
    sys.exit(0)
