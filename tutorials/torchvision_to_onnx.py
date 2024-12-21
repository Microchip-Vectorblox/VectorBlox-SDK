
import argparse
import torch as nn
import torch.onnx
import sys
import torchvision.models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=sorted(torchvision.models.list_models()))
    parser.add_argument('-i', '--input-size', type=int, default=224)
    args = parser.parse_args()

    if args.model in["googlenet", "inception_v3"]:
        modelFunc = getattr(torchvision.models,args.model)
        model = modelFunc(pretrained=True)
        if hasattr(model,'transform_input'):
            model.transform_input = False
    else:
        model = torchvision.models.get_model(args.model, weights="DEFAULT")
        model.eval()

    x = torch.randn(1,3,args.input_size,args.input_size)
    torch.onnx.export(model, x, args.model+'.onnx')
    sys.exit(0)
