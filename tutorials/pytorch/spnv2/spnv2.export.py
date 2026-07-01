import torch, types
from tools import _init_paths
from config import cfg, update_config
from nets import build_spnv2
import json

yaml = 'experiments/offline_train_full_config_phi3_BN.yaml'
weights = '../spnv2_efficientnetb3_fullconfig_offline.pth.tar'
width, height = 768, 512

# load model
update_config(cfg, types.SimpleNamespace(cfg=yaml, opts=[]))


camera = {"Nu": 1920, "Nv": 1200, "ppx": 5.86E-6, "ppy": 5.86E-6,
        "fx": 0.017513075965995915, "fy": 0.017511673079277208, "ccx": 960, "ccy": 600,
        "cameraMatrix": [ [ 2988.5795163815555, 0, 960 ], [ 0, 2988.3401159176124, 600 ], [ 0, 0, 1 ] ],
        "distCoeffs": [ -0.22383016606510672, 0.51409797089106379, -0.00066499611998340662, -0.00021404771667484594, -0.13124227429077406 ]}

with open('../camera.json', 'w') as f:
    json.dump(camera, f)

cfg.defrost()
cfg.DATASET.CAMERA = '../camera.json'
cfg.TEST.MODEL_FILE = weights
cfg.freeze()

model = build_spnv2(cfg)
model.load_state_dict(torch.load(weights, map_location='cpu'), strict=True)

# export backbone and heads
torch.onnx.export(model.backbone, torch.rand((1, 3, height, width)), "spnv2.backbone.onnx")
outputs = [torch.rand((1,160,height // 4 // 2**i, width//4 // 2**i)) for i in range(6)]
for h,head in enumerate(cfg.MODEL.HEAD.NAMES): 
    torch.onnx.export(model.heads[h], (outputs,), "spnv2.head.{}.onnx".format(head))

# merge backbone w/ efficientpose head
import onnx
m1 = onnx.load('spnv2.backbone.onnx')
m2 = onnx.load('spnv2.head.efficientpose.onnx')

shape = lambda x: [_.dim_value for _ in x.type.tensor_type.shape.dim]
pairs = []
for inp in m2.graph.input:
    for outp in m1.graph.output:
        if shape(inp) == shape(outp):
            pairs.append((outp.name, inp.name))
mm = onnx.compose.merge_models(m1,m2,pairs)
onnx.save_model(mm, 'spnv2.onnx')
