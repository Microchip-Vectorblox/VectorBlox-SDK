#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code is for loading a DarkNet model and converting it to an ONNX model
The ONNX model is meant for inference only
This is based on https://github.com/pjreddie/darknet
"""

import argparse
import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto
import os

def loadConfig(cfgFile):
    # parse_network_cfg(), parser.c 
    f = open(cfgFile)
    cfgLines=f.readlines()
    f.close()
    
    cfg = []
    for line in cfgLines:
        line = line.strip()
        if len(line)>0:
            if line[0]=='[':
                section = line.partition(']')[0][1:]
                cfg.append({'type':section})
            elif line[0] in '#;':
                continue
            else:
                option = line.partition('=')
                assert(len(option)==3)
                key = option[0].strip()
                value = option[2].strip()
                if value.isnumeric():
                    cfg[-1][key] = int(value)
                else:
                    try:
                        cfg[-1][key] = float(value)
                    except:
                        cfg[-1][key] = value
    return cfg


def setupNetwork(cfg):
    for n,sec in enumerate(cfg):
        if sec['type'] == 'net':
            # omitting options that are for training
            sec.setdefault('batch',1)
            h = sec.setdefault('height',0)
            w = sec.setdefault('width',0)
            c = sec.setdefault('channels',0)
            sec.setdefault('inputs',h*w*c)
            assert(h>0 and w>0 and c>0)
        elif sec['type'] == 'convolutional':
            n = sec.setdefault('filters',1)
            size = sec.setdefault('size',1)
            stride = sec.setdefault('stride',1)
            pad = sec.setdefault('pad',0)
            padding = sec.setdefault('padding',0)
            groups = sec.setdefault('groups',1)
            if pad:
                padding = int(size/2)
                sec['padding'] = padding
            sec.setdefault('activation','logistic')
            sec.setdefault('batch_normalize',0)
            sec.setdefault('binary',0)
            sec.setdefault('xnor',0)
            
            sec['h'] = h
            sec['w'] = w
            sec['c'] = c
            sec['nweights'] = int(c/groups)*n*size*size
            sec['nbiases'] = n
            w = int((w + 2*padding - size) / stride + 1)
            h = int((h + 2*padding - size) / stride + 1)
            c = n
            sec['out_w'] = w
            sec['out_h'] = h
            sec['out_c'] = c
            
            sec.setdefault('flipped',0)
            sec.setdefault('dot',0)
        elif sec['type'] == 'maxpool':
            stride = sec.setdefault('stride',1)
            size = sec.setdefault('size',stride)
            if 'padding' in sec:
                padding = sec['padding']
            else:
                padding = size-1    # this is the DarkNet default
                # sometimes the default padding is unnecessary though
                # this clause elimiates padding if it doesn't change the output size
                if int((w + padding - size)/stride) == int((w - size)/stride):
                    padding = 0
                sec['padding'] = padding
            
            sec['h'] = h
            sec['w'] = w
            sec['c'] = c
            w = int((w + padding - size)/stride + 1)
            h = int((h + padding - size)/stride + 1)
            sec['out_w'] = w
            sec['out_h'] = h
            sec['out_c'] = c
            
        elif sec['type'] == 'region':
            int(sec.setdefault('coords',4))
            int(sec.setdefault('classes',20))
            int(sec.setdefault('num',1))
            
            sec.setdefault('log',0)
            sec.setdefault('sqrt',0)
            sec.setdefault('softmax',0)
            sec.setdefault('background',0)
            sec.setdefault('max',30)
            sec.setdefault('jitter',.2)
            sec.setdefault('rescore',0)
            sec.setdefault('thresh',.5)
            sec.setdefault('classfix',0)
            sec.setdefault('absolute',0)
            sec.setdefault('random',0)
            
            sec.setdefault('coord_scale',1)
            sec.setdefault('object_scale',1)
            sec.setdefault('noobject_scale',1)
            sec.setdefault('mask_scale',1)
            sec.setdefault('class_scale',1)
            sec.setdefault('bias_match',0)
            
            sec.setdefault('tree',0)
            sec.setdefault('map',0)
            sec.setdefault('anchors',0)
            
            sec['h'] = h
            sec['w'] = w
            sec['c'] = c
        elif sec['type'] == 'yolo':
            sec.setdefault('classes',20)
            sec.setdefault('num',1)
            sec.setdefault('mask','')
            sec.setdefault('max',90)
            sec.setdefault('jitter',.2)
            sec.setdefault('ignore_thresh',.5)
            sec.setdefault('truth_thresh',1)
            sec.setdefault('random',0)
            sec.setdefault('map',0)
            
            sec['h'] = h
            sec['w'] = w
            sec['c'] = c
        elif sec['type'] == 'route':
            assert('layers' in sec)
            if isinstance(sec['layers'],str):
                routing = sec['layers'].split(',')
                sec['layers'] = []
                for layerInd in routing:
                    sec['layers'].append(int(layerInd))
                    
            else:
                sec['layers'] = [int(sec['layers'])]
            c = 0
            for ind,layerNum in enumerate(sec['layers']):
                if layerNum<0:
                    layerNum += (n-1)
                    sec['layers'][ind] = layerNum
                cfgNum = layerNum+1
                h = cfg[cfgNum]['out_h']
                w = cfg[cfgNum]['out_w']
                c += cfg[cfgNum]['out_c']
            sec['out_w'] = w
            sec['out_h'] = h
            sec['out_c'] = c
        elif sec['type'] == 'upsample':
            stride = sec.setdefault('stride',2)
            sec.setdefault('scale',1)
            
            sec['h'] = h
            sec['w'] = w
            sec['c'] = c
            if stride<0:
                stride = -stride
                w = int(w/stride)
                h = int(h/stride)
            else:
                w = int(w*stride)
                h = int(h*stride)
            sec['out_w'] = w
            sec['out_h'] = h
            sec['out_c'] = c
        elif sec['type'] == 'reorg':
            stride = sec.setdefault('stride',1)
            reverse = sec.setdefault('reverse',0)
            flatten = sec.setdefault('flatten',0)
            extra = sec.setdefault('extra',0)
            
            sec['h'] = h
            sec['w'] = w
            sec['c'] = c
            if reverse:
                w = int(w*stride)
                h = int(h*stride)
                c = int(c/(stride*stride))
            else:
                w = int(w/stride)
                h = int(h/stride)
                c = int(c*(stride*stride))
            sec['out_h'] = h
            sec['out_w'] = w
            sec['out_c'] = c
            
            # other settings are possible in darknet, but not currently support by this converter
            assert(reverse==0)
            assert(flatten==0)
            assert(extra==0)
        elif sec['type'] == 'shortcut':
            sec['index'] = int(sec['from'])
            if sec['index'] < 0:
                sec['index'] += (n-1)
            sec.setdefault('activation','logistic')
            sec.setdefault('alpha',1)
            sec.setdefault('beta',1)
            
            cfgNum = sec['index']+1
            sec['h'] = cfg[cfgNum]['out_h']
            sec['w'] = cfg[cfgNum]['out_w']
            sec['c'] = cfg[cfgNum]['out_c']
            sec['out_h'] = h
            sec['out_w'] = w
            sec['out_c'] = c
            
            # other settings are possible in darknet, but not currently support by this converter
            assert(sec['h']==sec['out_h'])
            assert(sec['w']==sec['out_w'])
            assert(sec['c']==sec['out_c'])
        else:
            raise NameError('unknown section type: {}'.format(sec['type']))
            
    net = cfg[1:]   # remove first section so that list index == layer number
    return net


def printLayers(net):
    print('layer type                 input                output')
    for n,sec in enumerate(net):
        if ('c' in sec) and ('out_c' in sec):
            print('{:5} {:15} {:4} x{:4} x{:4}  ->  {:4} x{:4} x{:4}'.format(n,sec['type'],sec['h'],sec['w'],sec['c'],sec['out_h'],sec['out_w'],sec['out_c']))
        elif 'out_c' in sec:
            print('{:5} {:15}                   ->  {:4} x{:4} x{:4}'.format(n,sec['type'],sec['out_h'],sec['out_w'],sec['out_c']))
        elif 'c' in sec:
            print('{:5} {:15} {:4} x{:4} x{:4}  ->                  '.format(n,sec['type'],sec['h'],sec['w'],sec['c']))
        else:
            print('{:5} {:15}'.format(n,sec['type']))


def loadWeights(net,weightsFile):

    def transpose_matrix(a, rows, cols):
        transpose = np.zeros(rows*cols, dtype=np.float32)
        for x in range(rows):
            for y in range(cols):
                transpose[y*rows + x] = a[x*cols + y]
        return transpose
    
    f = open(weightsFile, "rb")
    major = np.fromfile(f, dtype=np.int32, count=1)[0]
    minor = np.fromfile(f, dtype=np.int32, count=1)[0]
    revision = np.fromfile(f, dtype=np.int32, count=1)[0]
        
    if ((major*10 + minor) >= 2 and major < 1000 and minor < 1000):
        seen =  np.fromfile(f, dtype=np.int64, count=1)[0]
    else:
        seen =  np.fromfile(f, dtype=np.int32, count=1)[0]
    transpose = (major > 1000) or (minor > 1000)
    
    for sec in net:
        if sec['type'] in ['convolutional','deconvolutional']:
            #load_convolutional_weights(), parser.c
            n = sec['nbiases']
            sec['biases'] = np.fromfile(f, dtype=np.float32, count=n)
            if sec['batch_normalize']:  #&& (!l.dontloadscales)){
                sec['scales'] = np.fromfile(f, dtype=np.float32, count=n)
                sec['rolling_mean'] = np.fromfile(f, dtype=np.float32, count=n)
                sec['rolling_variance'] = np.fromfile(f, dtype=np.float32, count=n)
            sec['weights'] = np.fromfile(f, dtype=np.float32, count=sec['nweights'])
            if sec['flipped']:
                sec['weights'] = transpose_matrix(sec['weights'], sec['c']*sec['size']*sec['size'], n)    
    f.close()


def convertToOnnx(net, onnxFile):
    inputShape = ['N',net[0]['c'],net[0]['h'],net[0]['w']]
    X = helper.make_tensor_value_info('X0', TensorProto.FLOAT, inputShape)  # input tensor to the next layer
    inputs = [X]    # graph inputs
    nodes = []      # graph nodes
    inits = []      # graph parameters (weights, biases)
    outputs = []    # graph outputs
    tensors = {}    # temp storage for outputs in case they are used again
    for n,sec in enumerate(net):
        if sec['type'] in ['region','yolo']:
            outputs.append(X)
        elif sec['type'] == 'convolutional':
            # The conv and batch norm layers could be fused. The DarkNet formula for both layers is:
            #   out = ((W*X-mean)/(sqrt(var)+.000001))*scale + bias
            # A fused conv layer would be:
            #   out = W*X*S + bias-mean*S,   S=scale/(sqrt(var)+.000001)
            
            W = helper.make_tensor('W'+str(n), TensorProto.FLOAT, [sec['filters'],sec['c'],sec['size'],sec['size']], sec['weights'])
            B = helper.make_tensor('B'+str(n), TensorProto.FLOAT, [sec['filters']], sec['biases'])
            Y = helper.make_tensor_value_info('Y'+str(n), TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
            nodeInputs = [X.name,W.name]
            if not sec['batch_normalize']:
                nodeInputs.append(B.name)
            node = helper.make_node('Conv',
                                    nodeInputs,
                                    [Y.name],
                                    'Conv'+str(n),
                                    kernel_shape=[sec['size'],sec['size']],
                                    pads=[sec['padding'],sec['padding'],sec['padding'],sec['padding']],
                                    strides=[sec['stride'],sec['stride']])
            nodes.append(node)
            inits.extend([W, B])
            
            if sec['batch_normalize']:
                X = Y
                S = helper.make_tensor('scales'+str(n), TensorProto.FLOAT, [sec['filters']], sec['scales'])
                M = helper.make_tensor('mean'+str(n), TensorProto.FLOAT, [sec['filters']], sec['rolling_mean'])
                V = helper.make_tensor('var'+str(n), TensorProto.FLOAT, [sec['filters']], sec['rolling_variance'])
                Y = helper.make_tensor_value_info('Y'+str(n)+'Norm', TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
                node = onnx.helper.make_node('BatchNormalization',
                                              inputs=[X.name, S.name, B.name, M.name, V.name],
                                              outputs=[Y.name],
                                              name='BatchNormalization'+str(n),
                                              epsilon=0.000001)   # DarkNet default
                nodes.append(node)
                inits.extend([S,M,V])
                
        elif sec['type'] == 'maxpool':
            Y = helper.make_tensor_value_info('Y'+str(n), TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
            pad1 = int(np.floor(sec['padding']/2))
            pad2 = int(np.ceil(sec['padding']/2))
            node = helper.make_node('MaxPool',
                                    [X.name],
                                    [Y.name],
                                    'MaxPool'+str(n),
                                    kernel_shape=[sec['size'],sec['size']],
                                    pads=[pad1,pad1,pad2,pad2],
                                    strides=[sec['stride'],sec['stride']])
            nodes.append(node)
            
        elif sec['type'] == 'route':
            if len(sec['layers']) == 1: # if there's only one layer being routed, just assign the tensor
                layerNum = sec['layers'][0]
                Y = tensors[net[layerNum]['outputName']]
            else:   # if there are multiple layers, need to concatenate
                inputNames = []
                for layerNum in sec['layers']:
                    inputNames.append(net[layerNum]['outputName'])
                Y = helper.make_tensor_value_info('Y'+str(n), TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
                node = helper.make_node('Concat',
                                    inputNames,
                                    [Y.name],
                                    'Concat'+str(n),
                                    axis = 1)
                nodes.append(node)
                
        elif sec['type'] == 'upsample':
            Y = helper.make_tensor_value_info('Y'+str(n), TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
            scale = sec['stride']
            if scale<0:
                scale = 1.0/-scale
            Scales = helper.make_tensor('Scales'+str(n), TensorProto.FLOAT, [4], [1,1,scale,scale])
            node = helper.make_node('Resize',           # this is the Resize-10 interface
                                    [X.name,Scales.name],
                                    [Y.name],
                                    'Resize'+str(n))
            nodes.append(node)
            inits.append(Scales)
            
        elif sec['type'] == 'reorg':
            # https://github.com/thtrieu/darkflow/issues/173
            # channel_first = keras.layers.Permute((3, 1, 2))(input_tensor)            
            # reshape_tensor = keras.layers.Reshape((c // (stride ** 2), h, stride, w, stride))(channel_first)
            # permute_tensor = keras.layers.Permute((3, 5, 1, 2, 4))(reshape_tensor)
            # target_tensor = keras.layers.Reshape((-1, h // stride, w // stride))(permute_tensor)
            # channel_last = keras.layers.Permute((2, 3, 1))(target_tensor)
            # return keras.layers.Reshape((h // stride, w // stride, -1))(channel_last)
            
            s = sec['stride']
            h = sec['h']
            w = sec['w']
            c = sec['c']
            if 0:
                Shape1 = helper.make_tensor('Reorg'+str(n)+'_1Shape', TensorProto.INT64, [6], [-1,c//(s**2),h,s,w,s])
                Shape3 = helper.make_tensor('Reorg'+str(n)+'_3Shape', TensorProto.INT64, [4], [-1,c*(s**2),h//s,w//s])
                Y1name = 'Reorg'+str(n)+'_1Y'
                Y2name = 'Reorg'+str(n)+'_2Y'
                Y  = helper.make_tensor_value_info('Reorg'+str(n)+'_3Y', TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
                node1 = helper.make_node('Reshape', [X.name,Shape1.name], [Y1name], 'Reorg'+str(n)+'_1Reshape')
                node2 = helper.make_node('Transpose', [Y1name], [Y2name], name='Reorg'+str(n)+'_2Transpose', perm=[0,3,5,1,2,4])
                node3 = helper.make_node('Reshape', [Y2name,Shape3.name], [Y.name], 'Reorg'+str(n)+'_3Reshape')
                nodes.extend([node1,node2,node3])
                inits.extend([Shape1,Shape3])
            elif 0:
                Shape1 = helper.make_tensor('Reorg'+str(n)+'_1Shape', TensorProto.INT64, [5], [-1,c//(s**2)*h,s,w,s])
                Shape3 = helper.make_tensor('Reorg'+str(n)+'_3Shape', TensorProto.INT64, [4], [-1,c*(s**2),h//s,w//s])
                Y1name = 'Reorg'+str(n)+'_1Y'
                Y2name = 'Reorg'+str(n)+'_2Y'
                Y  = helper.make_tensor_value_info('Reorg'+str(n)+'_3Y', TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
                node1 = helper.make_node('Reshape', [X.name,Shape1.name], [Y1name], 'Reorg'+str(n)+'_1Reshape')
                node2 = helper.make_node('Transpose', [Y1name], [Y2name], name='Reorg'+str(n)+'_2Transpose', perm=[0,2,4,1,3])
                node3 = helper.make_node('Reshape', [Y2name,Shape3.name], [Y.name], 'Reorg'+str(n)+'_3Reshape')
                nodes.extend([node1,node2,node3])
                inits.extend([Shape1,Shape3])
            elif 0:
                Shape1 = helper.make_tensor('Reorg'+str(n)+'_1Shape', TensorProto.INT64, [5], [-1,c//(s**2)*h,s,w,s])
                Shape5 = helper.make_tensor('Reorg'+str(n)+'_5Shape', TensorProto.INT64, [4], [-1,c*(s**2),h//s,w//s])
                Y1name = 'Reorg'+str(n)+'_1Y'
                Y2name = 'Reorg'+str(n)+'_2Y'
                Y3name = 'Reorg'+str(n)+'_3Y'
                Y4name = 'Reorg'+str(n)+'_4Y'
                Y  = helper.make_tensor_value_info('Reorg'+str(n)+'_5Y', TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
                node1 = helper.make_node('Reshape', [X.name,Shape1.name], [Y1name], 'Reorg'+str(n)+'_1Reshape')
                node2 = helper.make_node('Transpose', [Y1name], [Y2name], name='Reorg'+str(n)+'_2Transpose', perm=[0,1,2,4,3])
                node3 = helper.make_node('Transpose', [Y2name], [Y3name], name='Reorg'+str(n)+'_3Transpose', perm=[0,2,1,3,4])
                node4 = helper.make_node('Transpose', [Y3name], [Y4name], name='Reorg'+str(n)+'_4Transpose', perm=[0,1,3,2,4])
                node5 = helper.make_node('Reshape', [Y4name,Shape5.name], [Y.name], 'Reorg'+str(n)+'_5Reshape')
                nodes.extend([node1,node2,node3,node4,node5])
                inits.extend([Shape1,Shape5])
            else:
                Shape1 = helper.make_tensor('Reorg'+str(n)+'_1Shape', TensorProto.INT64, [4], [-1,c//(s**2)*h*s,w,s])
                Shape3 = helper.make_tensor('Reorg'+str(n)+'_3Shape', TensorProto.INT64, [4], [-1,c//(s**2)*h,s,s*w])
                Shape5 = helper.make_tensor('Reorg'+str(n)+'_5Shape', TensorProto.INT64, [5], [-1,s,c//(s**2)*h,s,w])
                Shape7 = helper.make_tensor('Reorg'+str(n)+'_7Shape', TensorProto.INT64, [4], [-1,c*(s**2),h//s,w//s])
                Y1name = 'Reorg'+str(n)+'_1Y'
                Y2name = 'Reorg'+str(n)+'_2Y'
                Y3name = 'Reorg'+str(n)+'_3Y'
                Y4name = 'Reorg'+str(n)+'_4Y'
                Y5name = 'Reorg'+str(n)+'_5Y'
                Y6name = 'Reorg'+str(n)+'_6Y'
                Y  = helper.make_tensor_value_info('Reorg'+str(n)+'_7Y', TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
                node1 = helper.make_node('Reshape', [X.name,Shape1.name], [Y1name], 'Reorg'+str(n)+'_1Reshape')
                node2 = helper.make_node('Transpose', [Y1name], [Y2name], name='Reorg'+str(n)+'_2Transpose', perm=[0,1,3,2])
                node3 = helper.make_node('Reshape', [Y2name,Shape3.name], [Y3name], 'Reorg'+str(n)+'_3Reshape')
                node4 = helper.make_node('Transpose', [Y3name], [Y4name], name='Reorg'+str(n)+'_4Transpose', perm=[0,2,1,3])
                node5 = helper.make_node('Reshape', [Y4name,Shape5.name], [Y5name], 'Reorg'+str(n)+'_5Reshape')
                node6 = helper.make_node('Transpose', [Y5name], [Y6name], name='Reorg'+str(n)+'_6Transpose', perm=[0,1,3,2,4])
                node7 = helper.make_node('Reshape', [Y6name,Shape7.name], [Y.name], 'Reorg'+str(n)+'_7Reshape')
                nodes.extend([node1,node2,node3,node4,node5,node6,node7])
                inits.extend([Shape1,Shape3,Shape5,Shape7])
            
        elif sec['type'] == 'shortcut':
            # shortcut_cpu(l.batch, l.w, l.h, l.c, net.layers[l.index].output, l.out_w, l.out_h, l.out_c, l.alpha, l.beta, l.output);
            Y  = helper.make_tensor_value_info('Y'+str(n), TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
            node = helper.make_node('Add', [X.name,net[sec['index']]['outputName']], [Y.name], 'Add'+str(n))
            nodes.append(node)
            
        else:
            raise NameError('unknown section type: {}'.format(sec['type']))
        
        if 'activation' in sec:
            if sec['activation'] == 'leaky':
                X = Y
                Y = helper.make_tensor_value_info('Y'+str(n)+'Relu', TensorProto.FLOAT, ['N',sec['out_c'],sec['out_h'],sec['out_w']])
                node = onnx.helper.make_node('LeakyRelu',
                                             inputs=[X.name],
                                             outputs=[Y.name],
                                             name='LeakyRelu'+str(n),
                                             alpha=0.1)     # DarkNet default is 0.1
                nodes.append(node)
            elif sec['activation'] != 'linear':
                raise NameError('unsupported activation type: {}'.format(sec['activation']))
        
        tensors[Y.name] = Y     # store this in case it is used again in a "route" or "shortcut" layer
        sec['outputName'] = Y.name
        X = Y   # this layer's output is next layer's input
        
    graph_def = helper.make_graph(nodes, 'darknet conversion', inputs, outputs, inits)
    opset = helper.make_operatorsetid('',10)
    model_def = helper.make_model(graph_def, producer_name='darknetToOnnx',opset_imports=[opset])
    onnx.checker.check_model(model_def)
    onnx.save_model(model_def, onnxFile)
    print('saved onnx file: '+onnxFile)


def writeJson(cfg,jsonFile):
    import json
    
    ioCfg = []
    for n,layer in enumerate(cfg):
        if layer['type'] in ['net','region','yolo']:
            ioCfg.append(layer)
    
    with open(jsonFile, 'w') as outFile:
        json.dump(ioCfg, outFile, indent=4)
    
    # yoloLayer = net[-1]['type']=='yolo'
    # if yoloLayer:
    #     masks = '['
    #     entryPoints = '['
    #     for sec in net:
    #         if sec['type']=='yolo':
    #             if masks[-1]==']':
    #                 masks += ', '
    #             masks += '['+sec['mask']+']'
    #             if entryPoints[-1]=='\"':
    #                 entryPoints += ', '
    #             entryPoints += '\"'+sec['outputName']+'\"'
    #     masks += ']'
    #     entryPoints += ']'
    
    # f = open(jsonFile,'wt')
    # f.write('[\n')
    # f.write('  {\n')
    # f.write('    "id": "ONNXYOLO",\n')
    # f.write('    "match_kind": "general",\n')
    # f.write('    "custom_attributes": {\n')
    # f.write('      "anchors": [{}],\n'.format(net[-1]['anchors']))
    # f.write('      "classes": {},\n'.format(net[-1]['classes']))
    # f.write('      "num": {},\n'.format(net[-1]['num']))
    # if yoloLayer:
    #     f.write('      "masks": {},\n'.format(masks))
    #     f.write('      "entryPoints": {}\n'.format(entryPoints))
    #     # the previous yoloV3 json file has ""coords": 4,", but there is no "coords" in the yolo layer
    # else:
    #     f.write('      "coords": {},\n'.format(net[-1]['coords']))
    #     f.write('      "do_softmax": {}\n'.format(net[-1]['softmax']))
    # f.write('    }\n')
    # f.write('  }\n')
    # f.write(']\n')
    # f.close()
    # print('saved json file: '+jsonFile)


def darknetToOnnx(cfgFile,weightsFile,onnxFile,jsonFile,verbose=False):
    cfg = loadConfig(cfgFile)
    net = setupNetwork(cfg)
    if verbose:
        printLayers(net)
    if os.path.exists(weightsFile):
        loadWeights(net,weightsFile)
        convertToOnnx(net,onnxFile)
    else:
        print("WARNING! weights not found {}".format(weightsFile))
    writeJson(cfg,jsonFile)
    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('--weights', default='')
    parser.add_argument('--onnx', default='')
    parser.add_argument('--json', default='')
    args = parser.parse_args()
    if args.weights=='':
        args.weights = os.path.splitext(args.cfg)[0]+'.weights'
    if args.onnx=='':
        args.onnx = os.path.splitext(args.cfg)[0]+'.onnx'
    if args.json=='':
        args.json = os.path.splitext(args.cfg)[0]+'.json'
        
    darknetToOnnx(args.cfg, args.weights, args.onnx, args.json)
    
