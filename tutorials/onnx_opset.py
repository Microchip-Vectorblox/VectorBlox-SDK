import onnx
from onnx import version_converter
import argparse
import os
import subprocess
import json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx')
    parser.add_argument('-o', '--opset', type=int, default=13)

    args = parser.parse_args()

    try:
        model = onnx.load(args.onnx)
        converted_model = version_converter.convert_version(model, args.opset)
        onnx.save(converted_model, args.onnx.replace('.onnx', '.{}.onnx'.format(args.opset)))
    except:
        print('WARNING: offical conversion failed.')

        if args.opset == 13:
            print('Attempting manual conversion. Requires onnx2json and json2onnx.')
            subprocess.run(["onnx2json", '-if', args.onnx, '-oj', args.onnx.replace('.onnx', '.json')])
            with open(args.onnx.replace('.onnx', '.json')) as f:
                jgraph = json.load(f)
                jgraph['irVersion'] = 7
                jgraph['opsetImport'][0]['version'] = 13

                for n, node in enumerate(jgraph['graph']['node']):
                    if node['opType'] in ['Split']:
                        node['attribute'] = [_ for _ in node['attribute'] if not _['name'] in ['num_outputs']]

                    elif node['opType'] in ['Resize']:
                        valid_attributes = ["coordinate_transformation_mode", "cubic_coeff_a", "mode", "nearest_mode"]
                        node['attribute'] = [_ for _ in node['attribute'] if _['name'] in valid_attributes]

                    elif node['opType'] in ['AveragePool']:
                        node['attribute'] = [_ for _ in node['attribute'] if not _['name'] in ['auto_pad']]
                    elif node['opType'] in ['Reshape']:
                        node['attribute'] = [_ for _ in node['attribute'] if not _['name'] in ['allowzero']]

                    elif node['opType'] in ['Conv']:
                        node['attribute'] = [_ for _ in node['attribute'] if not _['name'] in ['auto_pad']]
                        for init in jgraph['graph']['initializer']:
                            if init['name'] == node['input'][1]:
                                node['attribute'].append({"name": "kernel_shape", "ints": init["dims"][-2:], "type": "INTS" })

                with open(args.onnx.replace('.onnx', '.13.json'), 'w') as f:
                    json.dump(jgraph, f)
                subprocess.run(["json2onnx", '-ij', args.onnx.replace('.onnx', '.13.json'), '-of', args.onnx.replace('.onnx', '.13.onnx')])
                print('Saving '+args.onnx.replace('.onnx', '.13.onnx'))
