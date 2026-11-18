import vbx.sim
import vbx.postprocess.dataset as dataset
import vbx.postprocess.classifier as classifier
import cv2
import numpy as np
import argparse
import os
import math
import json
import vbx.sim.model_run as mr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vbx.generate.utils import openvino_infer, openvino_input_shape
from vbx.generate.utils import onnx_infer, onnx_input_shape
from vbx.generate.utils import load_input, get_input_details, get_output_details
from vbx.generate.utils import existing_file


def plot(
    sample: np.ndarray, recon: np.ndarray, model_path: str, output_path: str
) -> None:
    """Plot input vs reconstruction and per-timestep anomaly score."""
    score = np.abs(sample - recon).mean(axis=-1)  # [T]

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(os.path.basename(model_path), fontsize=11)

    # — input (first 4 channels for readability)
    ax = axes[0]
    for c in range(min(4, sample.shape[1])):
        ax.plot(sample[:, c], alpha=0.7, linewidth=0.8, label=f"ch{c}")
    ax.set_ylabel("Input")
    ax.legend(loc="upper right", fontsize=7, ncol=4)

    # — reconstruction (same channels)
    ax = axes[1]
    for c in range(min(4, recon.shape[1])):
        ax.plot(recon[:, c], alpha=0.7, linewidth=0.8, linestyle="--")
    ax.set_ylabel("Reconstruction")

    # — anomaly score
    ax = axes[2]
    ax.fill_between(range(len(score)), score, alpha=0.6, color="crimson")
    ax.plot(score, color="crimson", linewidth=0.9)
    ax.set_ylabel("Anomaly score\n(mean |Δ|)")
    ax.set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def model_input_details(model):
    if model.endswith('.vnnx'):
        with open(model, 'rb') as mf:
            vnnx_model = vbx.sim.Model(mf.read())
        input_shape = vnnx_model.input_shape[0]
        quantization_scale, quantization_zero = vnnx_model.input_scale_factor[0], vnnx_model.input_zeropoint[0]
        input_dtype = vnnx_model.input_dtypes[0]

    elif model.endswith('.tflite'):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        interpreter= tf.lite.Interpreter(model_path=model)
        input_details = get_input_details(model)

        input_shape = input_details[0]["shape"]
        quantization_scale, quantization_zero = input_details[0].get('quantization', (0.0, 0))
        input_dtype = input_details[0]['dtype']

    return input_dtype, quantization_scale, quantization_zero


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=existing_file)
    parser.add_argument('sample', type=existing_file)
    parser.add_argument('--output', '-o', default="output.png", help='output image plot')
    args = parser.parse_args()

    # open sample and preprocess
    if not os.path.isfile(args.sample):
        print('Error: {} could not be read'.format(args.sample))
        os._exit(1)
    
    sample = np.load(args.sample).astype(np.float32)
    x = sample[np.newaxis].astype(np.float32)  # [1, T, C]
    dtype, sc, zp = model_input_details(args.model)
    if dtype == np.int8:
        x = np.clip(np.round(x / sc + zp), -128, 127).astype(np.int8)

    # run inference
    outputs, _ = mr.model_run(x, args.model)
    recon = outputs[0].squeeze()
   
    # plot outputs and draw
    score = np.abs(sample - recon).mean(axis=-1)
    print(f"  mean score : {score.mean():.4f}")
    print(f"  max  score : {score.max():.4f}")

    output_img = plot(sample, recon, args.model, args.output)
    print("Plot saved to", args.output)
    print()
