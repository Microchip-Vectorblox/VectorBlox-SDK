import numpy as np
from scipy.stats import entropy
import onnx
import mxnet as mx
from . import onnx_helper


"""
code taken from mxnet framework (1.5.1)
https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/contrib/quantization.py
"""
def get_optimal_threshold(hist_data, num_quantized_bins=255, use_unsigned=True):
    hist, hist_edges, min_val, max_val, _ = hist_data
    num_bins = len(hist)

    if use_unsigned and min_val >= 0:
        num_quantized_bins = num_quantized_bins * 2 + 1
    elif min_val >= 0:
        num_quantized_bins = num_quantized_bins * 2 + 1

    return mx.contrib.quantization._get_optimal_threshold(hist_data, 'int8', num_quantized_bins)


def combine_histogram(old_hist, arr, new_min, new_max, new_th):
    """ Collect layer histogram for arr and combine it with old histogram.
    """
    (old_hist, old_hist_edges, old_min, old_max, old_th) = old_hist
    if new_th <= old_th:
        hist, _ = np.histogram(arr, bins=len(old_hist), range=(-old_th, old_th))
        return (old_hist + hist, old_hist_edges, min(old_min, new_min), max(old_max, new_max), old_th)
    else:
        # Need to generate new histogram with new_th
        old_num_bins = len(old_hist)
        old_step = 2 * old_th / old_num_bins
        half_increased_bins = int((new_th - old_th) // old_step + 1)
        new_num_bins = half_increased_bins * 2 + old_num_bins
        new_th = half_increased_bins * old_step + old_th
        hist, hist_edges = np.histogram(arr, bins=new_num_bins, range=(-new_th, new_th))
        hist[half_increased_bins:new_num_bins - half_increased_bins] += old_hist
        return (hist, hist_edges, min(old_min, new_min), max(old_max, new_max), new_th)


def collect_histogram_data(arr, old_hist=None, num_bins=8001):
    min_range = np.min(arr)
    max_range = np.max(arr)
    th = max(abs(min_range), abs(max_range))
    if old_hist:
        return combine_histogram(old_hist, arr, min_range, max_range, th)
    else:
        hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-th, th))
        return (hist, hist_edges, min_range, max_range, th)



def get_valid_kld(onnx_model):
    valid_kld = {}

    graph = onnx.load(onnx_model).graph
    input_ids = [_.name for _ in graph.input]
    output_ids = [_.name for _ in graph.output]
    nodes = graph.node
    for n, node in enumerate(nodes):
        next_nodes = onnx_helper.get_node_inputs(nodes, node.output[0])

        # if node.name not in output_ids:
        # if node.op_type in ['Conv', 'Gemm', 'Sum']:
        #     valid_kld[node.output[0]] = 255
        #     if len(next_nodes) == 1 and next_nodes[0].op_type in ['Relu', 'LeakyRelu']:
        #         # valid_kld[next_nodes[0].output[0]] = 511
        #         valid_kld[node.output[0]] = 511
            # else:
            #     valid_kld[node.output[0]] = 255

        valid_kld[node.output[0]] = 255
        if n == 0:
            valid_kld[node.input[0]] = 255


    return valid_kld
