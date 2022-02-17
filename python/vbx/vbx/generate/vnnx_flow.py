import json
import os.path
import tempfile

from . import onnx_convert
from . import onnx_infer
from . import onnx_helper
from . import onnx_modify
from . import onnx_normalize
from . import onnx_to_json
from . import json_to_graph
from . import onnx_bias_correction


def generate_vnnx(xml_filename,
                  size_conf,
                  keep_temp=False,
                  image=None,
                  samples_folder=None,
                  samples_count=None,
                  output_filename=None,
                  cut_node=None,
                  bias_correction=False,
                  kld=False,
                  output_bytes=4):
    if keep_temp:
        tmp_dir ='keep_temp'
        try:
            os.mkdir(tmp_dir)
        except FileExistsError:
            pass

    else:
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp_dir = tmp_dir_obj.name
    model_name = os.path.join(tmp_dir, os.path.splitext(os.path.basename(xml_filename))[0])

    onnx_model = model_name + '.onnx'
    onnx_model_opt = model_name + '.opt.onnx'
    onnx_stats = model_name + '.statistics.json'
    onnx_model_pre = model_name + '.pre.onnx'
    onnx_model_norm = model_name + '.norm.onnx'
    onnx_model_post = model_name + '.post.onnx'
    onnx_model_biases = model_name + '.biases.onnx'
    io_json = model_name + '.io.json'
    graph_json = model_name + '.json'
    channels_json = onnx_model_post + '.channels.json'

    # convert from Openvino to ONNX
    nodes, ir_version = onnx_convert.parse_openvino_xml(xml_filename)
    if cut_node:
        nodes = onnx_convert.cut_after_node(nodes, cut_node)
    graph = onnx_convert.convert_openvino_xml_to_onnx(nodes, model_name, ir_version)
    onnx_helper.onnx_save_model(graph, onnx_model)
    onnx_modify.onnx_optimize_graph(onnx_model, onnx_model_opt)

    # gather stats
    # stats_np = onnx_infer.onnx_gather_stats(onnx_model_opt, nodes, samples_folder, samples_count, scale=None, kld_threshold=kld)
    # onnx_infer.onnx_save_stats(onnx_stats, stats_np)

    # activation/weight normalization
    onnx_modify.onnx_pre_graph(onnx_model_opt, onnx_model_pre)
    stats_np = onnx_infer.onnx_gather_stats(onnx_model_pre, nodes, samples_folder, samples_count, scale=None, kld_threshold=kld)
    onnx_infer.onnx_save_stats(onnx_stats, stats_np)

    input_scale_factors, output_scale_factors = onnx_normalize.run_normalize_graph(onnx_model_pre, onnx_stats, onnx_model_norm)
    onnx_modify.onnx_post_graph(onnx_model_norm, onnx_model_post)

    input_ids, output_ids = onnx_modify.onnx_get_io_ids(onnx_model_post)
    io_info = {'input_ids': input_ids, 'output_ids': output_ids, 'input_scale_factors': input_scale_factors, 'output_scale_factors': output_scale_factors}
    with open(io_json, 'w') as jf:
        json.dump(io_info, jf)

    # perform bias correction
    if bias_correction:
        onnx_modify.onnx_isolate_biases_graph(onnx_model_post, onnx_model_biases)
        json_string = onnx_to_json.run_generate_graph(onnx_model_biases, onnx_stats, io_info, image, ignore_strides=True)
        onnx_bias_correction.vnnx_bias_corrections(json_string, onnx_model_biases, size_conf, io_info, output_bytes, samples_folder, samples_count, tmp_dir)


    # convert ONNX to graph binary
    json_string = onnx_to_json.run_generate_graph(onnx_model_post, onnx_stats, io_info, image, inline_depthwise=True, remove_nops=True)
    with open(graph_json, 'w') as jf:
        jf.write(json_string)
    if bias_correction:
        bias_corrections = os.path.join(tmp_dir, 'bias_corrections.json')
        graph_binary = json_to_graph.json_to_graph(json_string, size_conf, io_info=io_info, output_bytes=output_bytes, bias_corrections=bias_corrections)
    else:
        graph_binary = json_to_graph.json_to_graph(json_string, size_conf, io_info=io_info, output_bytes=output_bytes)

    # clean up
    if not keep_temp:
        tmp_dir_obj.cleanup()

    if output_filename:
        import subprocess
        with open(output_filename, "wb") as output_file:
            output_file.write(graph_binary)
        hexfile = os.path.splitext(output_filename)[0]+".hex"
        subprocess.check_call(["objcopy", "-Ibinary","-Oihex", output_filename, hexfile])
    else:
        return graph_binary

