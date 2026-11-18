"""yolo_tflite.py — YOLO11n / YOLO26n fp32 export + aeq INT8 TFLite (zp=0 BATCH_MATMUL).

Reproduces yolo11n_aeq_softmax_add_int8.tflite and yolo26n_aeq_softmax_add_int8.tflite:
  ultralytics <model>.pt -> einsum attention -> fp32 TFLite
  -> ai_edge_quantizer (symmetric ZP=0 attention chain, Softmax ADD, PE Conv2D)
  -> float32 NCHW input/output.

Usage:
    python yolo_tflite.py --model yolo11n --calib 128     # auto-downloads COCO128
    python yolo_tflite.py --model yolo26n --calib 128
Requires: uv sync --group cpu
"""

import argparse, os, tempfile
import numpy as np
import torch
from torchao.quantization.pt2e.utils import _fuse_conv_bn_
import litert_torch



def _attn_forward_einsum(self, x: torch.Tensor) -> torch.Tensor:
    """Replaces Attention.forward with torch.einsum to emit 4D BATCH_MATMUL.

    The original @-operator path collapses B*H to 3D before BATCH_MATMUL.
    torch.einsum keeps the [B, H, *, *] shape, matching anomaly_transformer
    and VIT (_CleanMHA) patterns that VectorBlox parses cleanly.

    TFLite pattern produced:
      CONV(QKV) → TRANSPOSE → RESHAPE → SLICE×3
      einsum "bhki,bhkj->bhij" → BATCH_MATMUL [B,H,N,N]
      MUL(scale) → SOFTMAX
      einsum "bhdi,bhji->bhdj" → BATCH_MATMUL [B,H,hd,N]
      VIEW → ADD(pe) → PROJ
    """
    B, C, H, W = x.shape
    N = H * W
    qkv = self.qkv(x)
    q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
        [self.key_dim, self.key_dim, self.head_dim], dim=2
    )
    # BMM1: Q^T · K → [B, H, N, N]  (contract over key_dim k)
    attn = torch.einsum("bhki,bhkj->bhij", q, k) * self.scale
    attn = attn.softmax(dim=-1)
    # A tiny nonzero offset preserves an explicit ADD boundary after Softmax.
    attn = attn + self._attn_zero_const.to(device=attn.device, dtype=attn.dtype)
    # BMM2: V · attn^T → [B, H, head_dim, N]  (contract over N j)
    x = torch.einsum("bhdi,bhji->bhdj", v, attn).view(B, C, H, W)
    x = x + self.pe(v.reshape(B, C, H, W))
    return self.proj(x)


def _patch_attention_einsum(model: torch.nn.Module) -> int:
    """Replace Attention.forward with einsum version for 4D BATCH_MATMUL output.

    Returns the number of Attention modules patched.
    """
    from ultralytics.nn.modules.block import Attention
    import types

    count = 0
    for mod in model.modules():
        if isinstance(mod, Attention):
            if not hasattr(mod, "_attn_zero_const"):
                mod.register_buffer("_attn_zero_const", torch.full((1,), 1e-6))
            mod.forward = types.MethodType(_attn_forward_einsum, mod)
            count += 1
    return count


def export_fp32(model_name: str, out_path: str, imgsz: int = 640) -> str:
    """Load YOLO .pt, export full model (backbone + neck + head) as fp32 TFLite."""
    from ultralytics import YOLO

    yolo = YOLO(f"{model_name}.pt")
    model = yolo.model.eval().cpu()

    dummy = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        model(dummy)

    n = _patch_attention_einsum(model)
    if n:
        print(f"[attn]  patched {n} Attention module(s) → einsum 4D BATCH_MATMUL")
        with torch.no_grad():
            model(dummy)  # verify forward still works

    # Strip end-to-end NMS if present (yolo26n has end2end=True; yolo11n does not).
    # NMS ops (topk, gather, conditional) don't quantize cleanly with aeq.
    if getattr(model, "end2end", False):
        model.end2end = False
        print("[e2e]   stripped end-to-end NMS → raw head outputs for aeq")
        with torch.no_grad():
            model(dummy)  # re-verify

    # Wrap in a single-output module so torch.export captures only the fused
    # detection tensor [1, 84, 8400], not the intermediate DFL / feature-map
    # outputs that leak through when exporting the raw DetectionModel.
    # Ultralytics' own TFLite export does the same (1 output, INT8, NHWC).
    class _SingleOut(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x)[0]

    wrapped = _SingleOut(model).eval()
    ep = torch.export.export(wrapped, (dummy,)).module()
    _fuse_conv_bn_(ep)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    litert_torch.convert(ep, (dummy,)).export(out_path)
    print(f"[fp32]  {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def quantize(
    fp32_path: str,
    out_path: str,
    calib_data: list[dict],
) -> str:
    """Quantize fp32 TFLite to symmetric INT8 via ai_edge_quantizer.

    Symmetric (ZP=0) activations → BATCH_MATMUL LHS and RHS at ZP=0.
    ADD(zero) after attention SOFTMAX ensures the AV BMM first input
    stays at ZP=-128 (affine min=0 for [0,1] range), which is correct.

    INPUT and OUTPUT ops are kept fp32 (NO_QUANTIZE), matching the
    ultralytics litert export convention: callers pass normalized
    float32 NCHW and receive float32 output — no int8 boundary handling.
    """
    from ai_edge_quantizer import quantizer as aeq_mod
    from ai_edge_quantizer.qtyping import (
        TFLOperationName,
        OpQuantizationConfig,
        TensorQuantizationConfig,
        ComputePrecision,
        QuantGranularity,
    )

    # Global recipe: affine (asymmetric, ZP≠0) for all activations.
    # Affine correctly represents the non-zero-mean distributions of SiLU
    # outputs without wasting half the INT8 range on the negative side.
    # Weights stay symmetric (standard practice, no ZP for weights).
    recipe = [
        {
            "regex": ".*",
            "operation": "*",
            "algorithm_key": "min_max_uniform_quantize",
            "op_config": {
                "activation_tensor_config": {
                    "num_bits": 8,
                    "symmetric": False,  # affine ZP≠0 for all activations
                    "granularity": "TENSORWISE",
                    "dtype": "INT",
                    "block_size": 0,
                },
                "weight_tensor_config": {
                    "num_bits": 8,
                    "symmetric": True,
                    "granularity": "CHANNELWISE",
                    "dtype": "INT",
                    "block_size": 0,
                },
                "compute_precision": "INTEGER",
                "explicit_dequantize": False,
                "skip_checks": False,
            },
        }
    ]

    qt = aeq_mod.Quantizer(fp32_path)
    qt.load_quantization_recipe(recipe)

    # BATCH_MATMUL override: symmetric (ZP=0) on both inputs.
    # The attention QK^T and AV matmuls require ZP=0 on LHS and RHS so that
    # the zero-point correction term in the INT8 matmul expansion vanishes.
    _sym8 = TensorQuantizationConfig(
        num_bits=8, symmetric=True, granularity=QuantGranularity.TENSORWISE
    )
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=TFLOperationName.BATCH_MATMUL,
        algorithm_key="min_max_uniform_quantize",
        op_config=OpQuantizationConfig(
            activation_tensor_config=_sym8,
            weight_tensor_config=TensorQuantizationConfig(
                num_bits=8,
                symmetric=True,
                granularity=QuantGranularity.CHANNELWISE,
            ),
            compute_precision=ComputePrecision.INTEGER,
        ),
    )

    # Keep fp32 input/output — matches ultralytics litert convention:
    # callers pass normalized float32 NCHW, receive float32 output.
    for op in (TFLOperationName.INPUT, TFLOperationName.OUTPUT):
        qt.update_quantization_recipe(
            regex=".*",
            operation_name=op,
            algorithm_key="no_quantize",
        )

    # Keep only SiLU's LOGISTIC (sigmoid factor) as float32 computation.
    # Match by the exported SiLU activation scope, not by operation type alone:
    # the detection head's final class-score LOGISTIC must remain quantized with
    # its own probability-range parameters.
    # no_quantize on SiLU LOGISTIC while neighbours are INT8 produces:
    #   INT8 → DEQUANTIZE → LOGISTIC (float32) → QUANTIZE → INT8
    # VectorBlox handles this with a 256-entry lookup table.
    qt.update_quantization_recipe(
        regex=r"activation\.SiLU_act",
        operation_name=TFLOperationName.LOGISTIC,
        algorithm_key="no_quantize",
    )

    # Keep the final detection-head concatenation in float32 as well. The
    # concatenation combines pixel-scale box coordinates with [0, 1] class
    # probabilities; quantizing the combined tensor with the box-derived scale
    # (roughly 2.5) rounds all confidence scores to zero. Scope-match only the
    # exported model output so intermediate concatenations remain quantized.
    qt.update_quantization_recipe(
        regex=r"serving_default_output_0_output",
        operation_name=TFLOperationName.CONCATENATION,
        algorithm_key="no_quantize",
    )

    # Force the entire CONV→TRANSPOSE→RESHAPE→SLICE chain feeding BATCH_MATMUL
    # to be symmetric (ZP=0), eliminating the QUANTIZE re-quantize ops before
    # each BMM input. The Softmax ADD supplies the independently quantized
    # boundary used directly by BMM2.
    #
    # Match by module path rather than suffix index so the recipe applies to ALL
    # attention blocks regardless of how many PSABlocks the model has:
    #   Conv_qkv  — the QKV projection conv (CONV + TRANSPOSE outputs)
    #   Attention_attn  — the post-reshape/slice attention tensors
    # Both patterns match every C2PSA block in any YOLO variant.
    _sym8w = TensorQuantizationConfig(
        num_bits=8, symmetric=True, granularity=QuantGranularity.CHANNELWISE
    )
    _sym_cfg = OpQuantizationConfig(
        activation_tensor_config=_sym8,
        weight_tensor_config=_sym8w,
        compute_precision=ComputePrecision.INTEGER,
    )
    for op_name, regex in (
        # QKV projection: CONV→TRANSPOSE→RESHAPE→SLICE (BMM1 inputs)
        (TFLOperationName.CONV_2D, r"Conv_qkv.*Conv2d_conv;"),
        (TFLOperationName.TRANSPOSE, r"Conv_qkv.*Conv2d_conv;"),
        (TFLOperationName.RESHAPE, r"Attention_attn;"),
        (TFLOperationName.SLICE, r"Attention_attn;"),
        # BMM1 output chain: BMM1→MUL(scale)→SOFTMAX
        # Sym MUL: no QUANTIZE between BMM1 and MUL (both zp=0)
        # Sym SOFTMAX: SOFTMAX output uses symmetric (zp=0) rather than standard
        #   zp=-128, so MUL→SOFTMAX share the same zp=0 domain (no re-quantize)
        (TFLOperationName.BATCH_MATMUL, r"Attention_attn;"),
        (TFLOperationName.MUL, r"Attention_attn;"),
        (TFLOperationName.SOFTMAX, r"Attention_attn;"),
        # pe path after BMM2: RESHAPE→TRANSPOSE(Conv_pe)→DCONV→TRANSPOSE
        # Sym TRANSPOSE/DCONV: no QUANTIZE between RESHAPE(zp=0) and pe path
        (TFLOperationName.TRANSPOSE, r"Conv_pe.*Conv2d_conv;"),
        (TFLOperationName.DEPTHWISE_CONV_2D, r"Conv_pe.*Conv2d_conv;"),
    ):
        qt.update_quantization_recipe(
            regex=regex,
            operation_name=op_name,
            algorithm_key="min_max_uniform_quantize",
            op_config=_sym_cfg,
        )

    print(f"  [calib] {len(calib_data)} images …")
    result = qt.quantize(qt.calibrate({"serving_default": calib_data}))

    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp:
        tmp_path = tmp.name
    result.export_model(tmp_path, overwrite=True)
    flatbuf = open(tmp_path, "rb").read()
    os.unlink(tmp_path)

    flatbuf = _remove_softmax_add_requantize(flatbuf)
    flatbuf = _rewrite_attention_tail_add_to_nhwc(flatbuf)
    flatbuf = _attention_slices_to_strided_slice(flatbuf)
    flatbuf = _pe_depthwise_to_conv2d(flatbuf)
    open(out_path, "wb").write(flatbuf)
    print(f"  [int8]  {out_path}  ({len(flatbuf) / 1e6:.1f} MB)")
    return out_path


def _remove_softmax_add_requantize(flatbuf: bytes) -> bytes:
    """Route each attention Softmax ADD directly into BMM2.

    AEQ assigns Softmax's fixed (scale=1/256, zero_point=-128) encoding to
    the ADD output, then inserts QUANTIZE to the symmetric zero-point-0
    encoding selected for BMM2. Make the ADD output use the BMM input encoding
    directly, bypass the QUANTIZE tensor, and remove only that QUANTIZE op.
    """
    import copy
    from ai_edge_litert.tools import flatbuffer_utils
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    model = tfl_flatbuffer_utils.read_model(flatbuf)
    add_code = 0  # TFLite BuiltinOperator.ADD
    quantize_code = 114  # TFLite BuiltinOperator.QUANTIZE
    bmm_code = 126  # TFLite BuiltinOperator.BATCH_MATMUL
    rewrites = 0

    for subgraph in model.subgraphs:
        tensors = subgraph.tensors
        operators = subgraph.operators
        builtin_codes = [int(op_code.builtinCode) for op_code in model.operatorCodes]
        quantize_indices = set()
        for add_index, add_op in enumerate(operators):
            if (
                builtin_codes[add_op.opcodeIndex] != add_code
                or len(add_op.outputs) != 1
            ):
                continue
            add_output = int(add_op.outputs[0])
            add_name = tensors[add_output].name.decode(errors="ignore")
            if "Attention_attn;" not in add_name:
                continue

            consumers = [
                (op_index, op)
                for op_index, op in enumerate(operators)
                if add_output in [int(i) for i in op.inputs if int(i) >= 0]
            ]
            if len(consumers) != 1:
                continue
            quant_index, quant_op = consumers[0]
            if (
                builtin_codes[quant_op.opcodeIndex] != quantize_code
                or len(quant_op.outputs) != 1
            ):
                continue
            quant_output = int(quant_op.outputs[0])
            bmm_consumers = [
                (op_index, op)
                for op_index, op in enumerate(operators)
                if quant_output in [int(i) for i in op.inputs if int(i) >= 0]
            ]
            if len(bmm_consumers) != 1:
                continue
            bmm_index, bmm_op = bmm_consumers[0]
            if builtin_codes[bmm_op.opcodeIndex] != bmm_code:
                continue
            bmm_input_tensor = tensors[quant_output]
            add_tensor = tensors[add_output]
            if int(bmm_input_tensor.type) != 9:  # TFLite TensorType.INT8
                raise ValueError(f"Expected INT8 BMM input after {add_name}")

            # The ADD result now directly carries the encoding expected by BMM.
            add_tensor.quantization = copy.deepcopy(bmm_input_tensor.quantization)
            add_tensor.type = bmm_input_tensor.type
            add_tensor.shape = list(bmm_input_tensor.shape)
            bmm_op.inputs = [
                add_output if int(input_index) == quant_output else input_index
                for input_index in bmm_op.inputs
            ]
            quantize_indices.add(quant_index)
            rewrites += 1
        if quantize_indices:
            subgraph.operators = [
                op
                for index, op in enumerate(operators)
                if index not in quantize_indices
            ]

    if rewrites == 0:
        raise ValueError("No attention ADD→QUANTIZE→BATCH_MATMUL chains were rewritten")

    flatbuf = flatbuffer_utils.convert_object_to_bytearray(model)
    print(f"  [graph] removed {rewrites} Softmax ADD→QUANTIZE nodes before BMM")
    return bytes(flatbuf)


def _rewrite_attention_tail_add_to_nhwc(
    flatbuf: bytes, expected_paths: int | None = None
) -> bytes:
    """Match Ultralytics tail order: transpose both branches before ADD.

    BMM2 NHWC branch: BMM2→RESHAPE→TRANSPOSE→ADD.
    PE branch: DWCONV(NHWC)→ADD (remove the transpose back to NCHW).
    Projection: ADD→Conv (remove the transpose after ADD).
    The two branch QUANTIZE ops are bypassed, carrying their encodings onto the
    producer tensors. This matches the reference `yolo11n_full_integer_quant`.
    """
    import copy
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    model = tfl_flatbuffer_utils.read_model(flatbuf)
    codes = [int(code.builtinCode) for code in model.operatorCodes]
    add_code = int(schema.BuiltinOperator.ADD)
    quant_code = int(schema.BuiltinOperator.QUANTIZE)
    reshape_code = int(schema.BuiltinOperator.RESHAPE)
    transpose_code = int(schema.BuiltinOperator.TRANSPOSE)
    conv_code = int(schema.BuiltinOperator.CONV_2D)
    dwconv_code = int(schema.BuiltinOperator.DEPTHWISE_CONV_2D)
    bmm_code = int(schema.BuiltinOperator.BATCH_MATMUL)
    int8_type = int(schema.TensorType.INT8)
    rewrites = 0

    for subgraph in model.subgraphs:
        tensors, operators = subgraph.tensors, subgraph.operators
        producer = {
            int(tid): (idx, op)
            for idx, op in enumerate(operators)
            for tid in (op.outputs or [])
            if int(tid) >= 0
        }
        consumers = {}
        for idx, op in enumerate(operators):
            for tid in op.inputs or []:
                if int(tid) >= 0:
                    consumers.setdefault(int(tid), []).append((idx, op))
        remove_ops = set()
        additions = {}

        for add_index, add in enumerate(operators):
            if (
                codes[add.opcodeIndex] != add_code
                or len(add.inputs or []) != 2
                or not add.outputs
            ):
                continue
            add_out = int(add.outputs[0])
            name = tensors[add_out].name.decode(errors="ignore")
            if "Attention_attn;" not in name or "PSABlock" not in name:
                continue
            post_users = consumers.get(add_out, [])
            if (
                len(post_users) != 1
                or codes[post_users[0][1].opcodeIndex] != transpose_code
            ):
                continue
            post_transpose_index, post_transpose = post_users[0]
            post_out = int(post_transpose.outputs[0])
            project_users = consumers.get(post_out, [])
            if (
                len(project_users) != 1
                or codes[project_users[0][1].opcodeIndex] != conv_code
            ):
                continue
            if len(post_transpose.inputs or []) < 2:
                continue
            permutation = int(post_transpose.inputs[1])

            quantized_branches = []
            for input_id in add.inputs:
                q_output = int(input_id)
                q = producer.get(q_output)
                if (
                    q is None
                    or codes[q[1].opcodeIndex] != quant_code
                    or not q[1].inputs
                ):
                    break
                q_index, q_op = q
                pre_q = int(q_op.inputs[0])
                pre = producer.get(pre_q)
                if pre is None:
                    break
                quantized_branches.append((q_index, q_output, pre_q, pre))
            if len(quantized_branches) != 2:
                continue

            bmm_branch = pe_branch = None
            for branch in quantized_branches:
                _, _, pre_q, (pre_idx, pre_op) = branch
                pre_code = codes[pre_op.opcodeIndex]
                if pre_code == reshape_code and pre_op.inputs:
                    upstream = producer.get(int(pre_op.inputs[0]))
                    if upstream and codes[upstream[1].opcodeIndex] == bmm_code:
                        bmm_branch = branch
                elif pre_code == transpose_code and pre_op.inputs:
                    upstream = producer.get(int(pre_op.inputs[0]))
                    if upstream and codes[upstream[1].opcodeIndex] == dwconv_code:
                        pe_branch = branch
            if bmm_branch is None or pe_branch is None:
                continue

            bmm_q_index, bmm_q_output, bmm_reshape_out, _ = bmm_branch
            pe_q_index, pe_q_output, pe_transposed, (_, pe_transpose) = pe_branch
            pe_nhwc = int(pe_transpose.inputs[0])
            pe_source = producer.get(pe_nhwc)
            if pe_source is None or codes[pe_source[1].opcodeIndex] != dwconv_code:
                continue

            # Add a TRANSPOSE to BMM2 output using the reference NHWC perm.
            nhwc_shape = list(tensors[post_out].shape or [])
            new_transposed = schema.TensorT()
            new_transposed.shape = nhwc_shape
            new_transposed.type = int8_type
            new_transposed.buffer = 0
            new_transposed.name = (
                tensors[bmm_reshape_out].name.decode(errors="ignore")
                + "_nhwc_before_add"
            ).encode()
            new_transposed_index = len(tensors)
            tensors.append(new_transposed)

            new_transpose = schema.OperatorT()
            new_transpose.opcodeIndex = post_transpose.opcodeIndex
            new_transpose.inputs = [bmm_reshape_out, permutation]
            new_transpose.outputs = [new_transposed_index]
            new_transpose.builtinOptionsType = post_transpose.builtinOptionsType
            new_transpose.builtinOptions = copy.deepcopy(post_transpose.builtinOptions)
            additions.setdefault(id(add), []).append(new_transpose)

            # TRANSPOSE is layout-only. Match the reference by preserving the
            # source tensor's scale and zero point across the operation.
            bmm_source_encoding = tensors[bmm_reshape_out]
            output_encoding = tensors[post_out]
            if any(
                int(t.type) != int8_type
                for t in (bmm_source_encoding, tensors[pe_nhwc], output_encoding)
            ):
                raise ValueError("Expected INT8 encodings on attention tail branches")
            new_transposed.quantization = copy.deepcopy(
                bmm_source_encoding.quantization
            )
            new_transposed.type = bmm_source_encoding.type
            source_zp = list(bmm_source_encoding.quantization.zeroPoint)
            transpose_zp = list(new_transposed.quantization.zeroPoint)
            if source_zp != transpose_zp:
                raise ValueError(
                    "Attention-tail TRANSPOSE must preserve BMM2 reshape zero-point"
                )
            add_out_tensor = tensors[add_out]
            add_out_tensor.quantization = copy.deepcopy(output_encoding.quantization)
            add_out_tensor.type = output_encoding.type
            add_out_tensor.shape = nhwc_shape

            add.inputs = [
                new_transposed_index
                if int(t) == bmm_q_output
                else pe_nhwc
                if int(t) == pe_q_output
                else t
                for t in add.inputs
            ]
            project_conv = project_users[0][1]
            project_conv.inputs = [
                add_out if int(t) == post_out else t for t in project_conv.inputs
            ]
            remove_ops.update(
                (
                    id(operators[bmm_q_index]),
                    id(operators[pe_q_index]),
                    id(operators[pe_branch[3][0]]),
                    id(post_transpose),
                )
            )
            rewrites += 1

        if remove_ops or additions:
            rebuilt = []
            for op in operators:
                if id(op) in remove_ops:
                    continue
                rebuilt.extend(additions.get(id(op), []))
                rebuilt.append(op)
            subgraph.operators = rebuilt

    if expected_paths is not None and rewrites != expected_paths:
        raise ValueError(
            f"Expected {expected_paths} attention tail ADD paths; rewrote {rewrites}"
        )
    if rewrites == 0:
        raise ValueError("No two-QUANTIZE attention-tail ADD patterns were rewritten")
    print(
        f"  [graph] rewrote {rewrites} attention tails to TRANSPOSE→ADD→PROJ; removed two input QUANTIZEs and the output TRANSPOSE"
    )
    return bytes(flatbuffer_utils.convert_object_to_bytearray(model))


def _attention_slices_to_strided_slice(flatbuf: bytes) -> bytes:
    """Replace attention Q/K/V SLICE ops with equivalent STRIDED_SLICE ops.

    For these rank-4 channel slices, TFLite SLICE(start,size) maps to
    STRIDED_SLICE(begin=start, end=start+size, strides=1) with begin/end masks
    on batch, head, and sequence axes. Non-attention SLICE ops are untouched.
    """
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    model = tfl_flatbuffer_utils.read_model(flatbuf)
    builtin_codes = [int(code.builtinCode) for code in model.operatorCodes]
    slice_code = int(schema.BuiltinOperator.SLICE)
    strided_slice_code = int(schema.BuiltinOperator.STRIDED_SLICE)
    slice_opcode = next(
        (
            i
            for i, code in enumerate(model.operatorCodes)
            if int(code.builtinCode) == slice_code
        ),
        None,
    )
    if slice_opcode is None:
        return flatbuf
    strided_opcode = next(
        (
            i
            for i, code in enumerate(model.operatorCodes)
            if int(code.builtinCode) == strided_slice_code
        ),
        None,
    )
    if strided_opcode is None:
        opcode = schema.OperatorCodeT()
        opcode.builtinCode = strided_slice_code
        opcode.deprecatedBuiltinCode = min(strided_slice_code, 127)
        opcode.version = 1
        model.operatorCodes.append(opcode)
        strided_opcode = len(model.operatorCodes) - 1

    rewrites = 0
    for subgraph in model.subgraphs:
        tensors, operators = subgraph.tensors, subgraph.operators
        for op in operators:
            if (
                builtin_codes[op.opcodeIndex] != slice_code
                or len(op.inputs or []) != 3
                or not op.outputs
            ):
                continue
            out_index = int(op.outputs[0])
            out_name = tensors[out_index].name.decode(errors="ignore")
            if "Attention_attn;" not in out_name:
                continue

            input_index, begin_index, size_index = (int(i) for i in op.inputs)
            shape = list(tensors[input_index].shape or [])
            if len(shape) != 4:
                raise ValueError(
                    f"Expected rank-4 attention SLICE input, got {shape}: {out_name}"
                )
            begin_buffer = model.buffers[int(tensors[begin_index].buffer)]
            size_buffer = model.buffers[int(tensors[size_index].buffer)]
            if begin_buffer.data is None or size_buffer.data is None:
                raise ValueError(
                    f"Attention SLICE begin/size must be constant: {out_name}"
                )
            begin = np.frombuffer(bytes(begin_buffer.data), dtype=np.int32).copy()
            size = np.frombuffer(bytes(size_buffer.data), dtype=np.int32).copy()
            if begin.size != 4 or size.size != 4:
                raise ValueError(
                    f"Expected four-element attention SLICE parameters: {out_name}"
                )
            end = begin + size
            strides = np.ones(4, dtype=np.int32)

            def add_int32_const(suffix, values):
                buf = schema.BufferT()
                buf.data = values.astype(np.int32, copy=False).view(np.uint8).tolist()
                model.buffers.append(buf)
                tensor = schema.TensorT()
                tensor.shape = [4]
                tensor.type = int(schema.TensorType.INT32)
                tensor.buffer = len(model.buffers) - 1
                tensor.name = (
                    tensors[out_index].name.decode(errors="ignore") + suffix
                ).encode()
                tensor_index = len(tensors)
                tensors.append(tensor)
                return tensor_index

            begin_tensor = add_int32_const("_strided_begin", begin)
            end_tensor = add_int32_const("_strided_end", end)
            stride_tensor = add_int32_const("_strided_stride", strides)
            op.opcodeIndex = strided_opcode
            op.inputs = [input_index, begin_tensor, end_tensor, stride_tensor]
            op.builtinOptionsType = int(schema.BuiltinOptions.StridedSliceOptions)
            options = schema.StridedSliceOptionsT()
            options.beginMask = 0b1011
            options.endMask = 0b1011
            options.ellipsisMask = 0
            options.newAxisMask = 0
            options.shrinkAxisMask = 0
            op.builtinOptions = options
            rewrites += 1

    if rewrites == 0:
        raise ValueError("No attention SLICE ops found to convert to STRIDED_SLICE")
    print(
        f"  [graph] converted {rewrites} attention SLICE ops to equivalent STRIDED_SLICE"
    )
    return bytes(flatbuffer_utils.convert_object_to_bytearray(model))


def _pe_depthwise_to_conv2d(flatbuf: bytes, expected_ops: int | None = None) -> bytes:
    """Convert Conv_pe depthwise kernels to equivalent regular Conv2D kernels.

    Depthwise filters are [1, KH, KW, C]. The equivalent grouped-by-output
    regular Conv2D filter layout with one input channel per output is
    [C, KH, KW, 1], i.e. transpose axes (3,1,2,0). Preserve per-channel
    quantization scales while moving the quantized dimension from 3 to 0.
    """
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    model = tfl_flatbuffer_utils.read_model(flatbuf)
    codes = [int(code.builtinCode) for code in model.operatorCodes]
    dw_code = int(schema.BuiltinOperator.DEPTHWISE_CONV_2D)
    conv_code = int(schema.BuiltinOperator.CONV_2D)
    int8_type = int(schema.TensorType.INT8)
    conv_opcode = next(
        (
            i
            for i, code in enumerate(model.operatorCodes)
            if int(code.builtinCode) == conv_code
        ),
        None,
    )
    if conv_opcode is None:
        raise ValueError("Model has no CONV_2D opcode")
    converted = 0

    for subgraph in model.subgraphs:
        tensors = subgraph.tensors
        for op in subgraph.operators:
            if codes[op.opcodeIndex] != dw_code or len(op.inputs or []) < 3:
                continue
            filter_index = int(op.inputs[1])
            filter_tensor = tensors[filter_index]
            name = filter_tensor.name.decode(errors="ignore")
            if "Conv_pe" not in name:
                continue
            shape = list(filter_tensor.shape or [])
            if len(shape) != 4 or shape[0] != 1 or shape[-1] <= 1:
                raise ValueError(
                    f"Unexpected Conv_pe depthwise filter shape {shape}: {name}"
                )
            if int(filter_tensor.type) != int8_type:
                raise ValueError(f"Expected INT8 Conv_pe kernel: {name}")
            buffer_index = int(filter_tensor.buffer)
            raw = model.buffers[buffer_index].data
            raw_bytes = bytes(raw) if raw is not None else b""
            expected_size = int(np.prod(shape))
            if len(raw_bytes) != expected_size:
                raise ValueError(f"Conv_pe filter buffer size mismatch: {name}")
            weights = np.frombuffer(raw_bytes, dtype=np.int8).reshape(shape)
            conv_weights = np.ascontiguousarray(weights.transpose(3, 1, 2, 0))
            model.buffers[buffer_index].data = (
                conv_weights.view(np.uint8).reshape(-1).tolist()
            )
            filter_tensor.shape = list(conv_weights.shape)

            q = filter_tensor.quantization
            if q is not None and q.scale is not None and len(q.scale) > 1:
                q.quantizedDimension = 0

            old_options = op.builtinOptions
            if old_options is None:
                raise ValueError(f"Conv_pe depthwise op has no options: {name}")
            conv_options = schema.Conv2DOptionsT()
            conv_options.padding = old_options.padding
            conv_options.strideW = old_options.strideW
            conv_options.strideH = old_options.strideH
            conv_options.fusedActivationFunction = old_options.fusedActivationFunction
            conv_options.dilationWFactor = old_options.dilationWFactor
            conv_options.dilationHFactor = old_options.dilationHFactor
            # Match the reference regular Conv2D options exactly.
            conv_options.quantizedBiasType = 0
            op.opcodeIndex = conv_opcode
            op.builtinOptionsType = int(schema.BuiltinOptions.Conv2DOptions)
            op.builtinOptions = conv_options
            converted += 1

    if expected_ops is not None and converted != expected_ops:
        raise ValueError(
            f"Expected {expected_ops} Conv_pe depthwise ops, converted {converted}"
        )
    if converted == 0:
        raise ValueError("No Conv_pe DEPTHWISE_CONV_2D ops were converted")
    print(
        f"  [graph] converted {converted} Conv_pe depthwise ops to CONV_2D [C,KH,KW,1]"
    )
    return bytes(flatbuffer_utils.convert_object_to_bytearray(model))


def make_calib_data(
    dataset: str | None,
    n: int,
    imgsz: int,
    offset: int = 0,
) -> list[dict]:
    """Build aeq calibration list from a flat image folder (COCO) or random tensors.

    Supports both flat directories (COCO-style) and ImageFolder layouts.
    YOLO models expect float32 NCHW in [0, 1] — no ImageNet normalization.
    """
    if dataset and os.path.isdir(os.path.expanduser(dataset)):
        import pathlib
        import torchvision.transforms as T
        from PIL.Image import open as pil_open

        tf = T.Compose(
            [
                T.Resize(imgsz),
                T.CenterCrop(imgsz),
                T.ToTensor(),  # → [0,1] float32, no normalization (YOLO convention)
            ]
        )
        root = pathlib.Path(os.path.expanduser(dataset))
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        # Collect images: flat dir or recurse one level (ImageFolder)
        paths = sorted(p for p in root.iterdir() if p.suffix.lower() in exts)
        if not paths:  # ImageFolder layout — gather from subdirs
            paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)

        paths = paths[offset : offset + n]
        data = []
        for p in paths:
            img = tf(pil_open(str(p)).convert("RGB")).unsqueeze(0).numpy()
            data.append({"args_0": img.astype(np.float32)})
        print(f"  [data]  {len(data)} images from {dataset}")
        return data
    else:
        print(f"  [data]  {n} random images (no dataset)")
        return [
            {"args_0": np.random.uniform(0, 1, (1, 3, imgsz, imgsz)).astype(np.float32)}
            for _ in range(n)
        ]



COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
COCO128_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "coco128")


def ensure_coco128(dataset: str | None) -> str:
    """Return a calibration image folder, downloading COCO128 if it is missing.

    Default calibration set is COCO128 train2017 (128 images). If --dataset is
    given and exists, that folder is used unchanged.
    """
    if dataset:
        return os.path.expanduser(dataset)
    img_dir = os.path.join(COCO128_ROOT, "images", "train2017")
    if os.path.isdir(img_dir) and any(os.scandir(img_dir)):
        return img_dir
    import urllib.request, zipfile, tempfile
    os.makedirs(os.path.dirname(COCO128_ROOT), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "coco128.zip")
        print(f"  [data]  downloading COCO128 from {COCO128_URL}")
        urllib.request.urlretrieve(COCO128_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(os.path.dirname(COCO128_ROOT))
    if not os.path.isdir(img_dir):
        raise RuntimeError(f"COCO128 extracted but {img_dir} is missing")
    return img_dir



def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, choices=["yolo11n", "yolo26n"], help="YOLO variant to export")
    ap.add_argument("--dataset", default=None, help="COCO images folder (default: auto-download COCO128)")
    ap.add_argument("--calib", type=int, default=128)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--out", default=None, help="INT8 output (default: <model>_aeq_softmax_add_int8.tflite)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    out_path = a.out or os.path.join(a.outdir, f"{a.model}_aeq_softmax_add_int8.tflite")
    fp32_path = os.path.join(a.outdir, f"{a.model}_fp32_softmax_add.tflite")
    export_fp32(a.model, fp32_path, imgsz=a.imgsz)
    calib = make_calib_data(ensure_coco128(a.dataset), a.calib, a.imgsz)
    quantize(fp32_path, out_path, calib)


if __name__ == "__main__":
    main()
