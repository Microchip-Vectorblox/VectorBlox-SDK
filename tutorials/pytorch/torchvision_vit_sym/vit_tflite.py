"""vit_tflite.py — torchvision ViT-B/16 aeq INT8 TFLite (zp=0 BATCH_MATMUL).

Reproduces vit_b_16_aeq_softmax_add_attention_fc_add_bmm2_output_no_quantize_int8.tflite:
  torchvision vit_b_16 -> _CleanMHA (einsum 4D BMM) -> fp32 TFLite
  -> ai_edge_quantizer (symmetric ZP=0 attention chain, LN kept fp32)
  -> Softmax ADD, Q/K/V FC ADD (ZP=0), BMM2 output QUANTIZE bypass.

Usage:
    python vit_tflite.py --calib 512          # auto-downloads Imagenette 160px (default)
Requires: uv sync --group cpu
"""

import argparse, os, tempfile
import torch
import torch.nn as nn
import torch.nn.functional as _F
import torchvision.models as M
import torchvision.transforms as T
from torchao.quantization.pt2e.utils import _fuse_conv_bn_
import litert_torch

_SZ = (1, 3, 224, 224)


class _CleanMHA(nn.Module):
    """Replaces nn.MultiheadAttention with separate Q/K/V projections.

    Uses 3 separate Linear layers and torch.einsum to produce 4D BATCH_MATMUL
    matching anomaly_transformer.tflite exactly:
      FC(Q/K/V) → RESHAPE [B,S,H,D]
      einsum "blhe,bshe->bhls"  → TRANSPOSE×2 + BATCH_MATMUL [B,H,S,S]
      SOFTMAX
      einsum "bhls,bshd->blhd" → BATCH_MATMUL [B,H,D,S] + TRANSPOSE [B,S,H,D]
      RESHAPE [B,S,E]
    Eliminates the SLICE×3 from the combined-QKV split and the B*H 3D collapse.
    """

    def __init__(
        self,
        mha: nn.MultiheadAttention,
        softmax_add: bool = False,
    ):
        super().__init__()
        E, H = mha.embed_dim, mha.num_heads
        self.num_heads, self.head_dim, self.scale, self.embed_dim = (
            H,
            E // H,
            (E // H) ** -0.5,
            E,
        )
        self.softmax_add = softmax_add
        if softmax_add:
            self.register_buffer("_softmax_add_const", torch.full((1,), 1e-6))
        has_bias = mha.in_proj_bias is not None
        self.q_proj = nn.Linear(E, E, bias=has_bias)
        self.k_proj = nn.Linear(E, E, bias=has_bias)
        self.v_proj = nn.Linear(E, E, bias=has_bias)
        self.out_proj = nn.Linear(E, E, bias=mha.out_proj.bias is not None)
        with torch.no_grad():
            # in_proj_weight is [3E, E]: first E rows = Q, next E = K, last E = V
            self.q_proj.weight.copy_(mha.in_proj_weight[:E])
            self.k_proj.weight.copy_(mha.in_proj_weight[E : 2 * E])
            self.v_proj.weight.copy_(mha.in_proj_weight[2 * E :])
            if has_bias:
                self.q_proj.bias.copy_(mha.in_proj_bias[:E])
                self.k_proj.bias.copy_(mha.in_proj_bias[E : 2 * E])
                self.v_proj.bias.copy_(mha.in_proj_bias[2 * E :])
            self.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                self.out_proj.bias.copy_(mha.out_proj.bias)

    def forward(self, query, key, value, **kwargs):
        B, S, E = query.shape
        H, D = self.num_heads, self.head_dim
        q, k, v = self.q_proj(query), self.k_proj(key), self.v_proj(value)

        # Match anomaly_transformer.tflite op pattern exactly using einsum.
        # torch.einsum lowers to 4D BATCH_MATMUL in TFLite (no 3D collapse):
        #   FC(Q/K/V) → RESHAPE [B,S,H,D]
        #   einsum "blhe,bshe->bhls"  → TRANSPOSE×2 + BATCH_MATMUL [B,H,S,S]
        #   SOFTMAX
        #   einsum "bhls,bshd->blhd" → BATCH_MATMUL [B,H,D,S] + TRANSPOSE [B,S,H,D]
        #   RESHAPE [B,S,E]
        # This eliminates SLICE×3 and the 3D B*H merge, matching anomaly exactly.
        q = q.reshape(B, S, H, D)
        k = k.reshape(B, S, H, D)
        v = v.reshape(B, S, H, D)
        scores = torch.einsum("blhe,bshe->bhls", q, k)  # [B,H,S,S]
        attn = _F.softmax(scores * self.scale, dim=-1)
        if self.softmax_add:
            # Tiny nonzero offset keeps an explicit boundary in the exported
            # graph; AEQ calibrates the ADD output independently from SOFTMAX.
            attn = attn + self._softmax_add_const.to(
                device=attn.device, dtype=attn.dtype
            )
        out = torch.einsum("bhls,bshd->blhd", attn, v)  # [B,S,H,D]
        out = out.reshape(B, S, E)
        return self.out_proj(out), None


def _replace_mha_with_clean(
    model: nn.Module, *, softmax_add: bool = False
) -> nn.Module:
    for name, mod in list(model.named_children()):
        if isinstance(mod, nn.MultiheadAttention):
            setattr(model, name, _CleanMHA(mod, softmax_add=softmax_add))
        else:
            _replace_mha_with_clean(mod, softmax_add=softmax_add)
    return model


_AEQ_RECIPE = [
    {
        "regex": ".*",
        "operation": "*",
        "algorithm_key": "min_max_uniform_quantize",
        "op_config": {
            "activation_tensor_config": {
                "num_bits": 8,
                "symmetric": False,
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


_LN_SEQ = (
    "MEAN",
    "RESHAPE",
    "SQUARED_DIFFERENCE",
    "MEAN",
    "ADD",
    "RSQRT",
    "MUL",
    "SUB",
    "RESHAPE",
    "MUL",
    "RESHAPE",
    "ADD",
)


def find_ln_tensor_names(path: str) -> list[str]:
    """Return output tensor names of LayerNorm op sequences for aeq exemption."""
    from ai_edge_litert.interpreter import Interpreter

    interp = Interpreter(path)
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    tensors = {t["index"]: t for t in interp.get_tensor_details()}
    types = [op["op_name"] for op in ops]
    n = len(_LN_SEQ)
    names = []
    for i in range(len(types) - n + 1):
        if types[i : i + n] == list(_LN_SEQ):
            for op in ops[i : i + n]:
                for idx in op["outputs"]:
                    if idx >= 0 and idx in tensors:
                        names.append(tensors[idx]["name"])
    return names


def quantize_aeq_from_fp32(
    fp32_path: str,
    out_path: str,
    calib_data: list[dict],
    *,
    softmax_add: bool = False,
    attention_fc_add: bool = False,
    bypass_bmm2_output_quantize: bool = False,
) -> str:
    """Quantize a fp32 TFLite with aeq — mirrors the anomaly_quantize.py approach.

    Recipe (_AEQ_RECIPE base):
      ALL ops       → INT8 affine (symmetric=False, ZP≠0)
      BATCH_MATMUL  → INT8 symmetric (ZP=0) — required for attention correctness
      _CleanMHA chain → symmetric throughout (eliminates redundant QUANTIZE ops)
      LN sequence   → no_quantize (float32 wrapped in DEQUANTIZE/QUANTIZE)
      INPUT / OUTPUT → quantized by aeq (calibrated int8 from calib data)

    Produces a clean structure matching anomaly_transformer.tflite:
      FC/Conv/BMM: int8  |  LN: float32  |  LN affine extra: int8
    """
    import tempfile as _tmp
    from ai_edge_quantizer import quantizer as aeq_mod, algorithm_manager
    from ai_edge_quantizer.qtyping import (
        TFLOperationName,
        OpQuantizationConfig,
        TensorQuantizationConfig,
        ComputePrecision,
        QuantGranularity,
    )

    quantizer_path = fp32_path
    injected_fp32_path = None
    if attention_fc_add:
        injected = _insert_attention_fc_add(open(fp32_path, "rb").read())
        with _tmp.NamedTemporaryFile(suffix=".tflite", delete=False) as temp:
            injected_fp32_path = temp.name
            temp.write(injected)
        quantizer_path = injected_fp32_path

    qt = aeq_mod.Quantizer(quantizer_path)
    qt.load_quantization_recipe(_AEQ_RECIPE)

    # BATCH_MATMUL: symmetric ZP=0 required for INT8 attention correctness
    _sym8 = TensorQuantizationConfig(
        num_bits=8, symmetric=True, granularity=QuantGranularity.TENSORWISE
    )
    _sym8w = TensorQuantizationConfig(
        num_bits=8, symmetric=True, granularity=QuantGranularity.CHANNELWISE
    )
    _sym_cfg = OpQuantizationConfig(
        activation_tensor_config=_sym8,
        weight_tensor_config=_sym8w,
        compute_precision=ComputePrecision.INTEGER,
    )
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=TFLOperationName.BATCH_MATMUL,
        algorithm_key="min_max_uniform_quantize",
        op_config=_sym_cfg,
    )

    # Eliminate redundant QUANTIZE ops inside the _CleanMHA attention block by
    # forcing all ops in the chain to symmetric (ZP=0). Mirrors the same recipe
    # applied in yolo_quantize.py for the YOLO C2PSA attention blocks.
    #
    # Tensor name pattern: all VIT attention tensors live under _CleanMHA_self_attention;
    # This covers the full path:
    #   FC(Q/K/V) → RESHAPE → TRANSPOSE → BMM1 → MUL(scale) → SOFTMAX
    #   → TRANSPOSE(V) → BMM2 → TRANSPOSE(out) → RESHAPE
    # The SOFTMAX sym config keeps the attention chain symmetric; an optional
    # Softmax ADD experiment can replace the resulting boundary before BMM2.
    _ATTN_REGEX = r"_CleanMHA_self_attention;"
    for op_name in (
        TFLOperationName.FULLY_CONNECTED,
        TFLOperationName.RESHAPE,
        TFLOperationName.TRANSPOSE,
        TFLOperationName.BATCH_MATMUL,
        TFLOperationName.MUL,
        TFLOperationName.SOFTMAX,
    ):
        qt.update_quantization_recipe(
            regex=_ATTN_REGEX,
            operation_name=op_name,
            algorithm_key="min_max_uniform_quantize",
            op_config=_sym_cfg,
        )

    # LN variance core → float32 (like anomaly_quantize.py)
    ln_names = find_ln_tensor_names(fp32_path)
    if ln_names:
        # Negative lookahead prevents "ln;1" from matching "ln;12", "ln;13" etc.
        ln_regex = "|".join(n + r"(?!\d)" for n in ln_names)
        n_blocks = len(ln_names) // len(_LN_SEQ)
        print(f"  [norm]  {n_blocks} LN blocks ({len(ln_names)} tensors) → fp32")
        for op in set(_LN_SEQ):
            try:
                qt.update_quantization_recipe(
                    ln_regex,
                    TFLOperationName[op],
                    op_config=None,
                    algorithm_key=algorithm_manager.AlgorithmName.NO_QUANTIZE,
                )
            except KeyError:
                pass

    print(f"  [aeq]   calibrating {len(calib_data)} images …")
    result = qt.quantize(qt.calibrate({"serving_default": calib_data}))

    with _tmp.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp:
        tmp_path = tmp.name
    result.export_model(tmp_path, overwrite=True)
    flatbuf = open(tmp_path, "rb").read()
    os.unlink(tmp_path)
    if injected_fp32_path:
        os.unlink(injected_fp32_path)
    if softmax_add:
        flatbuf = _replace_softmax_quantize_with_add(flatbuf)
    if attention_fc_add:
        flatbuf = _replace_attention_fc_quantize_with_add(flatbuf)
    if bypass_bmm2_output_quantize:
        flatbuf = _remove_bmm2_output_quantize(flatbuf)
    with open(out_path, "wb") as f:
        f.write(flatbuf)
    print(f"  [aeq]   {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")
    return out_path


def _insert_attention_fc_add(flatbuf: bytes, expected_paths: int = 36) -> bytes:
    """Insert a tiny ADD directly after each attention Q/K/V projection FC."""
    import struct
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    model = tfl_flatbuffer_utils.read_model(flatbuf)
    add_code = int(schema.BuiltinOperator.ADD)
    fc_code = int(schema.BuiltinOperator.FULLY_CONNECTED)
    opcode_index = next(
        (
            i
            for i, code in enumerate(model.operatorCodes)
            if int(code.builtinCode) == add_code
        ),
        None,
    )
    if opcode_index is None:
        opcode = schema.OperatorCodeT()
        opcode.builtinCode = add_code
        opcode.deprecatedBuiltinCode = min(add_code, 127)
        opcode.version = 1
        model.operatorCodes.append(opcode)
        opcode_index = len(model.operatorCodes) - 1

    scalar_buffer = schema.BufferT()
    scalar_buffer.data = list(struct.pack("<f", 1e-6))
    model.buffers.append(scalar_buffer)
    scalar_buffer_index = len(model.buffers) - 1
    rewrites = 0

    for subgraph in model.subgraphs:
        tensors = subgraph.tensors
        operators = subgraph.operators
        consumers = {}
        for index, op in enumerate(operators):
            for tid in op.inputs or []:
                if int(tid) >= 0:
                    consumers.setdefault(int(tid), []).append((index, op))
        inserts = []

        for fc_index, fc_op in enumerate(operators):
            if int(model.operatorCodes[fc_op.opcodeIndex].builtinCode) != fc_code:
                continue
            if not fc_op.outputs:
                continue
            fc_output = int(fc_op.outputs[0])
            name = tensors[fc_output].name.decode(errors="ignore")
            if "_CleanMHA_self_attention" not in name or not any(
                proj in name
                for proj in ("Linear_q_proj", "Linear_k_proj", "Linear_v_proj")
            ):
                continue
            users = consumers.get(fc_output, [])
            if len(users) != 1:
                continue

            # FC output tensor remains as the first ADD operand. Route its
            # original consumer (normally QUANTIZE or RESHAPE) through ADD.
            add_output_tensor = schema.TensorT()
            add_output_tensor.shape = list(tensors[fc_output].shape)
            add_output_tensor.type = int(schema.TensorType.FLOAT32)
            add_output_tensor.buffer = 0
            add_output_tensor.name = (name + "_attention_fc_add").encode()
            add_output_index = len(tensors)
            tensors.append(add_output_tensor)

            scalar_tensor = schema.TensorT()
            scalar_tensor.shape = [1]
            scalar_tensor.type = int(schema.TensorType.FLOAT32)
            scalar_tensor.buffer = scalar_buffer_index
            scalar_tensor.name = (name + "_attention_fc_add_const").encode()
            scalar_tensor_index = len(tensors)
            tensors.append(scalar_tensor)

            add_op = schema.OperatorT()
            add_op.opcodeIndex = opcode_index
            add_op.inputs = [fc_output, scalar_tensor_index]
            add_op.outputs = [add_output_index]
            add_op.builtinOptionsType = int(schema.BuiltinOptions.AddOptions)
            add_op.builtinOptions = schema.AddOptionsT()

            user_index, user_op = users[0]
            user_op.inputs = [
                add_output_index if int(tid) == fc_output else tid
                for tid in user_op.inputs
            ]
            inserts.append((user_index, add_op))
            rewrites += 1

        for index, op in reversed(inserts):
            operators.insert(index, op)

    if rewrites != expected_paths:
        raise ValueError(
            f"Expected {expected_paths} attention FC outputs for ADD insertion; inserted {rewrites}"
        )
    print(f"  [graph] inserted {rewrites} ADD nodes after attention FC")
    return bytes(flatbuffer_utils.convert_object_to_bytearray(model))


def _replace_attention_fc_quantize_with_add(
    flatbuf: bytes, expected_paths: int = 36
) -> bytes:
    """Bypass QUANTIZE immediately after injected attention FC ADDs."""
    import copy
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    model = tfl_flatbuffer_utils.read_model(flatbuf)
    codes = [int(code.builtinCode) for code in model.operatorCodes]
    add_code = int(schema.BuiltinOperator.ADD)
    fc_code = int(schema.BuiltinOperator.FULLY_CONNECTED)
    quantize_code = int(schema.BuiltinOperator.QUANTIZE)
    dequantize_code = int(schema.BuiltinOperator.DEQUANTIZE)
    int8_type = int(schema.TensorType.INT8)
    rewrites = 0
    for subgraph in model.subgraphs:
        tensors, operators = subgraph.tensors, subgraph.operators
        producer = {
            int(tid): (index, op)
            for index, op in enumerate(operators)
            for tid in (op.outputs or [])
            if int(tid) >= 0
        }
        consumers = {}
        for index, op in enumerate(operators):
            for tid in op.inputs or []:
                if int(tid) >= 0:
                    consumers.setdefault(int(tid), []).append((index, op))
        remove = set()
        for add_index, add_op in enumerate(operators):
            if codes[add_op.opcodeIndex] != add_code or len(add_op.outputs or []) != 1:
                continue
            add_output = int(add_op.outputs[0])
            add_name = tensors[add_output].name.decode(errors="ignore")
            if (
                "_CleanMHA_self_attention" not in add_name
                or "attention_fc_add" not in add_name
            ):
                continue
            input_ids = [int(tid) for tid in (add_op.inputs or []) if int(tid) >= 0]
            dequant_inputs = [
                tid
                for tid in input_ids
                if (src := producer.get(tid)) is not None
                and codes[src[1].opcodeIndex] == dequantize_code
            ]
            constants = [tid for tid in input_ids if tid not in dequant_inputs]
            if len(dequant_inputs) != 1 or len(constants) != 1:
                continue
            dequant_output = dequant_inputs[0]
            dequant_index, dequant_op = producer[dequant_output]
            if not dequant_op.inputs:
                continue
            fc_output = int(dequant_op.inputs[0])
            fc_source = producer.get(fc_output)
            if (
                fc_source is None
                or codes[fc_source[1].opcodeIndex] != fc_code
                or "_CleanMHA_self_attention"
                not in tensors[fc_output].name.decode(errors="ignore")
            ):
                continue
            add_users = consumers.get(add_output, [])
            if len(add_users) != 1:
                continue
            quant_index, quant_op = add_users[0]
            if (
                codes[quant_op.opcodeIndex] != quantize_code
                or len(quant_op.outputs or []) != 1
            ):
                continue
            quant_output = int(quant_op.outputs[0])
            if len(consumers.get(quant_output, [])) != 1:
                continue
            expected = tensors[quant_output]
            if int(expected.type) != int8_type:
                raise ValueError("Expected INT8 encoding after attention FC ADD")
            add_tensor = tensors[add_output]
            add_tensor.quantization = copy.deepcopy(expected.quantization)
            add_tensor.type = expected.type
            add_tensor.shape = list(expected.shape)
            add_tensor.quantization.zeroPoint = [0]
            # TFLite ADD requires identical operand types. The injected FP32
            # epsilon is an identity boundary; encode it as INT8 zero using
            # the same ZP=0 affine domain as the ADD output.
            epsilon_tensor = tensors[constants[0]]
            if int(epsilon_tensor.type) == int(schema.TensorType.FLOAT32):
                epsilon_tensor.type = int8_type
                epsilon_tensor.shape = [1]
                epsilon_tensor.quantization = copy.deepcopy(expected.quantization)
                epsilon_tensor.quantization.zeroPoint = [0]
                epsilon_buffer = schema.BufferT()
                epsilon_buffer.data = [0]
                model.buffers.append(epsilon_buffer)
                epsilon_tensor.buffer = len(model.buffers) - 1
            user = consumers[quant_output][0][1]
            user.inputs = [
                add_output if int(tid) == quant_output else tid for tid in user.inputs
            ]
            # Bypass both AEQ boundary conversions around ADD: it now consumes
            # the FC's INT8 output and the scalar epsilon, and carries ZP=0.
            add_op.inputs = [
                fc_output if int(tid) == dequant_output else tid
                for tid in add_op.inputs
            ]
            # Preserve the ADD's calibrated scale but force symmetric encoding
            # through layout-only reshapes/transposes until the consuming BMM.
            frontier = [add_output]
            visited = set()
            while frontier:
                tensor_id = frontier.pop()
                if tensor_id in visited:
                    continue
                visited.add(tensor_id)
                for _, consumer in consumers.get(tensor_id, []):
                    code = codes[consumer.opcodeIndex]
                    if code not in (
                        int(schema.BuiltinOperator.RESHAPE),
                        int(schema.BuiltinOperator.TRANSPOSE),
                    ):
                        continue
                    for output_id in consumer.outputs or []:
                        output_id = int(output_id)
                        out_tensor = tensors[output_id]
                        if int(out_tensor.type) != int8_type:
                            continue
                        out_tensor.quantization.scale = list(
                            add_tensor.quantization.scale
                        )
                        out_tensor.quantization.zeroPoint = [0]
                        frontier.append(output_id)
            remove.update((quant_index, dequant_index))
            rewrites += 1
        if remove:
            subgraph.operators = [
                op for i, op in enumerate(operators) if i not in remove
            ]
    if rewrites != expected_paths:
        raise ValueError(
            f"Expected {expected_paths} FC ADD→QUANTIZE boundaries; replaced {rewrites}"
        )
    print(f"  [graph] replaced {rewrites} attention FC QUANTIZE nodes with ADD")
    return bytes(flatbuffer_utils.convert_object_to_bytearray(model))


def _replace_softmax_quantize_with_add(flatbuf: bytes) -> bytes:
    """Route each ViT attention ADD directly into BMM2, removing its QUANTIZE.

    Only rewrites the exact attention Softmax → ADD → QUANTIZE → BMM2 chain;
    classifier and other ADD/QUANTIZE paths are untouched.
    """
    import copy
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    model = tfl_flatbuffer_utils.read_model(flatbuf)
    codes = [int(code.builtinCode) for code in model.operatorCodes]
    softmax_code = int(schema.BuiltinOperator.SOFTMAX)
    quantize_code = int(schema.BuiltinOperator.QUANTIZE)
    bmm_code = int(schema.BuiltinOperator.BATCH_MATMUL)
    add_code = int(schema.BuiltinOperator.ADD)
    rewrites = 0

    for subgraph in model.subgraphs:
        tensors = subgraph.tensors
        operators = subgraph.operators
        producer = {
            int(output): op
            for op in operators
            for output in (op.outputs or [])
            if int(output) >= 0
        }
        consumers = {}
        for index, op in enumerate(operators):
            for input_index in op.inputs or []:
                if int(input_index) >= 0:
                    consumers.setdefault(int(input_index), []).append((index, op))
        remove_indices = set()

        def comes_from_softmax(tensor_index):
            seen = set()
            current = int(tensor_index)
            while current not in seen:
                seen.add(current)
                op = producer.get(current)
                if op is None:
                    return False
                code = codes[op.opcodeIndex]
                if code == softmax_code:
                    return len(tensors[current].shape or []) >= 3
                if code not in (quantize_code, add_code) or not op.inputs:
                    return False
                current = int(op.inputs[0])
            return False

        for bmm_index, bmm_op in enumerate(operators):
            if codes[bmm_op.opcodeIndex] != bmm_code or len(bmm_op.inputs or []) != 2:
                continue
            attention_position = next(
                (i for i, tid in enumerate(bmm_op.inputs) if comes_from_softmax(tid)),
                None,
            )
            if attention_position is None:
                continue
            quant_output = int(bmm_op.inputs[attention_position])
            q_op = producer.get(quant_output)
            if (
                q_op is None
                or codes[q_op.opcodeIndex] != quantize_code
                or len(q_op.inputs or []) != 1
            ):
                continue
            add_output = int(q_op.inputs[0])
            add_op = producer.get(add_output)
            if add_op is None or codes[add_op.opcodeIndex] != add_code:
                continue
            if not any(
                (sm := producer.get(int(i))) is not None
                and codes[sm.opcodeIndex] == softmax_code
                and len(tensors[int(i)].shape or []) >= 3
                for i in (add_op.inputs or [])
            ):
                continue
            if consumers.get(quant_output, []) != [(bmm_index, bmm_op)]:
                # A tensor shared with another consumer cannot be removed safely.
                continue
            q_index = next(i for i, op in enumerate(operators) if op is q_op)
            expected = tensors[quant_output]
            if int(expected.type) != int(schema.TensorType.INT8):
                raise ValueError("BMM2 attention input after Softmax ADD must be INT8")
            add_tensor = tensors[add_output]
            add_tensor.quantization = copy.deepcopy(expected.quantization)
            add_tensor.type = expected.type
            add_tensor.shape = list(expected.shape)
            bmm_op.inputs[attention_position] = add_output
            remove_indices.add(q_index)
            rewrites += 1

        if remove_indices:
            subgraph.operators = [
                op for index, op in enumerate(operators) if index not in remove_indices
            ]

    if rewrites == 0:
        raise ValueError(
            "No ViT attention SOFTMAX→ADD→QUANTIZE→BMM chains were rewritten"
        )
    rewritten = flatbuffer_utils.convert_object_to_bytearray(model)
    print(
        f"  [graph] replaced {rewrites} attention Softmax→QUANTIZE boundaries with ADD→BMM"
    )
    return bytes(rewritten)


def _remove_bmm2_output_quantize(flatbuf: bytes, expected_paths: int = 12) -> bytes:
    """Bypass QUANTIZE in each attention BMM2→TRANSPOSE→QUANTIZE→RESHAPE path.

    The transpose output adopts the downstream quantized tensor's encoding;
    the following RESHAPE then consumes the transpose output directly. The
    pass is deliberately strict about operator order, single-use tensors, and
    attention BMM identification so unrelated quantizers are untouched.
    """
    import copy
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.tools import flatbuffer_utils
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils

    model = tfl_flatbuffer_utils.read_model(flatbuf)
    codes = [int(code.builtinCode) for code in model.operatorCodes]
    softmax_code = int(schema.BuiltinOperator.SOFTMAX)
    add_code = int(schema.BuiltinOperator.ADD)
    quantize_code = int(schema.BuiltinOperator.QUANTIZE)
    bmm_code = int(schema.BuiltinOperator.BATCH_MATMUL)
    transpose_code = int(schema.BuiltinOperator.TRANSPOSE)
    reshape_code = int(schema.BuiltinOperator.RESHAPE)
    int8_type = int(schema.TensorType.INT8)
    rewrites = 0

    for subgraph in model.subgraphs:
        tensors = subgraph.tensors
        operators = subgraph.operators
        producer = {
            int(output): (index, op)
            for index, op in enumerate(operators)
            for output in (op.outputs or [])
            if int(output) >= 0
        }
        consumers = {}
        for index, op in enumerate(operators):
            for input_index in op.inputs or []:
                if int(input_index) >= 0:
                    consumers.setdefault(int(input_index), []).append((index, op))
        remove_indices = set()

        def comes_from_softmax(tensor_index):
            seen = set()
            current = int(tensor_index)
            while current not in seen:
                seen.add(current)
                source = producer.get(current)
                if source is None:
                    return False
                op = source[1]
                code = codes[op.opcodeIndex]
                if code == softmax_code:
                    return len(tensors[current].shape or []) >= 3
                if code not in (quantize_code, add_code) or not op.inputs:
                    return False
                current = int(op.inputs[0])
            return False

        for bmm_op in operators:
            if codes[bmm_op.opcodeIndex] != bmm_code or len(bmm_op.inputs or []) != 2:
                continue
            if not any(
                comes_from_softmax(tensor_index) for tensor_index in bmm_op.inputs
            ):
                continue
            bmm_output = int(bmm_op.outputs[0]) if bmm_op.outputs else -1
            transpose_users = consumers.get(bmm_output, [])
            if len(transpose_users) != 1:
                continue
            transpose_index, transpose_op = transpose_users[0]
            if (
                codes[transpose_op.opcodeIndex] != transpose_code
                or len(transpose_op.outputs or []) != 1
            ):
                continue
            transpose_output = int(transpose_op.outputs[0])
            quant_users = consumers.get(transpose_output, [])
            if len(quant_users) != 1:
                continue
            quant_index, quant_op = quant_users[0]
            if (
                codes[quant_op.opcodeIndex] != quantize_code
                or len(quant_op.inputs or []) != 1
                or len(quant_op.outputs or []) != 1
                or int(quant_op.inputs[0]) != transpose_output
            ):
                continue
            quant_output = int(quant_op.outputs[0])
            reshape_users = consumers.get(quant_output, [])
            if (
                len(reshape_users) != 1
                or codes[reshape_users[0][1].opcodeIndex] != reshape_code
            ):
                continue

            quant_tensor = tensors[quant_output]
            if int(quant_tensor.type) != int8_type:
                raise ValueError(
                    "Expected INT8 QUANTIZE output after attention BMM2 TRANSPOSE"
                )
            transpose_tensor = tensors[transpose_output]
            # With QUANTIZE removed, the value produced by BMM2 must already
            # use the encoding expected by the downstream FC. TRANSPOSE and
            # RESHAPE are metadata-only here, so make all three tensor records
            # use the quantizer's output encoding before bypassing it.
            bmm_tensor = tensors[bmm_output]
            bmm_tensor.quantization = copy.deepcopy(quant_tensor.quantization)
            bmm_tensor.type = quant_tensor.type
            transpose_tensor.quantization = copy.deepcopy(quant_tensor.quantization)
            transpose_tensor.type = quant_tensor.type
            transpose_tensor.shape = list(quant_tensor.shape)
            reshape_op = reshape_users[0][1]
            reshape_op.inputs = [
                transpose_output if int(index) == quant_output else index
                for index in reshape_op.inputs
            ]
            remove_indices.add(quant_index)
            rewrites += 1

        if remove_indices:
            subgraph.operators = [
                op for index, op in enumerate(operators) if index not in remove_indices
            ]

    if rewrites != expected_paths:
        raise ValueError(
            f"Expected {expected_paths} BMM2 TRANSPOSE→QUANTIZE→RESHAPE paths; "
            f"removed {rewrites}"
        )
    rewritten = flatbuffer_utils.convert_object_to_bytearray(model)
    print(f"  [graph] removed {rewrites} post-BMM2 TRANSPOSE→QUANTIZE nodes")
    return bytes(rewritten)

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", "imagenette2-160")


def ensure_imagenette(folder: str | None) -> str:
    """Return a folder of calibration images, downloading Imagenette 160px if missing.

    Calibration needs images only, so the folder is read flat (no class structure).
    """
    if folder:
        return os.path.expanduser(folder)
    val_dir = os.path.join(IMAGENETTE_ROOT, "val")
    if os.path.isdir(val_dir) and any(os.scandir(val_dir)):
        return val_dir
    import tarfile, urllib.request
    os.makedirs(os.path.dirname(IMAGENETTE_ROOT), exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tgz = os.path.join(tmp, "imagenette2-160.tgz")
        print(f"  [data]  downloading Imagenette 160px from {IMAGENETTE_URL}")
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz)
        with tarfile.open(tgz) as t:
            t.extractall(os.path.dirname(IMAGENETTE_ROOT))
    if not os.path.isdir(val_dir):
        raise RuntimeError(f"Imagenette extracted but {val_dir} is missing")
    return val_dir


def load_calib(folder: str, n: int, imgsz: int = 224) -> list[dict]:
    """n calibration inputs from a flat image folder, ImageNet-normalized NCHW float32."""
    import pathlib
    from PIL.Image import open as pil_open
    tf = T.Compose(
        [T.Resize(round(imgsz * 256 / 224)), T.CenterCrop(imgsz), T.ToTensor(),
         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = [p for p in pathlib.Path(folder).rglob("*") if p.suffix.lower() in exts]
    paths = sorted(paths)[:n]
    print(f"  [data]  {len(paths)} calibration images from {folder}")
    return [{"args_0": tf(pil_open(str(p)).convert("RGB")).unsqueeze(0).numpy()} for p in paths]


def export_fp32(model, path, sample):
    pre = torch.export.export(model, (sample,)).module()
    _fuse_conv_bn_(pre)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    litert_torch.convert(pre, (sample,)).export(path)
    print(f"  [fp32]  {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=None, help="calibration image folder (default: auto-download Imagenette 160px val)")
    ap.add_argument("--calib", type=int, default=512)
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    fp32 = M.get_model("vit_b_16", weights=M.get_model_weights("vit_b_16").DEFAULT).eval()
    sample = torch.randn(*_SZ)
    with torch.no_grad():
        fp32(sample)
    _replace_mha_with_clean(fp32, softmax_add=True)
    with torch.no_grad():
        fp32(sample)

    fp32_path = os.path.join(a.outdir, "vit_b_16_fp32_softmax_add_attention_fc_add.tflite")
    export_fp32(fp32, fp32_path, sample)

    aeq_calib = load_calib(ensure_imagenette(a.dataset), a.calib)
    out = os.path.join(a.outdir, "vit_b_16_aeq_softmax_add_attention_fc_add_bmm2_output_no_quantize_int8.tflite")
    quantize_aeq_from_fp32(
        fp32_path, out, aeq_calib,
        softmax_add=True, attention_fc_add=True, bypass_bmm2_output_quantize=True,
    )


if __name__ == "__main__":
    main()
