"""anomaly_FFN_rewrite.py — rewrite Conv1d FFN ops to FULLY_CONNECTED in a fp32 TFLite.

Finds the pattern:
    TRANSPOSE RESHAPE TRANSPOSE CONV_2D RESHAPE ADD
    TRANSPOSE GELU
    RESHAPE TRANSPOSE CONV_2D RESHAPE ADD

and replaces it with:
    FULLY_CONNECTED GELU FULLY_CONNECTED

Weight conversion: squeeze [out,1,1,in] → [out,in].
Bias conversion:   squeeze [1,1,C]      → [C].

The rewritten model is numerically identical to the original.
Intended to run between anomaly_export_int8.py --fp32_only and
anomaly_export_int8.py --fp32 <rewritten.tflite>.

Usage:
    python anomaly_FFN_rewrite.py exports/SMD_fp32.tflite exports/SMD_fp32_fc.tflite
"""

import argparse
import os
import sys
import numpy as np

# ── TFLite opcode constants ───────────────────────────────────────────────────
_ADD = 0
_CONV_2D = 3
_FC = 9
_RESHAPE = 22
_TRANSPOSE = 39
_GELU = 150

_FFN_PATTERN = [
    _TRANSPOSE,
    _RESHAPE,
    _TRANSPOSE,  # pre-conv1 layout
    _CONV_2D,
    _RESHAPE,
    _ADD,  # conv1 + bias
    _TRANSPOSE,
    _GELU,  # activation
    _RESHAPE,
    _TRANSPOSE,  # pre-conv2 layout
    _CONV_2D,
    _RESHAPE,
    _ADD,  # conv2 + bias
]


# ── helpers ───────────────────────────────────────────────────────────────────


def _opcodes(model, sg):
    return [model.operatorCodes[op.opcodeIndex].builtinCode for op in sg.operators]


def _ensure_opcode(model, builtin_code):
    from ai_edge_litert import schema_py_generated as schema

    for i, oc in enumerate(model.operatorCodes):
        if oc.builtinCode == builtin_code:
            return i
    oc = schema.OperatorCodeT()
    oc.builtinCode = builtin_code
    model.operatorCodes.append(oc)
    return len(model.operatorCodes) - 1


# ── rewrite ───────────────────────────────────────────────────────────────────


def rewrite_ffn(model) -> int:
    """Replace every FFN Conv1d block with FC+GELU+FC in-place.
    Returns number of blocks rewritten.
    """
    from ai_edge_litert import schema_py_generated as schema

    sg = model.subgraphs[0]
    codes = _opcodes(model, sg)
    n = len(_FFN_PATTERN)
    count = 0

    i = 0
    while i <= len(codes) - n:
        if codes[i : i + n] != _FFN_PATTERN:
            i += 1
            continue

        ops = sg.operators[i : i + n]

        # Boundary tensors that cross the block
        ffn_in = ops[0].inputs[0]  # [1, L, d_model] input to first TRANSPOSE
        ffn_out = ops[12].outputs[0]  # [1, L, d_model] output of last ADD

        # Extract conv weights and biases
        def _get(idx):
            t = sg.tensors[idx]
            raw = model.buffers[t.buffer].data
            return np.frombuffer(raw, dtype=np.float32).reshape(t.shape)

        w1 = _get(ops[3].inputs[1]).squeeze()  # [out, in]
        b1 = _get(ops[5].inputs[1]).squeeze()  # [out]
        w2 = _get(ops[10].inputs[1]).squeeze()  # [out, in]
        b2 = _get(ops[12].inputs[1]).squeeze()  # [out]

        # Add new weight/bias tensors
        def _add_const(name, arr):
            buf = schema.BufferT()
            buf.data = arr.astype(np.float32).tobytes()
            model.buffers.append(buf)
            t = schema.TensorT()
            t.name = name.encode()
            t.type = 0  # FLOAT32
            t.shape = list(arr.shape)
            t.buffer = len(model.buffers) - 1
            sg.tensors.append(t)
            return len(sg.tensors) - 1

        def _add_activation(name, shape):
            t = schema.TensorT()
            t.name = name.encode()
            t.type = 0
            t.shape = list(shape)
            t.buffer = 0  # null buffer — runtime allocated
            sg.tensors.append(t)
            return len(sg.tensors) - 1

        fc1_w = _add_const(f"ffn{count}_fc1_w", w1)
        fc1_b = _add_const(f"ffn{count}_fc1_b", b1)
        fc2_w = _add_const(f"ffn{count}_fc2_w", w2)
        fc2_b = _add_const(f"ffn{count}_fc2_b", b2)

        L = sg.tensors[ffn_in].shape[1]
        d_ff = w1.shape[0]
        fc1_out = _add_activation(f"ffn{count}_fc1_out", [1, L, d_ff])
        gelu_out = _add_activation(f"ffn{count}_gelu_out", [1, L, d_ff])

        # Build replacement ops
        def _make_fc(in_t, w_t, b_t, out_t):
            op = schema.OperatorT()
            op.opcodeIndex = _ensure_opcode(model, _FC)
            op.inputs = [in_t, w_t, b_t]
            op.outputs = [out_t]
            opt = schema.FullyConnectedOptionsT()
            opt.fusedActivationFunction = 0
            opt.weightsFormat = 0
            opt.keepNumDims = True
            op.builtinOptionsType = schema.BuiltinOptions.FullyConnectedOptions
            op.builtinOptions = opt
            return op

        def _make_gelu(in_t, out_t):
            op = schema.OperatorT()
            op.opcodeIndex = _ensure_opcode(model, _GELU)
            op.inputs = [in_t]
            op.outputs = [out_t]
            opt = schema.GeluOptionsT()
            opt.approximate = False
            op.builtinOptionsType = schema.BuiltinOptions.GeluOptions
            op.builtinOptions = opt
            return op

        new_ops = [
            _make_fc(ffn_in, fc1_w, fc1_b, fc1_out),
            _make_gelu(fc1_out, gelu_out),
            _make_fc(gelu_out, fc2_w, fc2_b, ffn_out),
        ]

        # Mark internal tensors as orphaned so the calibrator skips them
        boundary = {ffn_in, ffn_out}
        for op in ops:
            for idx in op.outputs or []:
                if idx >= 0 and idx not in boundary:
                    sg.tensors[idx].name = f"_removed_{count}_{idx}".encode()
                    sg.tensors[idx].shape = [0]

        sg.operators[i : i + n] = new_ops
        codes[i : i + n] = [_FC, _GELU, _FC]

        print(
            f"  [ffn{count}] {w1.shape[1]}→{w1.shape[0]}→{w2.shape[0]}  "
            f"ops[{i}:{i + n}] → FC+GELU+FC"
        )
        count += 1
        i += 3

    return count


# ── buffer compaction ─────────────────────────────────────────────────────────


def compact_buffers(model) -> int:
    """Remove buffers that are no longer referenced by any live tensor.

    Tensors orphaned by the FFN rewrite (shape=[0]) still reference the old
    Conv1d weight buffers.  This walks all tensors, collects which buffers are
    still needed, rebuilds the buffer list, and updates every tensor.buffer.

    Returns bytes freed.
    """
    sg = model.subgraphs[0]

    # Collect tensor indices still referenced by a live op
    live_tensors: set[int] = set()

    def _iter(xs):
        return xs if xs is not None else []

    for op in sg.operators:
        for idx in list(_iter(op.inputs)) + list(_iter(op.outputs)):
            if int(idx) >= 0:
                live_tensors.add(int(idx))
    for idx in list(_iter(sg.inputs)) + list(_iter(sg.outputs)):
        if int(idx) >= 0:
            live_tensors.add(int(idx))

    # Collect buffer indices needed by live tensors (always keep buffer 0)
    needed: set[int] = {0}
    for i, t in enumerate(sg.tensors):
        if i in live_tensors:
            needed.add(t.buffer)

    # Build compacted buffer list and old→new index map
    new_buffers = []
    remap: dict[int, int] = {}
    for old_idx, buf in enumerate(model.buffers):
        if old_idx in needed:
            remap[old_idx] = len(new_buffers)
            new_buffers.append(buf)

    freed = sum(
        len(model.buffers[i].data) if model.buffers[i].data is not None else 0
        for i in range(len(model.buffers))
        if i not in needed
    )

    # Update ALL tensor.buffer references; zero shape on anything pointing to
    # the null buffer (so calibrator's np.all(shape) guard skips them)
    for t in sg.tensors:
        new_buf = remap.get(t.buffer, 0)
        if new_buf == 0 and t.buffer != 0:
            # was a real buffer, now orphaned → mark shape [0] so calibrator skips it
            t.shape = [0]
        t.buffer = new_buf

    # Update model-level metadata buffer references (version info, etc.)
    if model.metadata:
        for m in model.metadata:
            if hasattr(m, "buffer") and m.buffer is not None:
                m.buffer = remap.get(int(m.buffer), 0)

    model.buffers = new_buffers
    return freed


# ── pack flatbuffer ───────────────────────────────────────────────────────────


def pack(model) -> bytes:
    import flatbuffers

    builder = flatbuffers.Builder(1 << 24)
    root = model.Pack(builder)
    builder._Builder__Finish(root, False, b"TFL3")
    return bytes(builder.Output())


# ── verify numerics ───────────────────────────────────────────────────────────


def verify(orig_path: str, new_path: str) -> None:
    import ai_edge_litert.interpreter as litert

    def _run(path, x):
        interp = litert.Interpreter(path)
        interp.allocate_tensors()
        id_ = interp.get_input_details()[0]
        od_ = interp.get_output_details()[0]
        interp.set_tensor(id_["index"], x)
        interp.invoke()
        return interp.get_tensor(od_["index"])

    np.random.seed(0)
    x = np.random.randn(1, 100, 38).astype(np.float32)

    out_orig = _run(orig_path, x)
    out_new = _run(new_path, x)

    diff = np.abs(out_orig - out_new)
    print(f"  max |Δ| = {diff.max():.2e}   mean |Δ| = {diff.mean():.2e}", end="  ")
    print("✓ PASS" if diff.max() < 1e-4 else "✗ FAIL")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", help="fp32 TFLite with Conv1d FFN")
    ap.add_argument("output", help="fp32 TFLite with FULLY_CONNECTED FFN")
    ap.add_argument(
        "--no-verify", action="store_true", help="skip numerical verification"
    )
    a = ap.parse_args()

    from ai_edge_litert import schema_py_generated as schema

    print(f"Loading {a.input} …")
    buf = open(a.input, "rb").read()
    model = schema.ModelT.InitFromPackedBuf(buf, 0)

    sg = model.subgraphs[0]
    print(f"  {len(sg.operators)} ops before rewrite")

    n = rewrite_ffn(model)
    print(f"  {n} FFN blocks rewritten → {len(sg.operators)} ops after")

    freed = compact_buffers(model)
    print(f"  compacted buffers: freed {freed / 1e6:.1f} MB")

    out_buf = pack(model)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    open(a.output, "wb").write(out_buf)
    print(f"  saved → {a.output}  ({len(out_buf) / 1e6:.1f} MB)")

    if not a.no_verify:
        print("Verifying …")
        verify(a.input, a.output)


if __name__ == "__main__":
    main()
