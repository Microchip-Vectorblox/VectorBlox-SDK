"""anomaly_quantize.py — NORM-aware int8 quantization of a fp32 TFLite.

Scans for the LayerNorm op sequence by structure (no name heuristics),
keeps those ops fp32, and quantizes everything else to int8.

Usage:
    python anomaly_quantize.py exports/SMD_fp32.tflite
    python anomaly_quantize.py exports/SMD_fp32_fc.tflite --output exports/SMD_int8.tflite
    python anomaly_quantize.py exports/SMD_fp32.tflite --dataset MSL --calib 128
"""

import argparse, os, sys, tempfile
import numpy as np
import ai_edge_litert.interpreter as litert
from ai_edge_quantizer import quantizer as aeq, algorithm_manager
from ai_edge_quantizer.qtyping import TFLOperationName

sys.path.insert(0, os.path.dirname(__file__))
from anomaly_train import get_synthetic_loader
from anomaly_export import PRESETS

# The 12-op LayerNorm sequence litert_torch always emits for F.layer_norm.
_LN_SEQ = [
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
]


def find_ln_tensor_names(fp32_path: str) -> list[str]:
    """Scan fp32 TFLite for LN op sequences; return their output tensor names."""
    interp = litert.Interpreter(fp32_path)
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    tensors = {t["index"]: t for t in interp.get_tensor_details()}
    types = [op["op_name"] for op in ops]
    n = len(_LN_SEQ)
    names = []
    for i in range(len(types) - n + 1):
        if types[i : i + n] == _LN_SEQ:
            for op in ops[i : i + n]:
                for idx in op["outputs"]:
                    if idx >= 0:
                        names.append(tensors[idx]["name"])
    if not names:
        raise ValueError("No LN sequence found — wrong model or fp32 path?")
    return names


def quantize(
    fp32_path: str, out_path: str, win_size: int, n_channels: int, calib: int
) -> str:
    recipe = [
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

    qt = aeq.Quantizer(fp32_path)
    qt.load_quantization_recipe(recipe)

    ln_names = find_ln_tensor_names(fp32_path)
    ln_regex = "|".join(ln_names)
    print(
        f"[norm]  {len(ln_names) // len(_LN_SEQ)} LN blocks, {len(ln_names)} tensors → fp32"
    )

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

    loader = get_synthetic_loader(
        batch_size=1,
        win_size=win_size,
        n_channels=n_channels,
        n_windows=calib,
        mode="train",
        seed=42,
    )
    calib_data = [{"args_0": x.numpy().astype(np.float32)} for x, _ in loader]
    print(f"[quant] calibrating ({calib} windows) …")
    result = qt.quantize(qt.calibrate({"serving_default": calib_data}))

    with tempfile.NamedTemporaryFile(suffix=".tflite", delete=False) as tmp:
        tmp_path = tmp.name
    result.export_model(tmp_path, overwrite=True)
    flatbuf = open(tmp_path, "rb").read()
    os.unlink(tmp_path)

    open(out_path, "wb").write(flatbuf)
    print(f"[int8]  {out_path}  ({len(flatbuf) / 1e6:.1f} MB)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("fp32", help="input fp32 TFLite")
    ap.add_argument(
        "--output",
        "-o",
        default=None,
        help="output path (default: <input stem>_norm_int8.tflite)",
    )
    ap.add_argument("--dataset", default="SMD", choices=list(PRESETS.keys()))
    ap.add_argument("--calib", type=int, default=64)
    a = ap.parse_args()

    cfg = PRESETS[a.dataset]
    out = a.output or os.path.join(
        os.path.dirname(a.fp32),
        os.path.splitext(os.path.basename(a.fp32))[0] + "_norm_int8.tflite",
    )
    quantize(a.fp32, out, cfg["win_size"], cfg["enc_in"], a.calib)


if __name__ == "__main__":
    main()
