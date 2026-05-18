from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if sys.path and Path(sys.path[0]).resolve() == SCRIPT_DIR:
    sys.path.pop(0)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _print_onnx_shapes(path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime is not installed; skipping exported shape inspection")
        return

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for input_info in session.get_inputs():
        print(f"input {input_info.name} {input_info.shape} {input_info.type}")
    for output_info in session.get_outputs():
        print(f"output {output_info.name} {output_info.shape} {output_info.type}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a YOLO model to dynamic-batch ONNX for batched inference."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("weights/yolo26s.pt"),
        help="Source YOLO .pt weights.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square image size used for export.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Nominal export batch size for dynamic ONNX.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Optional ONNX opset override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights))
    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "dynamic": True,
        "batch": args.batch,
        "simplify": True,
    }
    if args.opset is not None:
        export_kwargs["opset"] = args.opset
    exported = Path(model.export(**export_kwargs))
    print(f"Exported ONNX model: {exported}")
    _print_onnx_shapes(exported)


if __name__ == "__main__":
    main()
