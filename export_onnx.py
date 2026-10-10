import argparse
import os
import torch
from laya.agent import Agent


def quantize_model(model_path: str, output_path: str, per_channel: bool = False) -> str:
    """Write an INT8 weight-only dynamically quantized copy of model_path."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnx

    model = onnx.load(model_path)
    del model.graph.value_info[:]
    quantize_dynamic(
        model_input=model,
        model_output=output_path,
        op_types_to_quantize=["MatMul"],
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
    )
    # Inline external data if needed to make model self-contained
    m = onnx.load(output_path)
    has_ext = any(init.HasField("data_location") for init in m.graph.initializer)
    if has_ext:
        for init in m.graph.initializer:
            if init.HasField("data_location"):
                init.ClearField("data_location")
                del init.external_data[:]
        onnx.save(m, output_path)

    return output_path


def int8_output_path(output_path: str) -> str:
    root, ext = os.path.splitext(output_path)
    return "%s.int8%s" % (root, ext or ".onnx")


def export_to_onnx(model_id_or_path: str, output_path: str):
    print(f"Loading PyTorch Agent from: {model_id_or_path}")
    agent = Agent(model_id_or_path, compile=False, device="cpu")

    batch, seq_len, num_markers = 2, 17, 3
    dummy_input_ids = torch.randint(0, 100, (batch, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch, seq_len), dtype=torch.long)
    dummy_marker_pos = torch.tensor([[1, 5, 9]] * batch, dtype=torch.long)
    dummy_marker_mask = torch.ones((batch, num_markers), dtype=torch.bool)
    dummy_qtype = torch.zeros(batch, dtype=torch.long)

    inputs = (
        dummy_input_ids,
        dummy_attention_mask,
        dummy_marker_pos,
        dummy_marker_mask,
        dummy_qtype,
    )

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }

    input_names = [
        "input_ids",
        "attention_mask",
        "marker_pos",
        "marker_mask",
        "qtype",
    ]
    output_names = ["logits", "act_logits"]

    print(f"Exporting to {output_path}...")
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        agent.model,
        inputs,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"Successfully exported ONNX model to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Laya model to ONNX format")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya", help="HuggingFace Hub ID or local path")
    parser.add_argument("--output", type=str, default="laya.onnx", help="Output path for the ONNX file")
    parser.add_argument("--quantize", action="store_true", help="Also write an INT8 weight-only quantized copy")
    parser.add_argument("--per-channel", action="store_true", help="Quantize weights per output channel instead of per tensor")
    args = parser.parse_args()

    export_to_onnx(args.model, args.output)
    if args.quantize:
        int8_path = quantize_model(args.output, int8_output_path(args.output), per_channel=args.per_channel)
        print(f"Successfully wrote INT8 quantized model to: {int8_path}")
