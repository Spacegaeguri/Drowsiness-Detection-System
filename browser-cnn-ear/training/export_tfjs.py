"""Export a 64x64 Keras eye model to a fresh TensorFlow.js directory."""

import argparse
import json
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "best_eye_model.keras"


def normalize_input_config(value):
    """Adapt the Keras 3 InputLayer shape key for TF.js LayersModel loading."""
    if isinstance(value, dict):
        if value.get("class_name") == "InputLayer":
            config = value["config"]
            if "batch_shape" in config:
                config["batch_input_shape"] = config.pop("batch_shape")
        for child in value.values():
            normalize_input_config(child)
    elif isinstance(value, list):
        for child in value:
            normalize_input_config(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    import tensorflow as tf
    import tensorflowjs as tfjs

    original = tf.keras.models.load_model(args.model, compile=False)
    if original.input_shape != (None, 64, 64, 3) or original.output_shape != (None, 1):
        parser.error("Expected a 64x64 RGB eye model with one sigmoid output")
    wrapped = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 3)), original
    ])
    wrapped(tf.zeros((1, 64, 64, 3)))
    args.output.mkdir(parents=True, exist_ok=False)
    tfjs.converters.save_keras_model(wrapped, str(args.output))
    model_json = args.output / "model.json"
    document = json.loads(model_json.read_text(encoding="utf-8"))
    normalize_input_config(document["modelTopology"])
    model_json.write_text(json.dumps(document, indent=2), encoding="utf-8")
    for group in document["weightsManifest"]:
        for shard in group["paths"]:
            if not (args.output / shard).is_file():
                raise RuntimeError(f"Missing weight shard: {shard}")
    print(f"Exported {args.output}; verify loading and inference before deployment.")


if __name__ == "__main__":
    main()
