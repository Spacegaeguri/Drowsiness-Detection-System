"""Evaluate a separate eye-image directory; report counts and exact threshold."""

import argparse
import json
from pathlib import Path

CLASS_NAMES = ["close eyes", "open eyes"]
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "best_eye_model.keras"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if not args.model.is_file() or not args.data_dir.is_dir():
        parser.error("Model file and data directory must exist")
    classes = sorted(p.name for p in args.data_dir.iterdir() if p.is_dir())
    if classes != CLASS_NAMES:
        parser.error(f"Expected classes {CLASS_NAMES}; got {classes}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    dataset = tf.keras.utils.image_dataset_from_directory(
        args.data_dir, image_size=(64, 64), batch_size=32,
        label_mode="binary", color_mode="rgb", shuffle=False,
        class_names=CLASS_NAMES,
    )
    model = tf.keras.models.load_model(args.model, compile=False)
    true_batches, predicted_batches, score_batches = [], [], []
    for images, labels in dataset:
        scores = model.predict(images, verbose=0).reshape(-1)
        true_batches.append(labels.numpy().reshape(-1).astype(int))
        predicted_batches.append((scores >= args.threshold).astype(int))
        score_batches.append(scores)
    truth = np.concatenate(true_batches)
    predictions = np.concatenate(predicted_batches)
    scores = np.concatenate(score_batches)
    loss = tf.keras.losses.binary_crossentropy(
        truth.reshape(-1, 1).astype("float32"), scores.reshape(-1, 1)
    )
    report = {
        "model": str(args.model.resolve()), "data_dir": str(args.data_dir.resolve()),
        "class_names": CLASS_NAMES, "threshold": args.threshold,
        "samples": int(truth.size), "accuracy": float(accuracy_score(truth, predictions)),
        "binary_crossentropy": float(tf.reduce_mean(loss).numpy()),
        "confusion_matrix": confusion_matrix(truth, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            truth, predictions, labels=[0, 1], target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        ),
        "scope": "CNN only; independence depends on the supplied data split.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
