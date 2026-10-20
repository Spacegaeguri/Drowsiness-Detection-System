"""Recreate the documented 64x64 eye CNN; keep deployment artifacts intact."""

import argparse
import json
from pathlib import Path

CLASS_NAMES = ["close eyes", "open eyes"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.epochs <= 15:
        parser.error("--epochs must be between 1 and 15")
    if args.train_dir.resolve() == args.validation_dir.resolve():
        parser.error("Training and validation directories must be different")
    for directory in (args.train_dir, args.validation_dir):
        if not directory.is_dir():
            parser.error(f"Directory not found: {directory}")
        classes = sorted(p.name for p in directory.iterdir() if p.is_dir())
        if classes != CLASS_NAMES:
            parser.error(f"Expected classes {CLASS_NAMES}; got {classes} in {directory}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    common = dict(image_size=(64, 64), batch_size=32, label_mode="binary",
                  color_mode="rgb", class_names=CLASS_NAMES)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.train_dir, shuffle=True, seed=args.seed, **common
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        args.validation_dir, shuffle=False, **common
    )
    # Pixel normalization is inside the model, including at browser inference.
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    validation_ds = validation_ds.prefetch(tf.data.AUTOTUNE)
    layers = tf.keras.layers
    model = tf.keras.Sequential([
        layers.Input(shape=(64, 64, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()
    args.output.mkdir(parents=True, exist_ok=False)
    run = {
        "train_dir": str(args.train_dir.resolve()),
        "validation_dir": str(args.validation_dir.resolve()),
        "class_names": CLASS_NAMES, "image_size": [64, 64, 3],
        "seed": args.seed, "epochs_max": args.epochs, "batch_size": 32,
        "tensorflow": tf.__version__,
        "note": "New run; historical Colab metrics are not guaranteed to reproduce.",
    }
    (args.output / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(args.output / "best_eye_model.keras"),
            monitor="val_loss", save_best_only=True
        ),
    ]
    history = model.fit(train_ds, validation_data=validation_ds,
                        epochs=args.epochs, callbacks=callbacks)
    (args.output / "history.json").write_text(
        json.dumps(history.history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
