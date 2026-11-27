"""Train CharNet32 on EMNIST (digits + letters).

The model, dataset pipeline and training loop live in main.py so the
desktop app and this script always use exactly the same preprocessing.
"""

import argparse

from main import (
    CHAR_CLASSES,
    MODEL_PATH,
    NUM_CLASSES,
    train_model,
    get_device,
)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Train the CharNet32 digit/letter recogniser."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="number of training epochs (default: 30)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="mini-batch size (default: 128)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="initial learning rate (default: 0.001)"
    )

    parser.add_argument(
        "--output",
        default=MODEL_PATH,
        help=f"where to save the weights (default: {MODEL_PATH})"
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = get_device()

    print("================================")
    print("       CHARNET32 TRAINING")
    print("================================")
    print("Training on:", device)
    print("Classes:", NUM_CLASSES)
    print(" ".join(CHAR_CLASSES))
    print()

    train_model(
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_path=args.output
    )


if __name__ == "__main__":
    main()
