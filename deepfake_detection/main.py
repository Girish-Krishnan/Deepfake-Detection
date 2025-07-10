import argparse

from .train import train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Deepfake detection training")
    parser.add_argument("--model", required=True, help="Model name: basic_cnn, resnet, inception, vit, xception, wavelet_clip")
    parser.add_argument("--train-real", required=True, help="Path to directory with real training images")
    parser.add_argument("--train-fake", required=True, help="Path to directory with fake training images")
    parser.add_argument("--val-real", required=True, help="Path to directory with real validation images")
    parser.add_argument("--val-fake", required=True, help="Path to directory with fake validation images")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", default=None, help="Device to train on")
    parser.add_argument("--save-model", default=None, help="Path to save best model")
    return parser.parse_args()


def main():
    args = parse_args()
    train_model(
        model_name=args.model,
        train_real=args.train_real,
        train_fake=args.train_fake,
        val_real=args.val_real,
        val_fake=args.val_fake,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_path=args.save_model,
    )


if __name__ == "__main__":
    main()

