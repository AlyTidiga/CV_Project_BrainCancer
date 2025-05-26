import argparse
from src.train_pytorch import train_pytorch_model  
from src.train_tensorflow import train_tensorflow


def main():
    parser = argparse.ArgumentParser(description="Train Breast Cancer CNN model")
    parser.add_argument(
        '--model',
        choices=['pytorch', 'tensorflow'],
        required=True,
        help="Choix du framework pour entraîner le modèle"
    )
    args = parser.parse_args()

    if args.model == 'pytorch':
        print(" Entraînement du modèle PyTorch")
        train_pytorch_model()
    else:
        print(" Entraînement du modèle TensorFlow")
        train_tensorflow()

if __name__ == "__main__":
    main()
