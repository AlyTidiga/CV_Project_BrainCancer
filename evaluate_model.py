import argparse
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from cnn_pytorch import CNNPyTorch
from src.train_tensorflow import create_tf_model
from utils.prep import get_pytorch_loaders

def plot_confusion_matrix(y_true, y_pred, classes, title="Matrice de confusion"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.show()

def test_pytorch():
    print(" Évaluation du modèle PyTorch")
    _, test_loader, num_classes = get_pytorch_loaders()
    model = CNNPyTorch(num_classes)
    model.load_state_dict(torch.load("models/cnn_pytorch.torch"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f" Précision : {accuracy * 100:.2f}%")
    print("\n Rapport de classification :\n", classification_report(all_labels, all_preds))
    plot_confusion_matrix(all_labels, all_preds, classes=[str(i) for i in range(num_classes)])

def test_tensorflow():
    print(" Évaluation du modèle TensorFlow")

    # Charger les données
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        "data/testing",
        image_size=(64, 64),
        batch_size=32
    )
    class_names = test_dataset.class_names
    num_classes = len(class_names)

    # Prétraitement
    test_dataset = test_dataset.map(lambda x, y: (x / 255.0, y))

    # Charger le modèle
    model = tf.keras.models.load_model("models/cnn_tensorflow.keras")

    # Prédiction
    y_true = []
    y_pred = []

    for images, labels in test_dataset:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    print(f" Précision : {accuracy * 100:.2f}%")
    print("\n Rapport de classification :\n", classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion_matrix(y_true, y_pred, classes=class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=["torch", "tensorflow"], required=True)
    args = parser.parse_args()

    if args.framework == "torch":
        test_pytorch()
    else:
        test_tensorflow()
