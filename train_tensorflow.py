import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image as keras_image

from src.utils import preprocess_tf_image, augment_tf_image

def build_tf_model(num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

def plot_learning_curves(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train acc')
    plt.plot(history.history['val_accuracy'], label='Val acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/learning_curves.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    plt.show()

def train_tensorflow(train_dir='data/breast_cancer/train', test_dir='data/breast_cancer/test'):
    print("Chargement des données...")

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=(224, 224), batch_size=32, label_mode='int'
    )
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        test_dir, image_size=(224, 224), batch_size=32, label_mode='int'
    )

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    train_ds = train_ds_raw.map(lambda x, y: (preprocess_tf_image(x), y)) \
                           .map(augment_tf_image).prefetch(tf.data.AUTOTUNE)

    test_ds = test_ds_raw.map(lambda x, y: (preprocess_tf_image(x), y)) \
                         .prefetch(tf.data.AUTOTUNE)

    model = build_tf_model(num_classes)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Entraînement du modèle TensorFlow...")
    history = model.fit(train_ds, epochs=10, validation_data=test_ds)

    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    print("\nAccuracy:", np.mean(np.array(y_pred) == np.array(y_true)))
    print("\nConfusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=class_names))

    plot_learning_curves(history)
    plot_confusion_matrix(y_true, y_pred, class_names)

    os.makedirs('models', exist_ok=True)
    model.save('models/Aly_model.keras')
    print("Modèle sauvegardé dans 'models/Aly_model.keras'")

    return model, class_names

def predict_tensorflow(img_path, model=None, model_path='models/Aly_model.keras'):
    """
    Prédit la classe d'une image à partir d'un modèle TensorFlow.
    - Si model est donné, il est utilisé directement.
    - Sinon, le modèle est chargé depuis model_path.
    """
    try:
        if model is None:
            model = tf.keras.models.load_model(model_path)

        img = keras_image.load_img(img_path, target_size=(224, 224))
        img_array = keras_image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=1)[0]
        confidence = float(preds[0][predicted_class]) * 100
        return predicted_class, confidence, preds[0]
    except Exception as e:
        print(f"[ERREUR - Prédiction] {e}")
        return None, None, None

if __name__ == "__main__":
    model, class_names = train_tensorflow()

    test_image_path = 'data/breast_cancer/test/NORMAL/sample_image.jpeg'
    predicted_class, confidence, scores = predict_tensorflow(test_image_path, model=model)
    if predicted_class is not None:
        print(f"\nPrédiction : {class_names[predicted_class]} ({confidence:.2f}%)")
