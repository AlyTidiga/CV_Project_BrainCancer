# src/utils.py

import torch
from torchvision import transforms
import tensorflow as tf
import numpy as np
from PIL import Image

# Classes de prédiction
CLASS_NAMES = ['benign', 'malignant']

def get_pytorch_transforms(train=True, normalize=True, augment=True):
    """Retourne les transformations pour PyTorch"""
    transform_list = []
    if augment and train:
        transform_list += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ]
    transform_list.append(transforms.Resize((224, 224)))
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize([0.5], [0.5]))
    return transforms.Compose(transform_list)

def preprocess_tf_image(image):
    """Prétraitement pour TensorFlow"""
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image

def augment_tf_image(image, label):
    """Augmentations pour TensorFlow"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    return image, label

def preprocess_image_for_inference(image_path, framework):
    """Préparation d'une image pour la prédiction"""
    image = Image.open(image_path).convert('RGB')
    if framework == 'pytorch':
        transform = get_pytorch_transforms(train=False)
        image = transform(image).unsqueeze(0)  # batch de 1
        return image
    elif framework == 'tensorflow':
        image = image.resize((224, 224))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)  # batch de 1
        return image
    else:
        raise ValueError("Framework non supporté")
