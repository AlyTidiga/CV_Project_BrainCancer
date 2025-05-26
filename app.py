import os
import sys
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# Ajouter le dossier src au chemin pour importer train_tensorflow
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from train_tensorflow import predict_tensorflow

import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MODEL_PATH'] = 'models/Aly_model.keras'  # Le chemin du modèle entraîné

# Charger le modèle une seule fois ici
model = tf.keras.models.load_model(app.config['MODEL_PATH'])

# Classes
class_names = ['BENIGN', 'MALIGNANT', 'NORMAL']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error='Aucune image reçue.')

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error='Nom de fichier vide.')

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        #  Appel du modèle
        pred_class_idx, confidence, _ = predict_tensorflow(filepath, model=model)

        if pred_class_idx is None:
            return render_template('index.html', error="Erreur de prédiction.")

        predicted_class = class_names[pred_class_idx]

        return render_template('index.html',
                               image_filename=filename,
                               predicted_class=predicted_class,
                               confidence=round(confidence, 2))

    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
