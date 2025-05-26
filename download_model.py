import gdown
import os

file_id = "1Zh0yKz1zmF9l_5-oATn84txtFrkWasx8"
output = "models/Aly_model.keras"

os.makedirs(os.path.dirname(output), exist_ok=True)
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(output):
    print("📥 Téléchargement du modèle...")
    gdown.download(url, output, quiet=False)
    print("✅ Modèle téléchargé.")
else:
    print("✅ Le modèle est déjà présent.")
