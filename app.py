from flask import Flask, render_template, request, redirect
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Flask app
app = Flask(__name__)

# Upload folder path
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model = load_model("model/garbage_classifier.h5")

# Class labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


# Prediction
def predict_image(img_path):
    img = Image.open(img_path)

    img = img.convert("RGB")
    img = img.resize((180, 180))

    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = round(np.max(prediction[0]) * 100, 2)

    return predicted_class, confidence


# Home page
@app.route("/")
def home():
    return render_template("home.html")


# Prediction page
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    predicted_class, confidence = predict_image(filepath)

    # Eco suggestions
    suggestions = {

        "cardboard":
        "📦 Recycle cardboard materials or reuse them for packaging and storage purposes.",

        "glass":
        "🍾 Glass can be recycled repeatedly without losing quality. Dispose it in glass recycling bins.",

        "metal":
        "🔩 Metals are highly recyclable. Recycling helps conserve natural resources and energy.",

        "paper":
        "📄 Recycle paper waste or reuse sheets whenever possible to reduce deforestation.",

        "plastic":
        "♻ Reduce single-use plastics whenever possible. Consider recycling or advanced methods like pyrolysis.",

        "trash":
        "🗑 Dispose non-recyclable waste responsibly and minimize environmental pollution."
    }
    links = {

    "cardboard":
    "https://earth911.com/recycling-guide/how-to-recycle-cardboard/",

    "glass":
    "https://earth911.com/recycling-guide/how-to-recycle-glass/",

    "metal":
    "https://earth911.com/recycling-guide/how-to-recycle-metal/",

    "paper":
    "https://earth911.com/recycling-guide/how-to-recycle-paper/",

    "plastic":
    "https://www.unep.org/interactives/beat-plastic-pollution/",

    "trash":
    "https://www.epa.gov/recycle/reducing-waste-what-you-can-do"
}
    
    resource_link = links[predicted_class]

    eco_tip = suggestions[predicted_class]

    image_path = "uploads/" + file.filename

    return render_template(
        "result.html",
        prediction=predicted_class,
        confidence=confidence,
        image_path=image_path,
        eco_tip=eco_tip,
resource_link=resource_link
    )

#About page
@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/uploads")
def uploads():
    upload_folder = "static/uploads"
    images = os.listdir(upload_folder)
    
    return render_template(
        "uploads.html",
        images=images
    )

@app.route("/delete_images", methods=["POST"])
def delete_images():

    selected_images = request.form.getlist("delete_images")

    for img in selected_images:

        img_path = os.path.join(
            "static/uploads",
            img
        )

        if os.path.exists(img_path):
            os.remove(img_path)

    return redirect("/uploads")

    
# Run app
if __name__ == "__main__":
    app.run(debug=True)