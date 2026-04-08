from flask import Flask, request, render_template
from pathlib import Path
import pickle

BASE_DIR      = Path(__file__).resolve().parent.parent
MODELS_DIR    = BASE_DIR / "models"

app           = Flask(__name__)
loaded_model  = pickle.load(open(MODELS_DIR / "smartphone_addiction_model.sav", "rb"))
loaded_scaler = pickle.load(open(MODELS_DIR / "scaler.sav", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        raw = [[
            float(request.form["daily_screen_time"]),
            float(request.form["weekend_screen_time"]),
            float(request.form["social_media_hours"]),
            float(request.form["app_opens"])
        ]]
        result = loaded_model.predict(loaded_scaler.transform(raw))[0]
        prediction = "Addicted 📵" if result == 1 else "Not Addicted ✅"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)