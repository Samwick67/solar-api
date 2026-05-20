from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app)

# ==========================================
# HEALTH CHECK (IMPORTANT FOR RENDER)
# ==========================================
@app.route("/health")
def health():
    return "ok"

# ==========================================
# DATASET
# ==========================================

data = {
    "Hour": [6,7,8,9,10,11,12,13,14,15,16,17,18],
    "Loadshedding": [1,1,2,2,2,3,3,3,2,2,2,1,1],
    "Light_Intensity": [120,250,380,520,650,780,900,870,760,620,450,300,150],
    "Battery_Level": [62,65,68,72,78,82,90,92,88,84,76,69,61],
    "Panel_Angle": [20,30,40,50,60,75,90,100,110,120,130,140,150],
    "Weather": ["Sunny"]*10 + ["Cloudy"]*3
}

df = pd.DataFrame(data)

# ==========================================
# ENCODING
# ==========================================

encoder = LabelEncoder()
df["Weather_Encoded"] = encoder.fit_transform(df["Weather"])

# ==========================================
# MODEL
# ==========================================

X = df[[
    "Hour",
    "Loadshedding",
    "Light_Intensity",
    "Battery_Level",
    "Weather_Encoded"
]]

y = df["Panel_Angle"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ==========================================
# ROUTES
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # validate input safely
        required_fields = [
            "hour",
            "loadshedding",
            "light_intensity",
            "battery_level",
            "weather"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # encode weather safely
        try:
            weather_encoded = encoder.transform([data["weather"]])[0]
        except:
            weather_encoded = 0  # fallback if unknown value

        input_data = pd.DataFrame([{
            "Hour": int(data["hour"]),
            "Loadshedding": int(data["loadshedding"]),
            "Light_Intensity": int(data["light_intensity"]),
            "Battery_Level": int(data["battery_level"]),
            "Weather_Encoded": weather_encoded
        }])

        prediction = model.predict(input_data)

        return jsonify({
            "predicted_angle": float(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
