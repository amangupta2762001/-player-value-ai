from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Health check route
@app.route('/')
def home():
    return jsonify({"message": "API is working"}), 200

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

    data = request.get_json()

    try:
        # Load model and encoder
        model = joblib.load('player_value_model.pkl')
        le = joblib.load('label_encoder.pkl')

        # Encode position
        position_encoded = le.transform([data['position']])[0]

        # Prepare input features
        features = [[
            data['age'],
            data['goals'],
            data['assists'],
            data['appearances'],
            data['club_rank'],
            position_encoded
        ]]

        # Predict
        prediction = model.predict(features)[0]
        return jsonify({"predicted_value_million_eur": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ✅ IMPORTANT for Render to detect the open port
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
