# %%writefile app.py
from flask import Flask, request, jsonify
import joblib

# Load saved model
model = joblib.load("sentiment_model.pkl")

app = Flask(__name__)

# Health check route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Sentiment Analysis API is running "})

@app.route("/predict", methods=["POST"])
def predict_sentiment():
 
        data = request.get_json()

        tweet = data["tweet"]
        pred = model.predict([tweet])[0]

        return jsonify({
            "tweet": tweet,
            "predicted_sentiment": str(pred)
        })




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
