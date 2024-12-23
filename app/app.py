from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Load models (ensure these are the correct paths to your .pkl files)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
regression_model_path = os.path.join(BASE_DIR, "best_wine_quality_regression_pipeline.pkl")
classification_model_path = os.path.join(BASE_DIR, "best_wine_quality_classification_pipeline.pkl")

# Load pre-trained pipelines
try:
    reg_pipeline = joblib.load(regression_model_path)
    clf_pipeline = joblib.load(classification_model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Regression prediction route
@app.route("/predict_regression", methods=["POST"])
def predict_regression():
    try:
        raw_features = request.form.get('features', '').strip()
        if not raw_features:
            raise ValueError("No input provided. Please enter 11 comma-separated values.")

        features = [x.strip() for x in raw_features.split(',')]
        if len(features) != 11:
            raise ValueError("Invalid number of features. Please enter exactly 11 comma-separated values.")

        # Convert features to numeric values
        features = [float(x) for x in features]
        features = np.array(features).reshape(1, -1)

        # Predict using the regression pipeline
        prediction = reg_pipeline.predict(features)[0]

        return render_template(
            "index.html",
            regression_result=f"Predicted Quality (Regression): {prediction:.2f}"
        )
    except ValueError as ve:
        return render_template(
            "index.html",
            error=f"Input Error: {str(ve)}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Error in Regression Prediction: {str(e)}"
        )

# Classification prediction route
@app.route("/predict_classification", methods=["POST"])
def predict_classification():
    try:
        raw_features = request.form.get('features', '').strip()
        if not raw_features:
            raise ValueError("No input provided. Please enter 11 comma-separated values.")

        features = [x.strip() for x in raw_features.split(',')]
        if len(features) != 11:
            raise ValueError("Invalid number of features. Please enter exactly 11 comma-separated values.")

        # Convert features to numeric values
        features = [float(x) for x in features]
        features = np.array(features).reshape(1, -1)

        # Predict using the classification pipeline
        prediction = clf_pipeline.predict(features)[0]
        proba = clf_pipeline.predict_proba(features)[0, 1]

        return render_template(
            "index.html",
            classification_result=(f"Predicted Class: {int(prediction)}, "
                                   f"Probability of Class 1: {proba:.2f}")
        )
    except ValueError as ve:
        return render_template(
            "index.html",
            error=f"Input Error: {str(ve)}"
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Error in Classification Prediction: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
