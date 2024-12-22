from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

# Create Flask app
app = Flask(__name__)

# Load models (ensure these are the correct paths to your .pkl files)
regression_model_path = "C:/Users/Lenovo/Downloads/Assignment2/best_wine_quality_regression_pipeline.pkl"
classification_model_path = "C:/Users/Lenovo/Downloads/Assignment2/best_wine_quality_classification_pipeline.pkl"

# Load pre-trained pipelines
reg_pipeline = joblib.load(regression_model_path)
clf_pipeline = joblib.load(classification_model_path)

# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Regression prediction route
@app.route("/predict_regression", methods=["POST"])
def predict_regression():
    try:
        # Parse input features from form and split into individual numbers
        features = [float(x.strip()) for x in request.form['features'].split(',')]
        features = np.array(features).reshape(1, -1)
        
        # Pass raw features into the pipeline
        prediction = reg_pipeline.predict(features)[0]
        return render_template(
            "index.html",
            regression_result=f"Predicted Quality (Regression): {prediction:.2f}"
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
        # Parse input features from form and split into individual numbers
        features = [float(x.strip()) for x in request.form['features'].split(',')]
        features = np.array(features).reshape(1, -1)
        
        # Pass raw features into the pipeline
        prediction = clf_pipeline.predict(features)[0]
        proba = clf_pipeline.predict_proba(features)[0, 1]
        return render_template(
            "index.html",
            classification_result=(f"Predicted Class: {int(prediction)}, "
                                   f"Probability of Class 1: {proba:.2f}")
        )
    except Exception as e:
        return render_template(
            "index.html",
            error=f"Error in Classification Prediction: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
