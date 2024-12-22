# Wine Quality Prediction

This project is a web application for predicting the quality of red wine based on various chemical properties.

## Features
- **Regression Model**: Predicts the quality score of the wine.
- **Classification Model**: Classifies the wine into one of the quality classes (e.g., high quality or low quality).

## Requirements

Before running the app, make sure you have the following Python packages installed:
- Flask
- joblib
- scikit-learn
- numpy
- pandas

You can install these dependencies by running:
```bash
pip install -r requirements.txt
```

## Setup Instructions

1. Clone the repository to your local machine:
```bash
git clone https://github.com/yourusername/repository-name.git
```

2. Navigate to the project folder:
```bash
cd repository-name
```

3. Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

4. Run the app with:
```bash
python app.py
```

5. Open your browser and go to `http://127.0.0.1:5000/` to start using the app.

## How to Use

1. Enter the chemical features of a wine into the input field. You need to input these values as a comma-separated list.
2. Click on either "Predict Regression" or "Predict Classification" to get the quality prediction.
3. The results will be displayed below the input form.

### Sample Input

For regression:
```bash
7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4
```

For classification:
```bash
7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4
```

### Sample Output

Regression:
```bash
Predicted Quality: 5.49
```

Classification:
```bash
Predicted Class: 1, Probability of Class 1: 0.78
```

## Troubleshooting

If you encounter any issues during setup or running the app, check the following:

- Ensure you have installed all dependencies via `pip install -r requirements.txt`
- Make sure your Python version is compatible with the libraries used
- If you run into a port issue, try changing the Flask app's default port

## Notes
- If you make any changes to the model or the app, make sure to retrain your model and update the pkl files before using the app again.
- If you're deploying to a cloud or other environment, ensure you update the app's configurations to work in that environment.
