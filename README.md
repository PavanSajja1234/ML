Repository Structure:
/models: Contains the saved machine learning models.

/src: Source code for the machine learning algorithms and Flask application.

/templates: HTML templates for the Flask web interface.

app.py: Flask application main file.

requirements.txt: Required Python libraries.


Features:
Utilizes advanced machine learning algorithms for high accuracy in diabetes prediction.
Employs a 10-fold cross-validation to enhance model reliability.
Implements Voting and Stacking classifiers to integrate multiple model predictions, achieving up to 95% accuracy.
Technologies Used:
Python for implementing machine learning algorithms.
Libraries such as Scikit-learn for model building and evaluation.
Flask framework for creating a web interface to interact with the model.

Dataset:
The model is trained on a comprehensive dataset that includes various diabetic indicators from patients, sourced from public medical records.

How to Use:
Install necessary Python libraries as listed in requirements.txt.
Run the Flask application to start the web interface.
Input patient data through the web form to receive diabetes predictions.
