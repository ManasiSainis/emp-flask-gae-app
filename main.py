import datetime

from flask import Flask, jsonify, render_template
from google.cloud import storage
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import joblib

app = Flask(__name__)


# https://towardsdatascience.com/creating-restful-apis-using-flask-and-python-655bad51b24
@app.route("/call-gcs")
def call_storage():
    client = storage.Client()
    bucket = client.get_bucket("ems-model-for-logistic-regression", timeout=None)
    blob = bucket.blob("opensearch-version-changes.txt")
    downloaded_blob = blob.download_as_string().decode("utf-8")
    print(downloaded_blob)
    return downloaded_blob


@app.route('/person')
def hello():
    return jsonify({'name': 'Jimit',
                    'address': 'India'})

model = joblib.load('model.pkl')

#Issue Reporting Mechanism
@app.route('/issuepredict', methods=['POST'])
def predict_endpoint():
    # Get data from request
    data = request.get_json(force=True)
    # Convert data to numpy array (if needed)
    input_data = np.array(data['input'])
    # Get predictions
    predictions = issuepredict(input_data)
    # Return predictions as JSON
    return jsonify(predictions.tolist())

@app.route('/check', methods=['GET'])
def Model_Server_Check():
    return "Model server is up and running!"

#Candidate Finder Random Forest Predictor
def main(extra_skill, salary):
    # Load the dataset
    ds = pd.read_csv('EmployeeDataBook.csv')

    # Data preprocessing and model training (same as in the provided code)

    # Filter data based on inputs
    filtered_data = ds[(ds['extra_skills'] == extra_skill) & (ds['salary'].astype(int) >= salary)]

    # Split data for Decision Trees
    X = filtered_data[['rating', 'salary']].astype(int)
    y_rating = filtered_data['rating'].astype(int)
    y_feedback = filtered_data['extra_skills']

    # Decision Tree 1: Rating
    X_train_rating, X_test_rating, y_train_rating, y_test_rating = train_test_split(X, y_rating, test_size=0.2, random_state=42)
    dt_rating = DecisionTreeClassifier()
    dt_rating.fit(X_train_rating, y_train_rating)

    # Decision Tree 2: Feedback
    X_train_feedback, X_test_feedback, y_train_feedback, y_test_feedback = train_test_split(X, y_feedback, test_size=0.2, random_state=42)
    dt_feedback = DecisionTreeClassifier()
    dt_feedback.fit(X_train_feedback, y_train_feedback)

    # Final Random Forest layer
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y_rating)

    # Split data into training and testing sets for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y_rating, test_size=0.2, random_state=42)

    # Evaluate the model on the testing set
    test_accuracy = accuracy_score(y_test, rf.predict(X_test))

    # Generate output
    output_data = pd.DataFrame()
    output_data['Emp_id'] = filtered_data['id']
    output_data['Full_name'] = filtered_data['full_name']
    output_data['Title'] = filtered_data['title']
    output_data['Salary'] = filtered_data['salary']
    output_data['Technical_skills'] = filtered_data['tech_skills']
    output_data['Extra_skills'] = dt_feedback.predict(X)
    output_data['Rating'] = dt_rating.predict(X)

    # Save the trained model
    joblib.dump(rf, 'random_forest_model.joblib')

    return output_data.head(10), test_accuracy

#Candidate Finder Mechanism
@app.route('/candidatefinder', methods=['POST'])
def candidatefinder():
    if request.method == 'POST':
        data = request.get_json()
        extra_skill = data['extra_skill']
        salary = float(data['salary'])
        candidates, test_accuracy = main(extra_skill, salary)
        return jsonify({"candidates": candidates.to_dict(orient='records'), "test_accuracy": test_accuracy})

@app.route("/")
def root():
    # For the sake of example, use static information to inflate the template.
    # This will be replaced with real information in later steps.
    dummy_times = [
        datetime.datetime(2018, 1, 1, 10, 0, 0),
        datetime.datetime(2018, 1, 2, 10, 30, 0),
        datetime.datetime(2018, 1, 3, 11, 0, 0),
    ]

    return render_template("index.html", times=dummy_times)

if __name__ == "__main__":
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host="127.0.0.1", port=8080, debug=True)
