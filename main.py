import datetime

from flask import Flask, jsonify, render_template
from google.cloud import storage

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
