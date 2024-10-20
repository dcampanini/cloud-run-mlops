# [START functions_cloudevent_storage]
from cloudevents.http import CloudEvent
import functions_framework
from google.cloud import storage

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import datetime
import pandas as pd

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def hello_gcs(cloud_event: CloudEvent) -> tuple:
    """This function is triggered by a change in a storage bucket.

    Args:
        cloud_event: The CloudEvent that triggered this function.
    Returns:
        The event ID, event type, bucket, name, metageneration, and timeCreated.
    """
    data = cloud_event.data

    event_id = cloud_event["id"]
    event_type = cloud_event["type"]

    bucket = data["bucket"]
    name = data["name"]
    metageneration = data["metageneration"]
    timeCreated = data["timeCreated"]
    updated = data["updated"]

    print(f"Event ID: {event_id}")
    print(f"Event type: {event_type}")
    print(f"Bucket: {bucket}")
    print(f"File: {name}")
    print(f"Metageneration: {metageneration}")
    print(f"Created: {timeCreated}")
    print(f"Updated: {updated}")

    # train model
    df = pd.read_csv('gs://mlops-2024/pima-indians-diabetes.csv')
    print(df.head())
    # split data into X and y
    X = df.iloc[:,0:8]
    Y = df.iloc[:,8]
    # split data into train and test sets
    seed = 7
    test_size = 0.33
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    # fit model no training data
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions for test data
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    #%% save model
    # Get the current date and time
    now = datetime.datetime.now()
    # Format the date and time as a string (e.g., "2024-10-13_14-30-00")
    formatted_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # save model
    storage_client = storage.Client()
    output_bucket = storage_client.bucket('mlops-models1')
    # Define the filename
    filename = f"xgb_{formatted_time}.pkl"
    blob = output_bucket.blob(filename)

    # save model to the bucket
    with blob.open(mode='wb') as f:
        pickle.dump(model, f)

    return event_id, event_type, bucket, name, metageneration, timeCreated, updated