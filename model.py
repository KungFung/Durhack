import nltk
import pandas as pd
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def train_model():
    file_path = 'data3.csv'
    df = pd.read_csv(file_path)
    df['Response Time seconds'] = df['Response Time'] * 60 * 60

    # Drop the original 'Hours' column if you no longer need it
    df = df.drop(columns=['Response Time'])
    features = [
        'Response Time seconds',
        'Sentiment',
    ]
    X = df[features]
    y = df['Relationship Duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Download the VADER lexicon (needed once)
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

df = pd.read_json('messages.json')

# Option 2: Read with json module first (safer for lists of dicts)
with open("messages.json", 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

print(df)

message = "I killed your dog!"

score = sia.polarity_scores(df['message'][0])
print(score)

relationship_score = 0
total_response_time = 0
message_count = 0

temp_prev_time = datetime.strptime(df['timestamp'][0], "%Y-%m-%d %H:%M:%S") 
temp_prev_time = temp_prev_time.replace(tzinfo=timezone.utc)
prev_time = int(temp_prev_time.timestamp())

for index, row in df.iterrows():
    message_count += 1
    relationship_score += sia.polarity_scores(df['message'][index])["compound"]
    
    dt = datetime.strptime(df['timestamp'][index], "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)

    message_time = int(dt.timestamp())


    total_response_time += message_time - prev_time

    prev_time = message_time

with open("results.txt", "w") as f:
    f.write(str(total_response_time/message_count) + "\n")
    f.write(str(relationship_score / message_count))

print("Average Response Time: ")
response = 1500

sentiment =0.19

model = train_model()
new_data_point = np.array([[response, sentiment]])
predict = model.predict(new_data_point)
print(predict)


def train_model():
    file_path = 'data3.csv'
    df = pd.read_csv(file_path)
    df['Response Time seconds'] = df['Response Time'] * 60 * 60

    # Drop the original 'Hours' column if you no longer need it
    df = df.drop(columns=['Response Time'])
    features = [
        'Response Time seconds',
        'Sentiment',
    ]
    X = df[features]
    y = df['Relationship Duration']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def get_duration():

    # Download the VADER lexicon (needed once)
    nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()

    df = pd.read_json('messages.json')

    # Option 2: Read with json module first (safer for lists of dicts)
    with open("messages.json", 'r') as f:
        data = json.load(f)

    df = pd.DataFrame(data)



    message = "I killed your dog!"

    score = sia.polarity_scores(df['message'][0])
    print(score)

    relationship_score = 0
    total_response_time = 0
    message_count = 0

    temp_prev_time = datetime.strptime(df['timestamp'][0], "%Y-%m-%d %H:%M:%S")
    temp_prev_time = temp_prev_time.replace(tzinfo=timezone.utc)
    prev_time = int(temp_prev_time.timestamp())

    for index, row in df.iterrows():
        message_count += 1
        relationship_score += sia.polarity_scores(df['message'][index])["compound"]

        dt = datetime.strptime(df['timestamp'][index], "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)

        message_time = int(dt.timestamp())

        total_response_time += message_time - prev_time

        prev_time = message_time

    with open("results.txt", "w") as f:
        f.write(str(total_response_time / message_count) + "\n")
        f.write(str(relationship_score / message_count))


    response = 1500

    sentiment = 0.19

    model = train_model()
    new_data_point = np.array([[response, sentiment]])
    predict = model.predict(new_data_point)
    return predict
