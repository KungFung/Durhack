from flask import Flask, render_template, request, redirect, url_for
import json
from datetime import datetime
import os
import nltk
import pandas as pd
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import numpy as np

app = Flask(__name__)
MESSAGE_FILE = 'messages.json'
USERS = ['User1', 'User2']

# Ensure message file exists
if not os.path.exists(MESSAGE_FILE):
    with open(MESSAGE_FILE, 'w') as f:
        json.dump([], f)

def load_messages():
    with open(MESSAGE_FILE, 'r') as f:
        return json.load(f)

def save_message(sender, receiver, message):
    messages = load_messages()
    messages.append({
        "sender": sender,
        "receiver": receiver,
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    with open(MESSAGE_FILE, 'w') as f:
        json.dump(messages, f, indent=4)

def clear_messages():
    with open(MESSAGE_FILE, 'w') as f:
        json.dump([], f, indent=4)

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




        # Option 2: Read with json module first (safer for lists of dicts)
        with open("messages.json", 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        if df.empty:
            return 0




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


        response = total_response_time / message_count

        sentiment = relationship_score / message_count

        model = train_model()
        new_data_point = np.array([[response, sentiment]])
        predict = model.predict(new_data_point)
        return predict


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sender = request.form.get('sender')
        receiver = request.form.get('receiver')
        message = request.form.get('message', '').strip()
        if sender in USERS and receiver in USERS and sender != receiver and message:
            # 1. Message is SAVED to the file
            save_message(sender, receiver, message)

            # 2. Flask redirects, forcing the browser to send a new GET request
        return redirect(url_for('index'))
    predicted_duration = get_duration()
    messages = load_messages()
    return render_template('index.html', users=USERS, messages=messages, predicted_duration=predicted_duration)

@app.route('/clear', methods=['POST'])
def clear():
    clear_messages()
    return redirect(url_for('index'))

# Prevent favicon 404s in debug
@app.route('/favicon.ico')
def favicon():
    return ('', 204)

if __name__ == '__main__':
    app.run(debug=True)
