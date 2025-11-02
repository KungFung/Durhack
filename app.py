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
DEFAULT_USERS = ['User1', 'User2']


def _msg_timestamp_to_epoch(msg: dict) -> int:
    """Return an epoch seconds integer for a message dict's timestamp.

    Handles several common formats:
    - message['timestamp'] as formatted string '%Y-%m-%d %H:%M:%S'
    - numeric seconds or milliseconds in 'timestamp' or 'timestamp_ms'
    - ISO-like string via fromisoformat as a last attempt
    If parsing fails return 0 so message sorts early.
    """
    if not isinstance(msg, dict):
        return 0
    # Common keys
    ts = msg.get('timestamp')
    if ts is None:
        ts = msg.get('timestamp_ms') or msg.get('timestamp_s') or msg.get('ts')

    # Numeric
    if isinstance(ts, (int, float)):
        val = float(ts)
        # heuristic: milliseconds if large
        if abs(val) > 1e11:
            val = val / 1000.0
        try:
            return int(val)
        except Exception:
            return 0

    # String
    if isinstance(ts, str):
        s = ts.strip()
        # Try known formatted pattern
        try:
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            pass
        # Try ISO
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except Exception:
            pass
        # Try numeric string
        if s.isdigit():
            try:
                val = float(s)
                if abs(val) > 1e11:
                    val = val / 1000.0
                return int(val)
            except Exception:
                return 0

    return 0

def _write_json_file(path, data):
    # Write as UTF-8 and preserve Unicode characters
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# Ensure message file exists
if not os.path.exists(MESSAGE_FILE):
    _write_json_file(MESSAGE_FILE, [])

def load_messages():
    # Try reading as UTF-8 first; if it fails, try common Windows encodings then fallback
    try:
        with open(MESSAGE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # Try other encodings
        with open(MESSAGE_FILE, 'rb') as f:
            raw = f.read()
        for enc in ('utf-8', 'utf-8-sig', 'cp1252', 'latin-1'):
            try:
                text = raw.decode(enc)
                return json.loads(text)
            except Exception:
                continue
        # Last resort: decode with replacement to avoid crash
        text = raw.decode('utf-8', errors='replace')
        return json.loads(text)


def get_users():
    """Return a list of two users for the UI.

    Preference order:
    - If messages contain sender/receiver names, use the two unique names found.
    - Otherwise fall back to DEFAULT_USERS.
    """
    msgs = load_messages()
    if not msgs:
        return DEFAULT_USERS

    names = []
    for m in msgs:
        s = m.get('sender')
        r = m.get('receiver')
        if s and s not in names:
            names.append(s)
        if r and r not in names:
            names.append(r)
        if len(names) >= 2:
            break
    if len(names) >= 2:
        return names[:2]
    # fallback
    return DEFAULT_USERS

def save_message(sender, receiver, message, time):
    messages = load_messages()
    messages.append({
        "sender": sender,
        "receiver": receiver,
        "message": message,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "time": time
    })
    with open(MESSAGE_FILE, 'w') as f:
        json.dump(messages, f, indent=4)

def clear_messages():
    _write_json_file(MESSAGE_FILE, [])

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
        data = load_messages()

        df = pd.DataFrame(data)
        if df.empty:
            return 0

        df['time'] = pd.to_numeric(df['time'], errors='coerce').fillna(0)

        # Step 2: Calculate the total sum
        total_time = df['time'].sum()


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

        with open("results.txt", "w", encoding='utf-8') as f:
            f.write(str(total_response_time / message_count) + "\n")
            f.write(str(relationship_score / message_count))

        print(total_time)
        response = total_time / message_count
        response = int(response)

        sentiment = relationship_score / message_count

        model = train_model()
        new_data_point = np.array([[response, sentiment]])
        predict = model.predict(new_data_point)
        return predict


@app.route('/', methods=['GET', 'POST'])
def index():
    users = get_users()
    if request.method == 'POST':
        sender = request.form.get('sender')
        receiver = request.form.get('receiver')
        message = request.form.get('message', '').strip()
        time = request.form.get('time')
        if sender in users and receiver in users and sender != receiver and message:
            # 1. Message is SAVED to the file
            save_message(sender, receiver, message, time)

        # 2. Redirect (POST-redirect-GET)
        return redirect(url_for('index'))
    predicted_duration = get_duration()
    messages = load_messages()
    # Sort messages by timestamp ascending so conversation reads oldest -> newest
    try:
        messages = sorted(messages, key=_msg_timestamp_to_epoch)
    except Exception:
        # If sorting fails, leave original order
        pass
    return render_template('index.html', users=users, messages=messages, predicted_duration=predicted_duration)

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
