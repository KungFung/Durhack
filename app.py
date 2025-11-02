from flask import Flask, render_template, request, redirect, url_for, session
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
import Firebase
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
app = Flask(__name__)
MESSAGE_FILE = 'messages.json'
DEFAULT_USERS = ['User1', 'User2']

app.secret_key = 'super_secret_chat_key_for_testing'


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
    """
    Retrieves all messages from the Firestore 'Messages' collection
    and returns them as a list of Python dictionaries (JSON-like structure).

    Returns:
        list: A list of message dictionaries, or an empty list if an error occurs.
    """
    try:
        chat_key = session.get('chat_key')
        # 1. Call the function that retrieves and formats data from Firestore
        all_messages = Firebase.load_as_json(chat_key)

        # 2. Return the list directly (it's already in the desired dictionary format)
        return all_messages

    except Exception as e:
        # Handle any potential Firestore connection or query errors
        print(f"❌ Error loading messages from Firestore: {e}")
        # In case of failure, return an empty list or handle as appropriate for your Flask app
        return []


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

def get_vader_compound_score(text):
    analyzer = SentimentIntensityAnalyzer()

    if isinstance(text, str):
        # Only call VADER if the input is definitely a string
        vs = analyzer.polarity_scores(text)
        return vs['compound']
    else:
        # Return a neutral score (or NaN) for missing/non-text data
        return 0.0 # You could also use np.nan instead of 0.0

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

        data = load_messages()

        df = pd.DataFrame(data)
        if df.empty:
            return 0

        df['time'] = pd.to_numeric(df['time'], errors='coerce').fillna(0)

        # Step 2: Calculate the total sum
        total_time = df['time'].sum()
        df['sentiment'] = df['message'].apply(get_vader_compound_score)
        total_sentiment = df['sentiment'].sum()
        len(df)


        response = total_time / len(df)
        response = int(response)

        sentiment = total_sentiment / len(df)

        model = train_model()
        new_data_point = np.array([[response, sentiment]])
        predict = model.predict(new_data_point)
        return predict


@app.route('/', methods=['GET', 'POST'])
def login():
    """Handles the login form submission to set up the chat key."""
    if request.method == 'POST':
        # 1. Get user inputs
        current_user = request.form['current_user'].strip().lower()
        target_user = request.form['target_user'].strip().lower()

        if not current_user or not target_user:
            return render_template('login.html', error="Both usernames are required.")

        # 2. Create the unique chat key
        # Sorting ensures that ('alice', 'bob') and ('bob', 'alice') result in the same key: 'alice_bob'
        sorted_users = sorted([current_user, target_user])
        chat_key = f"{sorted_users[0]}_{sorted_users[1]}"

        # 3. Store the keys in the session
        session['current_user'] = current_user
        session['target_user'] = target_user
        session['chat_key'] = chat_key

        print(f"Generated Chat Key: {chat_key}")

        # 4. Redirect to the chat page
        return redirect(url_for('chat_room'))

    # Display the form on GET request
    return render_template('login.html')


@app.route('/chat', methods=['GET', 'POST'])
def chat_room():
    """Displays the main chat page and handles message submission."""
    current_user = session.get('current_user')
    target_user = session.get('target_user')  # FIX 1: Retrieve target_user
    chat_key = session.get('chat_key')

    if not chat_key:
        # If no key is set, redirect back to login
        return redirect(url_for('login'))

    if request.method == 'POST':
        # Retrieve the message content from the form
        message = request.form.get('message', '').strip()
        time = request.form.get('time')

        if message:
            # Use current_user (from session) as sender, and chat_key (from session) to save
            # We pass None for time to use the Firestore SERVER_TIMESTAMP
            Firebase.save_message(current_user, message, time, chat_key)

        # Redirect (POST-redirect-GET pattern) to prevent double submissions
        return redirect(url_for('chat_room'))

    # GET request logic: Display chat history
    messages = load_messages()
    predicted_duration = get_duration()

    # FIX 2: Create the 'users' list as expected by the chat.html template
    users = [current_user, target_user]

    # We remove the usage of undeclared functions like get_users(), get_duration(), and _msg_timestamp_to_epoch
    # as messages are already sorted by 'time' in get_all_messages_as_json_list().

    return render_template('chat.html',
                           current_user=current_user,
                           chat_key=chat_key,
                           messages=messages,
                           users=users,
                           predicted_duration=predicted_duration)  # FIX 3: Pass the 'users' list to the template
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
