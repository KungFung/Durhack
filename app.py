from flask import Flask, render_template, request, redirect, url_for
import json
from datetime import datetime
import os

app = Flask(__name__)
MESSAGE_FILE = 'messages.json'

# Fixed users
USERS = ['User1', 'User2']

# Ensure JSON file exists
if not os.path.exists(MESSAGE_FILE):
    with open(MESSAGE_FILE, 'w') as f:
        json.dump([], f)

def load_messages():
    with open(MESSAGE_FILE, 'r') as f:
        return json.load(f)

def save_message(sender, message):
    messages = load_messages()
    # Automatically set receiver as the other user
    receiver = USERS[1] if sender == USERS[0] else USERS[0]
    messages.append({
        'sender': sender,
        'receiver': receiver,
        'message': message,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    with open(MESSAGE_FILE, 'w') as f:
        json.dump(messages, f, indent=4)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sender = request.form.get('sender')
        message = request.form.get('message')
        if sender in USERS and message:
            save_message(sender, message)
        return redirect(url_for('index'))

    messages = load_messages()
    return render_template('index.html', messages=messages, users=USERS)

if __name__ == '__main__':
    app.run(debug=True)
