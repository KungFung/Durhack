from flask import Flask, render_template, request, redirect, url_for
from flask_socketio import SocketIO, join_room, emit
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import uuid

# --- App setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-me'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*")

USERS = ['User1', 'User2']


# --- Database Model ---
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    room = db.Column(db.String(50), nullable=False)
    sender = db.Column(db.String(50), nullable=False)
    receiver = db.Column(db.String(50), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.String(50), nullable=False)

    def as_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "message": self.message,
            "timestamp": self.timestamp
        }


with app.app_context():
    db.create_all()


# --- Routes ---
@app.route('/')
def home():
    """Create a new chat room and redirect"""
    room_id = uuid.uuid4().hex[:8]
    return redirect(url_for('room', room_id=room_id))


@app.route('/r/<room_id>')
def room(room_id):
    """Render chat interface for this room"""
    messages = Message.query.filter_by(room=room_id).all()
    return render_template('index.html', users=USERS, messages=messages, room_id=room_id)


# --- Socket.IO events ---
@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    emit('status', {'msg': f"{data['user']} joined the room."}, room=room)


@socketio.on('send_message')
def handle_message(data):
    room = data['room']
    msg = Message(
        room=room,
        sender=data['sender'],
        receiver=data['receiver'],
        message=data['message'],
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    db.session.add(msg)
    db.session.commit()
    emit('new_message', msg.as_dict(), room=room)


@socketio.on('clear_room')
def clear_room(data):
    room = data['room']
    Message.query.filter_by(room=room).delete()
    db.session.commit()
    emit('cleared', room=room)


# --- Main ---
if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
