import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from datetime import datetime, timezone

# --- IMPORTANT: Replace 'path/to/your/firebase-key.json' with your file's path ---
# 1. Load the downloaded service account key file
cred = credentials.Certificate("durhack-db9fe-firebase-adminsdk-fbsvc-e460cc5e7e.json")

# 2. Initialize the Firebase app using the credentials
# The app initialization connects to the project specified within the key file.
try:
    if not firebase_admin._apps:  # Prevents re-initialization error
        firebase_admin.initialize_app(cred)

    # 3. Get the Firestore client instance
    db = firestore.client()
    print("✅ Firestore client initialized using Service Account Key.")
except Exception as e:
    print(f"❌ Error initializing Firebase app: {e}")
    db = None


# The save_message function remains the same
def save_message(sender, message, time, chat_key):
    if db is None:
        print("Cannot save message: Firestore client is not initialized.")
        return None

    try:
        collection_path = f"Chats/{chat_key}/Messages"

        messages_ref = db.collection(collection_path)
        update_time, doc_ref = messages_ref.add({
            "user": sender,
            "message": message,
            "time": time,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        print(f"Document added. ID: {doc_ref.id}")
        return doc_ref.id
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


from google.cloud import firestore


# Assume 'db' is your initialized firestore.client() instance

def load_as_json(chat_key):
    if db is None:
        return []

    try:
        collection_path = f"Chats/{chat_key}/Messages"
        messages_ref = db.collection(collection_path)

        # Sort by 'time' to keep messages in order
        docs = messages_ref.order_by('time').stream()

        message_list = []
        for doc in docs:
            data = doc.to_dict()
            time_value = data.get("time")

            # Convert Firestore Timestamp to Unix timestamp (milliseconds) for JSON
            if isinstance(time_value, datetime):
                # Convert to milliseconds for common chat standards
                data["time"] = int(time_value.timestamp() * 1000)
            elif time_value is not None:
                data["time"] = int(time_value)

            message_list.append(data)

        return message_list
    except Exception as e:
        print(f"❌ Error retrieving messages: {e}")
        return []

print(load_as_json("daniel_evie"))
