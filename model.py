import nltk
import pandas as pd
import json
from nltk.sentiment import SentimentIntensityAnalyzer
from datetime import datetime, timezone

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

print("Average Response Time: ")
print(total_response_time/message_count)

print(relationship_score)