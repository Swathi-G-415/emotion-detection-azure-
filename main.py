import requests
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load Azure credentials
load_dotenv()
key = os.getenv("AZURE_KEY")
endpoint = os.getenv("AZURE_ENDPOINT")
url = endpoint + "/text/analytics/v3.1/sentiment"

headers = {
    "Ocp-Apim-Subscription-Key": key,
    "Content-Type": "application/json"
}

# Input text samples
texts = [
    "I am very happy today!",
    "This is the worst day of my life.",
    "It's okay, I guess.",
    "I'm confused and unsure about my feelings."
]

documents = {"documents": [{"id": str(i+1), "language": "en", "text": t} for i, t in enumerate(texts)]}

response = requests.post(url, headers=headers, json=documents)
result = response.json()

# Print results
for doc in result['documents']:
    print(f"Text {doc['id']} → {doc['sentiment'].capitalize()}, Confidence: {doc['confidenceScores']}")

# Visualize
labels = ['positive', 'neutral', 'negative']
sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
for doc in result['documents']:
    sentiment_counts[doc['sentiment']] += 1

plt.pie(sentiment_counts.values(), labels=labels, autopct='%1.1f%%')
plt.title('Emotion Distribution')
plt.savefig("images/pie_chart.png")
plt.show()
