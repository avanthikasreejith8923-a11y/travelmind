import requests
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

aviationstack_key = os.getenv("AVIATIONSTACK_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

url = "http://api.aviationstack.com/v1/flights"
params = {
    "access_key": aviationstack_key,
    "dep_iata": "COK",
    "arr_iata": "DEL",
    "limit": 5
}

response = requests.get(url, params=params)
data = response.json()

flight_texts = []
flight_ids = []

for i, flight in enumerate(data["data"]):
    text = f"""
Flight {flight["flight"]["iata"]} from Kochi (COK) to Delhi (DEL).
Departure: {flight["departure"]["scheduled"]}.
Arrival: {flight["arrival"]["scheduled"]}.
Status: {flight["flight_status"]}.
Airline: {flight["airline"]["name"]}.
"""
    flight_texts.append(text)
    flight_ids.append(f"flight_{i}")

print("Embedding flights...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(flight_texts).tolist()

client = chromadb.Client()
collection = client.create_collection("flights")

collection.add(
    documents=flight_texts,
    embeddings=embeddings,
    ids=flight_ids
)

print(f"Stored {collection.count()} flights in ChromaDB!")

user_question = "Which flights are active right now from Kochi to Delhi?"

query_embedding = model.encode([user_question]).tolist()
results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

retrieved_flights = "\n".join(results["documents"][0])

prompt = f"""You are a helpful travel assistant.
Based on the following real flight data, answer the user's question.

Flight data:
{retrieved_flights}

User question: {user_question}

Give a clear, helpful answer."""

groq_client = Groq(api_key=groq_key)

chat_response = groq_client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

print("\nTravelMind Answer:")
print(chat_response.choices[0].message.content)