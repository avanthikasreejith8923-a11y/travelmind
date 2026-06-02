import requests
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()

api_key = os.getenv("AVIATIONSTACK_API_KEY")

url = "http://api.aviationstack.com/v1/flights"
params = {
    "access_key": api_key,
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

query = "active flights"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=2
)

print("\nTest query results:")
for doc in results["documents"][0]:
    print(doc)
    print("---")