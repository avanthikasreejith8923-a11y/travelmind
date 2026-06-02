import requests
import os
from dotenv import load_dotenv

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

for flight in data["data"]:
    text = f"""
Flight {flight["flight"]["iata"]} from Kochi (COK) to Delhi (DEL).
Departure: {flight["departure"]["scheduled"]}.
Arrival: {flight["arrival"]["scheduled"]}.
Status: {flight["flight_status"]}.
Airline: {flight["airline"]["name"]}.
"""
    flight_texts.append(text)
    print(text)

print(f"Total flights formatted: {len(flight_texts)}")