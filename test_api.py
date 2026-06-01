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
    "limit": 3
}

response = requests.get(url, params=params)
data = response.json()

for flight in data["data"]:
    print("Flight:", flight["flight"]["iata"])
    print("Departure:", flight["departure"]["scheduled"])
    print("Arrival:", flight["arrival"]["scheduled"])
    print("Status:", flight["flight_status"])
    print("---")