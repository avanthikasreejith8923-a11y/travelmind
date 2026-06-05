import streamlit as st
import requests
import os
import random
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import plotly.express as px
import pandas as pd
from database import init_db, save_flight, get_price_history, get_prices_by_day, get_avg_price

load_dotenv()

base_prices = {
    ("COK", "DEL"): 4000, ("BLR", "DEL"): 4500,
    ("BOM", "DEL"): 3500, ("MAA", "DEL"): 5000,
    ("COK", "BOM"): 3000, ("BLR", "HKG"): 18000,
    ("COK", "DXB"): 8000, ("BLR", "SIN"): 12000,
    ("DEL", "LHR"): 45000, ("BOM", "JFK"): 55000,
    ("COK", "SIN"): 11000, ("COK", "KUL"): 9000,
    ("DEL", "DXB"): 12000, ("BOM", "DXB"): 10000,
    ("DEL", "SIN"): 15000, ("MAA", "SIN"): 13000,
    ("DEL", "NRT"): 40000, ("BOM", "LHR"): 50000,
    ("DEL", "JFK"): 60000, ("BLR", "DXB"): 11000,
    ("HYD", "DEL"): 4000, ("CCU", "DEL"): 3500,
    ("GOI", "BOM"): 3000, ("COK", "DOH"): 9000,
    ("DEL", "BKK"): 18000, ("BOM", "BKK"): 16000,
}

airports = {
    "Kochi (COK)": "COK",
    "Delhi (DEL)": "DEL",
    "Mumbai (BOM)": "BOM",
    "Bangalore (BLR)": "BLR",
    "Chennai (MAA)": "MAA",
    "Hyderabad (HYD)": "HYD",
    "Kolkata (CCU)": "CCU",
    "Kannur (CNN)": "CNN",
    "Calicut (CCJ)": "CCJ",
    "Trivandrum (TRV)": "TRV",
    "Goa (GOI)": "GOI",
    "Pune (PNQ)": "PNQ",
    "Ahmedabad (AMD)": "AMD",
    "Jaipur (JAI)": "JAI",
    "Lucknow (LKO)": "LKO",
    "Dubai (DXB)": "DXB",
    "Abu Dhabi (AUH)": "AUH",
    "Doha (DOH)": "DOH",
    "Singapore (SIN)": "SIN",
    "Kuala Lumpur (KUL)": "KUL",
    "London Heathrow (LHR)": "LHR",
    "New York (JFK)": "JFK",
    "Paris (CDG)": "CDG",
    "Tokyo (NRT)": "NRT",
    "Sydney (SYD)": "SYD",
    "Toronto (YYZ)": "YYZ",
    "Frankfurt (FRA)": "FRA",
    "Amsterdam (AMS)": "AMS",
    "Bangkok (BKK)": "BKK",
    "Hong Kong (HKG)": "HKG",
    "Colombo (CMB)": "CMB",
    "Kathmandu (KTM)": "KTM",
    "Muscat (MCT)": "MCT",
    "Riyadh (RUH)": "RUH",
    "Bahrain (BAH)": "BAH",
    "Kuwait (KWI)": "KWI",
}

def get_price(departure, arrival):
    base = base_prices.get((departure, arrival),
           base_prices.get((arrival, departure), 10000))
    return round(random.uniform(base * 0.9, base * 1.2))

def get_weather(city):
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data["cod"] == 200:
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        humidity = data["main"]["humidity"]
        return f"🌤️ {city}: {temp}°C, {desc}, humidity {humidity}%"
    return "Weather data unavailable"

def get_destination_info(city):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""Give me a short travel guide for {city} with exactly this format:

🏛️ Top 3 Places to Visit:
1. 
2. 
3. 

🏨 Top 3 Hotels:
1. 
2. 
3. 

🍜 Top 3 Foods to Try:
1. 
2. 
3. 

Keep it short and helpful."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

st.set_page_config(page_title="TravelMind", page_icon="✈️", layout="wide")
init_db()

st.sidebar.title("✈️ TravelMind")
page = st.sidebar.radio("Navigate", ["🔍 Flight Search", "👥 Group Planner"])

if page == "🔍 Flight Search":

    st.title("✈️ TravelMind")
    st.subheader("Your AI-powered travel assistant")

    col1, col2 = st.columns(2)
    with col1:
        departure_city = st.selectbox("From", list(airports.keys()), index=0)
        departure = airports[departure_city]
    with col2:
        arrival_city = st.selectbox("To", list(airports.keys()), index=1)
        arrival = airports[arrival_city]

    question = st.text_input("Ask anything", placeholder="Which flights are available today?")

    if st.button("Search"):
        with st.spinner("Fetching flights..."):

            url = "http://api.aviationstack.com/v1/flights"
            params = {
                "access_key": os.getenv("AVIATIONSTACK_API_KEY"),
                "dep_iata": departure,
                "arr_iata": arrival,
                "limit": 5
            }
            response = requests.get(url, params=params)
            data = response.json()

            flight_texts = []
            flight_ids = []

            for i, flight in enumerate(data["data"]):
                price = get_price(departure, arrival)
                text = f"""
Flight {flight["flight"]["iata"]} from {departure} to {arrival}.
Departure: {flight["departure"]["scheduled"]}.
Arrival: {flight["arrival"]["scheduled"]}.
Status: {flight["flight_status"]}.
Airline: {flight["airline"]["name"]}.
Estimated Price: ₹{price}.
"""
                flight_texts.append(text)
                flight_ids.append(f"flight_{i}")
                save_flight(
                    flight["flight"]["iata"],
                    departure,
                    arrival,
                    price,
                    flight["airline"]["name"],
                    flight["flight_status"]
                )

            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = model.encode(flight_texts).tolist()

            client = chromadb.Client()
            collection = client.get_or_create_collection("flights")
            collection.add(
                documents=flight_texts,
                embeddings=embeddings,
                ids=flight_ids
            )

            query_embedding = model.encode([question]).tolist()
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3
            )

            retrieved_flights = "\n".join(results["documents"][0])

            prompt = f"""You are a helpful travel assistant.
Based on the following real flight data, answer the user's question clearly.

Flight data:
{retrieved_flights}

User question: {question}

Give a clear, helpful answer."""

            groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            chat_response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = chat_response.choices[0].message.content
            city_name = arrival_city.split("(")[0].strip()
            weather = get_weather(city_name)

            st.success("✅ Flights found!")

            st.subheader("🛫 Available Flights")
            for flight in flight_texts:
                st.info(flight)

            st.subheader("🤖 TravelMind Answer")
            st.write(answer)

            st.subheader("🌤️ Weather at Destination")
            st.info(weather)

            st.subheader("🔮 Price Prediction")
            avg_price = get_avg_price(departure, arrival)
            current_price = get_price(departure, arrival)
            if avg_price:
                if current_price < avg_price:
                    st.success(f"✅ Good time to book! Current ₹{current_price} is below average ₹{round(avg_price)}")
                else:
                    st.warning(f"⚠️ Prices are high! Current ₹{current_price} is above average ₹{round(avg_price)}. Consider waiting.")
            else:
                st.info("Not enough data for prediction yet!")

            st.subheader("📅 Best Day to Fly")
            day_data = get_prices_by_day(departure, arrival)
            if day_data:
                days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
                df_days = pd.DataFrame(day_data, columns=["Day", "Avg Price (₹)"])
                df_days["Day"] = df_days["Day"].apply(lambda x: days[int(x)])
                fig2 = px.bar(df_days, x="Day", y="Avg Price (₹)",
                              title="Cheapest Days to Fly",
                              color="Avg Price (₹)",
                              color_continuous_scale="RdYlGn_r")
                st.plotly_chart(fig2)
            else:
                st.info("Search more times to build heatmap data!")

            st.subheader("📈 Price Trend")
            history = get_price_history(departure, arrival)
            if history:
                df = pd.DataFrame(history, columns=["Date", "Avg Price (₹)"])
                fig = px.line(df, x="Date", y="Avg Price (₹)", title=f"Price Trend: {departure} → {arrival}")
                st.plotly_chart(fig)
            else:
                st.info("Search more times to build price history!")

            st.subheader("🗺️ Destination Guide")
            destination_info = get_destination_info(city_name)
            st.write(destination_info)

elif page == "👥 Group Planner":

    st.title("👥 Group Travel Planner")
    st.subheader("Find the cheapest flights for your whole group")

    destination_city = st.selectbox("Where is everyone going?", list(airports.keys()), index=1)
    destination = airports[destination_city]

    num_people = st.number_input("How many people?", min_value=2, max_value=10, value=3)

    origins = []
    for i in range(int(num_people)):
        person_city = st.selectbox(f"Person {i+1} flying from:", list(airports.keys()), key=f"person_{i}")
        origins.append((person_city, airports[person_city]))

    if st.button("Find Cheapest Group Flights"):
        st.subheader("💰 Group Flight Summary")
        total_cost = 0
        results_list = []

        for person_city, person_iata in origins:
            price = get_price(person_iata, destination)
            total_cost += price
            results_list.append((person_city, destination_city, price))
            st.info(f"✈️ {person_city} → {destination_city}: ₹{price}")

        st.success(f"💰 Total group cost: ₹{total_cost}")
        avg_per_person = round(total_cost / num_people)
        st.info(f"👤 Average per person: ₹{avg_per_person}")

        cheapest = min(results_list, key=lambda x: x[2])
        st.success(f"🏆 Cheapest route: {cheapest[0]} → {cheapest[1]} at ₹{cheapest[2]}")

        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        summary = "\n".join([f"Person from {r[0]}: ₹{r[2]}" for r in results_list])
        prompt = f"""A group of {num_people} people are traveling to {destination_city}.
Here are their flights and costs:
{summary}
Total cost: ₹{total_cost}
Average per person: ₹{avg_per_person}

Give a short helpful travel tip for this group."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        st.subheader("🤖 AI Travel Tip for Your Group")
        st.write(response.choices[0].message.content)