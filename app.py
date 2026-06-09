import streamlit as st
import requests
import os
import random
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import plotly.express as px
import pandas as pd
from database import init_db, save_flight, get_price_history, get_prices_by_day, get_avg_price


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
        return f"{city}: {temp}C, {desc}, Humidity {humidity}%"
    return "Weather data unavailable"

def get_destination_info(city):
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    prompt = f"""Give me a short travel guide for {city} with exactly this format:

Top 3 Places to Visit:
1. 
2. 
3. 

Top 3 Hotels:
1. 
2. 
3. 

Top 3 Foods to Try:
1. 
2. 
3. 

Keep it short and helpful."""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def add_style():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Orbitron:wght@700;900&display=swap');

    * { font-family: 'Inter', sans-serif; box-sizing: border-box; }

    .stApp { background-color: #ffffff !important; }

    .block-container {
        padding: 2rem 4rem !important;
        background: #ffffff !important;
        max-width: 1100px !important;
    }

    .logo { text-align: center; padding: 10px 0 4px 0; }

    .logo-text {
        font-family: 'Orbitron', sans-serif !important;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: 3px;
    }

    .logo-travel { color: #000000; }
    .logo-mind { color: #76b900; }

    .logo-bar {
        width: 80px;
        height: 3px;
        background: #76b900;
        margin: 6px auto 0 auto;
        border-radius: 2px;
    }

    .tagline {
        text-align: center;
        color: #666666 !important;
        font-size: 0.95rem;
        font-weight: 400;
        margin: 8px 0 32px 0;
        letter-spacing: 0.5px;
    }

    .section-label {
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #000000 !important;
        margin-bottom: 6px;
        padding-left: 2px;
    }

    .section-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #000000 !important;
        border-left: 3px solid #76b900;
        padding-left: 10px;
        margin: 28px 0 14px 0;
    }

    .card {
        background: #f7f7f7;
        border: 1px solid #e4e4e4;
        border-left: 4px solid #76b900;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 12px;
        color: #111111 !important;
    }

    .card p {
        color: #111111 !important;
        margin: 4px 0;
        font-size: 14px;
        line-height: 1.8;
    }

    .card .price {
        color: #76b900 !important;
        font-weight: 700;
        font-size: 20px;
        margin-top: 8px !important;
    }

    .card .label {
        color: #555555 !important;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .card-success {
        background: #f4fae8;
        border: 1px solid #d4edaa;
        border-left: 4px solid #76b900;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    .card-success p {
        color: #2a4a00 !important;
        margin: 0;
        font-weight: 600;
        font-size: 14px;
    }

    .card-warning {
        background: #fff8e6;
        border: 1px solid #ffd980;
        border-left: 4px solid #ffaa00;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    .card-warning p {
        color: #7a4a00 !important;
        margin: 0;
        font-weight: 600;
        font-size: 14px;
    }

    .divider {
        border: none;
        border-top: 1px solid #eeeeee;
        margin: 24px 0;
    }

    .stButton > button {
        background: #76b900 !important;
        color: #000000 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 12px 36px !important;
        font-size: 13px !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stButton > button:hover {
        background: #000000 !important;
        color: #76b900 !important;
    }

    .stSelectbox > div > div {
        background: #f7f7f7 !important;
        border: 1.5px solid #dddddd !important;
        border-radius: 4px !important;
        color: #111111 !important;
    }

    .stSelectbox > div > div > div {
        color: #111111 !important;
    }

    .stTextInput > div > div > input {
        background: #f7f7f7 !important;
        border: 1.5px solid #dddddd !important;
        border-radius: 4px !important;
        color: #111111 !important;
        padding: 10px 14px !important;
        font-size: 14px !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #aaaaaa !important;
    }

    .stTextInput > div > div > input:focus {
        border: 1.5px solid #76b900 !important;
        background: #ffffff !important;
        box-shadow: 0 0 0 3px rgba(118,185,0,0.08) !important;
    }

    .stNumberInput > div > div > input {
        background: #f7f7f7 !important;
        border: 1.5px solid #dddddd !important;
        border-radius: 4px !important;
        color: #111111 !important;
    }

    div[data-testid="stInfo"],
    div[data-testid="stSuccess"],
    div[data-testid="stWarning"],
    div[data-testid="stAlert"] {
        background: #f7f7f7 !important;
        border: 1px solid #e4e4e4 !important;
        border-left: 4px solid #76b900 !important;
        border-radius: 6px !important;
        color: #111111 !important;
        padding: 14px 18px !important;
    }

    div[data-testid="stWarning"] {
        border-left: 4px solid #ffaa00 !important;
        background: #fff8e6 !important;
    }

    section[data-testid="stSidebar"] {
        background: #000000 !important;
        border-right: 3px solid #76b900 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    .stRadio label {
        color: #ffffff !important;
        font-size: 14px !important;
        font-weight: 500 !important;
    }

    .stSpinner > div { border-color: #76b900 !important; }

    label { color: #111111 !important; }
    p { color: #111111 !important; }

    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(page_title="TravelMind", page_icon="", layout="wide")
init_db()
add_style()

st.sidebar.markdown("""
<div style='font-family:Orbitron,sans-serif;font-size:1.2rem;font-weight:900;
letter-spacing:2px;color:#76b900;padding:10px 0 4px 0'>TRAVELMIND</div>
<div style='font-size:11px;color:#888888;letter-spacing:0.5px;margin-bottom:20px'>
AI-powered travel assistant</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")
page = st.sidebar.radio("", ["Flight Search", "Group Planner"])

if page == "Flight Search":

    st.markdown("""
    <div class='logo'>
        <div class='logo-text'>
            <span class='logo-travel'>TRAVEL</span><span class='logo-mind'>MIND</span>
        </div>
        <div class='logo-bar'></div>
    </div>
    <p class='tagline'>AI-powered flight search — real-time data, smart analytics, instant answers</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-label'>From</div>", unsafe_allow_html=True)
        departure_city = st.selectbox("", list(airports.keys()), index=0, key="dep")
        departure = airports[departure_city]
    with col2:
        st.markdown("<div class='section-label'>To</div>", unsafe_allow_html=True)
        arrival_city = st.selectbox("", list(airports.keys()), index=1, key="arr")
        arrival = airports[arrival_city]

    st.markdown("<div class='section-label'>Ask anything</div>", unsafe_allow_html=True)
    question = st.text_input("", placeholder="e.g. Which flights are cheapest this weekend?", key="question")

    search = st.button("Search Flights", key="search_btn")

    if search:
        with st.spinner("Fetching real-time flight data..."):

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
            flight_prices = []

            for i, flight in enumerate(data["data"]):
                price = get_price(departure, arrival)
                flight_prices.append(price)
                text = f"Flight {flight['flight']['iata']} from {departure} to {arrival}. Departure: {flight['departure']['scheduled']}. Arrival: {flight['arrival']['scheduled']}. Status: {flight['flight_status']}. Airline: {flight['airline']['name']}. Estimated Price: Rs{price}."
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
            try:
                client.delete_collection("flights")
            except:
                pass
            collection = client.create_collection("flights")
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

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Available Flights</div>", unsafe_allow_html=True)

            for i, flight in enumerate(data["data"]):
                dep_time = flight['departure']['scheduled'][11:16] if flight['departure']['scheduled'] else "N/A"
                arr_time = flight['arrival']['scheduled'][11:16] if flight['arrival']['scheduled'] else "N/A"
                st.markdown(f"""
                <div class='card'>
                    <p><strong>{flight['flight']['iata']}</strong> &nbsp;&nbsp; {flight['airline']['name']}</p>
                    <p class='label'>{departure} → {arrival} &nbsp;|&nbsp; Departs {dep_time} &nbsp;|&nbsp; Arrives {arr_time} &nbsp;|&nbsp; {flight['flight_status'].upper()}</p>
                    <p class='price'>Rs {flight_prices[i]:,}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>AI Answer</div>", unsafe_allow_html=True)
            formatted_answer = answer.replace("\n", "<br>")
            st.markdown(f"<div class='card'><p>{formatted_answer}</p></div>", unsafe_allow_html=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Weather at Destination</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><p>{weather}</p></div>", unsafe_allow_html=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Price Prediction</div>", unsafe_allow_html=True)
            avg_price = get_avg_price(departure, arrival)
            current_price = get_price(departure, arrival)
            if avg_price:
                if current_price < avg_price:
                    st.markdown(f"<div class='card-success'><p>Good time to book — Current Rs {current_price:,} is below the 30-day average of Rs {round(avg_price):,}</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='card-warning'><p>Prices are above average — Current Rs {current_price:,} vs average Rs {round(avg_price):,}. Consider waiting.</p></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='card'><p>Not enough data for prediction yet. Search more routes to build history.</p></div>", unsafe_allow_html=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Best Day to Fly</div>", unsafe_allow_html=True)
            day_data = get_prices_by_day(departure, arrival)
            if day_data:
                days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
                df_days = pd.DataFrame(day_data, columns=["Day", "Avg Price (Rs)"])
                df_days["Day"] = df_days["Day"].apply(lambda x: days[int(x)])
                fig2 = px.bar(df_days, x="Day", y="Avg Price (Rs)",
                              color="Avg Price (Rs)",
                              color_continuous_scale="RdYlGn_r",
                              template="plotly_white")
                fig2.update_layout(
                    font_family="Inter",
                    font_color="#000000",
                    title_text="",
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(gridcolor="#000000", color="#000000", linecolor="#000000", tickfont=dict(color="#000000")),
                    xaxis=dict(gridcolor="#000000", color="#000000", linecolor="#000000", tickfont=dict(color="#000000"))
                )
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.markdown("<div class='card'><p>Search more times to build heatmap data.</p></div>", unsafe_allow_html=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Price Trend</div>", unsafe_allow_html=True)
            history = get_price_history(departure, arrival)
            if history:
                df = pd.DataFrame(history, columns=["Date", "Avg Price (Rs)"])
                fig = px.line(df, x="Date", y="Avg Price (Rs)", template="plotly_white")
                fig.update_xaxes(tickformat="%b %d")
                fig.update_traces(line_color="#76b900", line_width=3)
                fig.update_layout(
                    font_family="Inter",
                    font_color="#000000",
                    title_text="",
                    plot_bgcolor="#ffffff",
                    paper_bgcolor="#ffffff",
                    margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(gridcolor="#000000", color="#000000", linecolor="#000000", tickfont=dict(color="#000000")),
                    xaxis=dict(gridcolor="#000000", color="#000000", linecolor="#000000", tickfont=dict(color="#000000"))
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.markdown("<div class='card'><p>Search more times to build price history.</p></div>", unsafe_allow_html=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Destination Guide</div>", unsafe_allow_html=True)
            destination_info = get_destination_info(city_name)
            formatted = destination_info.replace("\n", "<br>")
            st.markdown(f"<div class='card'><p>{formatted}</p></div>", unsafe_allow_html=True)

elif page == "Group Planner":

    st.markdown("""
    <div class='logo'>
        <div class='logo-text'>
            <span class='logo-travel'>TRAVEL</span><span class='logo-mind'>MIND</span>
        </div>
        <div class='logo-bar'></div>
    </div>
    <p class='tagline'>Find the cheapest flights for your entire group in one click</p>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Where is everyone going?</div>", unsafe_allow_html=True)
    destination_city = st.selectbox("", list(airports.keys()), index=1, key="dest")
    destination = airports[destination_city]

    st.markdown("<div class='section-label'>How many people?</div>", unsafe_allow_html=True)
    num_people = st.number_input("", min_value=2, max_value=10, value=3)

    origins = []
    cols = st.columns(2)
    for i in range(int(num_people)):
        with cols[i % 2]:
            st.markdown(f"<div class='section-label'>Person {i+1} flying from</div>", unsafe_allow_html=True)
            person_city = st.selectbox("", list(airports.keys()), key=f"person_{i}")
            origins.append((person_city, airports[person_city]))

    if st.button("Find Cheapest Group Flights", key="group_btn"):
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Group Flight Summary</div>", unsafe_allow_html=True)

        total_cost = 0
        results_list = []

        cols2 = st.columns(2)
        for idx, (person_city, person_iata) in enumerate(origins):
            price = get_price(person_iata, destination)
            total_cost += price
            results_list.append((person_city, destination_city, price))
            with cols2[idx % 2]:
                st.markdown(f"""
                <div class='card'>
                    <p class='label'>Person {idx+1}</p>
                    <p><strong>{person_city} → {destination_city}</strong></p>
                    <p class='price'>Rs {price:,}</p>
                </div>
                """, unsafe_allow_html=True)

        avg_per_person = round(total_cost / num_people)
        cheapest = min(results_list, key=lambda x: x[2])

        st.markdown(f"<div class='card-success'><p>Total group cost: Rs {total_cost:,} &nbsp;|&nbsp; Average per person: Rs {avg_per_person:,}</p></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-success'><p>Cheapest route: {cheapest[0]} to {cheapest[1]} at Rs {cheapest[2]:,}</p></div>", unsafe_allow_html=True)

        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        summary = "\n".join([f"Person from {r[0]}: Rs{r[2]}" for r in results_list])
        prompt = f"""A group of {num_people} people are traveling to {destination_city}.
Flights and costs: {summary}
Total: Rs{total_cost}, Average: Rs{avg_per_person}
Give a short helpful travel tip."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>AI Travel Tip</div>", unsafe_allow_html=True)
        formatted_tip = response.choices[0].message.content.replace("\n", "<br>")
        st.markdown(f"<div class='card'><p>{formatted_tip}</p></div>", unsafe_allow_html=True)