import streamlit as st
import requests
import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from database import init_db

load_dotenv()

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
    <p class='tagline'>AI-powered flight search — real-time data, smart recommendations, instant answers</p>
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
    question = st.text_input("", placeholder="e.g. Which flights are available today?", key="question")

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

            for i, flight in enumerate(data["data"]):
                text = f"Flight {flight['flight']['iata']} from {departure} to {arrival}. Departure: {flight['departure']['scheduled']}. Arrival: {flight['arrival']['scheduled']}. Status: {flight['flight_status']}. Airline: {flight['airline']['name']}."
                flight_texts.append(text)
                flight_ids.append(f"flight_{i}")

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
                    <p style='color:#76b900;font-size:11px;margin-top:6px;font-weight:600'>Check airline website for latest prices</p>
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
    <p class='tagline'>Find the best flights for your entire group in one click</p>
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

    if st.button("Find Group Flights", key="group_btn"):
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Group Flight Summary</div>", unsafe_allow_html=True)

        cols2 = st.columns(2)
        for idx, (person_city, person_iata) in enumerate(origins):
            with cols2[idx % 2]:
                st.markdown(f"""
                <div class='card'>
                    <p class='label'>Person {idx+1}</p>
                    <p><strong>{person_city} → {destination_city}</strong></p>
                    <p style='color:#76b900;font-size:11px;margin-top:6px;font-weight:600'>Check airline website for latest prices</p>
                </div>
                """, unsafe_allow_html=True)

        groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        origins_text = "\n".join([f"Person {i+1} from {o[0]}" for i, o in enumerate(origins)])
        prompt = f"""A group of {num_people} people are traveling to {destination_city}.
Origins: {origins_text}
Give a short helpful travel tip for this group."""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>AI Travel Tip</div>", unsafe_allow_html=True)
        formatted_tip = response.choices[0].message.content.replace("\n", "<br>")
        st.markdown(f"<div class='card'><p>{formatted_tip}</p></div>", unsafe_allow_html=True)