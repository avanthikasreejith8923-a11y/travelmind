import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("travelmind.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS flight_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flight_number TEXT,
            departure TEXT,
            arrival TEXT,
            price REAL,
            date TEXT,
            airline TEXT,
            status TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_flight(flight_number, departure, arrival, price, airline, status):
    conn = sqlite3.connect("travelmind.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO flight_prices (flight_number, departure, arrival, price, date, airline, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (flight_number, departure, arrival, price, datetime.now().strftime("%Y-%m-%d"), airline, status))
    conn.commit()
    conn.close()

def get_price_history(departure, arrival):
    conn = sqlite3.connect("travelmind.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT date, AVG(price) FROM flight_prices
        WHERE departure=? AND arrival=?
        GROUP BY date
        ORDER BY date
    """, (departure, arrival))
    data = cursor.fetchall()
    conn.close()
    return data

def get_prices_by_day(departure, arrival):
    conn = sqlite3.connect("travelmind.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT strftime('%w', date) as day, AVG(price)
        FROM flight_prices
        WHERE departure=? AND arrival=?
        GROUP BY day
    """, (departure, arrival))
    data = cursor.fetchall()
    conn.close()
    return data

def get_avg_price(departure, arrival):
    conn = sqlite3.connect("travelmind.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT AVG(price) FROM flight_prices
        WHERE departure=? AND arrival=?
    """, (departure, arrival))
    data = cursor.fetchone()[0]
    conn.close()
    return data

init_db()