"""
SQLite e-commerce database — schema definition and sample data seeding.
Run this file directly to (re)create the database: python database.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "ecommerce.db"

CREATE_TABLES = [
    """CREATE TABLE IF NOT EXISTS customers (
        id           INTEGER PRIMARY KEY,
        name         TEXT    NOT NULL,
        email        TEXT    NOT NULL,
        city         TEXT,
        country      TEXT,
        signup_date  TEXT
    )""",
    """CREATE TABLE IF NOT EXISTS products (
        id       INTEGER PRIMARY KEY,
        name     TEXT    NOT NULL,
        category TEXT    NOT NULL,
        price    REAL    NOT NULL,
        stock    INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS orders (
        id          INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        order_date  TEXT    NOT NULL,
        status      TEXT    NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(id)
    )""",
    """CREATE TABLE IF NOT EXISTS order_items (
        id         INTEGER PRIMARY KEY,
        order_id   INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity   INTEGER NOT NULL,
        unit_price REAL    NOT NULL,
        FOREIGN KEY (order_id)   REFERENCES orders(id),
        FOREIGN KEY (product_id) REFERENCES products(id)
    )""",
]

CUSTOMERS = [
    (1,  "Alice Johnson",    "alice@email.com",    "New York",      "US",  "2022-03-10"),
    (2,  "Bob Martinez",     "bob@email.com",      "Los Angeles",   "US",  "2022-05-22"),
    (3,  "Clara Schmidt",    "clara@email.com",    "Berlin",        "DE",  "2022-07-14"),
    (4,  "David Chen",       "david@email.com",    "Shanghai",      "CN",  "2022-08-01"),
    (5,  "Eva Rossi",        "eva@email.com",      "Milan",         "IT",  "2022-09-30"),
    (6,  "Frank Dubois",     "frank@email.com",    "Paris",         "FR",  "2022-11-05"),
    (7,  "Grace Kim",        "grace@email.com",    "Seoul",         "KR",  "2023-01-18"),
    (8,  "Henry Brown",      "henry@email.com",    "London",        "UK",  "2023-02-27"),
    (9,  "Isabelle Nguyen",  "isabelle@email.com", "Toronto",       "CA",  "2023-04-03"),
    (10, "James Wilson",     "james@email.com",    "Sydney",        "AU",  "2023-05-11"),
    (11, "Karen Davis",      "karen@email.com",    "Chicago",       "US",  "2023-06-15"),
    (12, "Luis Fernandez",   "luis@email.com",     "Madrid",        "ES",  "2023-07-20"),
    (13, "Mia Tanaka",       "mia@email.com",      "Tokyo",         "JP",  "2023-08-08"),
    (14, "Noah Anderson",    "noah@email.com",     "Boston",        "US",  "2023-09-14"),
    (15, "Olivia Taylor",    "olivia@email.com",   "Amsterdam",     "NL",  "2023-10-22"),
    (16, "Paul Müller",      "paul@email.com",     "Munich",        "DE",  "2023-11-30"),
    (17, "Quinn Patel",      "quinn@email.com",    "Mumbai",        "IN",  "2024-01-07"),
    (18, "Rachel Lee",       "rachel@email.com",   "San Francisco", "US",  "2024-02-19"),
    (19, "Samuel Okafor",    "samuel@email.com",   "Lagos",         "NG",  "2024-03-03"),
    (20, "Tina Kowalski",    "tina@email.com",     "Warsaw",        "PL",  "2024-04-11"),
]

PRODUCTS = [
    # Electronics
    (1,  "Wireless Headphones",     "Electronics",   89.99,  120),
    (2,  "Bluetooth Speaker",       "Electronics",   49.99,  200),
    (3,  "USB-C Hub",               "Electronics",   34.99,  350),
    (4,  "Mechanical Keyboard",     "Electronics",  109.99,   80),
    (5,  "4K Webcam",               "Electronics",   79.99,   90),
    (6,  "Portable Charger",        "Electronics",   29.99,  400),
    # Clothing
    (7,  "Merino Wool Sweater",     "Clothing",      69.99,  160),
    (8,  "Running Shoes",           "Clothing",      94.99,  210),
    (9,  "Waterproof Jacket",       "Clothing",     129.99,   75),
    (10, "Cotton T-Shirt",          "Clothing",      19.99,  500),
    (11, "Slim Fit Chinos",         "Clothing",      54.99,  180),
    # Books
    (12, "Clean Code",              "Books",         35.99,  300),
    (13, "Designing Data-Intensive Applications", "Books", 49.99, 250),
    (14, "The Pragmatic Programmer","Books",         42.99,  280),
    (15, "Deep Learning with Python","Books",        44.99,  190),
    (16, "Atomic Habits",           "Books",         18.99,  450),
    # Home & Garden
    (17, "Bamboo Desk Organizer",   "Home & Garden", 24.99,  310),
    (18, "Ceramic Plant Pot",       "Home & Garden", 14.99,  420),
    (19, "LED Desk Lamp",           "Home & Garden", 39.99,  150),
    (20, "Memory Foam Pillow",      "Home & Garden", 44.99,  200),
    (21, "Stainless Steel Water Bottle","Home & Garden",22.99,380),
    # Sports
    (22, "Yoga Mat",                "Sports",        32.99,  260),
    (23, "Resistance Bands Set",    "Sports",        18.99,  330),
    (24, "Jump Rope",               "Sports",         9.99,  500),
    (25, "Foam Roller",             "Sports",        27.99,  175),
]

ORDERS = [
    # (id, customer_id, order_date, status)
    (1,   1, "2024-01-05", "delivered"),
    (2,   2, "2024-01-08", "delivered"),
    (3,   3, "2024-01-12", "delivered"),
    (4,   4, "2024-01-15", "delivered"),
    (5,   5, "2024-01-20", "delivered"),
    (6,   6, "2024-02-01", "delivered"),
    (7,   7, "2024-02-05", "delivered"),
    (8,   8, "2024-02-10", "delivered"),
    (9,   9, "2024-02-14", "delivered"),
    (10, 10, "2024-02-18", "delivered"),
    (11,  1, "2024-02-22", "delivered"),
    (12,  2, "2024-03-01", "delivered"),
    (13,  3, "2024-03-05", "delivered"),
    (14, 11, "2024-03-10", "delivered"),
    (15, 12, "2024-03-15", "delivered"),
    (16, 13, "2024-03-20", "delivered"),
    (17, 14, "2024-03-25", "delivered"),
    (18, 15, "2024-04-01", "delivered"),
    (19, 16, "2024-04-05", "delivered"),
    (20, 17, "2024-04-10", "shipped"),
    (21,  1, "2024-04-12", "shipped"),
    (22,  5, "2024-04-14", "shipped"),
    (23,  8, "2024-04-16", "processing"),
    (24, 18, "2024-04-18", "processing"),
    (25, 19, "2024-04-19", "processing"),
    (26, 20, "2024-04-20", "pending"),
    (27,  3, "2024-04-21", "pending"),
    (28,  7, "2024-04-22", "pending"),
    (29, 10, "2024-04-23", "pending"),
    (30,  2, "2024-04-23", "cancelled"),
    (31,  6, "2024-03-28", "cancelled"),
    (32,  9, "2024-03-02", "delivered"),
    (33, 11, "2024-03-08", "delivered"),
    (34, 13, "2024-02-25", "delivered"),
    (35, 14, "2024-01-30", "delivered"),
    (36, 15, "2024-02-08", "delivered"),
    (37, 16, "2024-01-22", "delivered"),
    (38, 17, "2024-03-18", "delivered"),
    (39, 18, "2024-02-28", "delivered"),
    (40, 19, "2024-04-02", "shipped"),
]

ORDER_ITEMS = [
    # (id, order_id, product_id, quantity, unit_price)
    (1,  1,  1, 1, 89.99),
    (2,  1,  6, 2, 29.99),
    (3,  2,  8, 1, 94.99),
    (4,  2, 10, 3, 19.99),
    (5,  3,  4, 1,109.99),
    (6,  3, 12, 1, 35.99),
    (7,  4,  2, 2, 49.99),
    (8,  4, 18, 3, 14.99),
    (9,  5,  9, 1,129.99),
    (10, 5, 22, 1, 32.99),
    (11, 6,  3, 2, 34.99),
    (12, 6, 19, 1, 39.99),
    (13, 7,  5, 1, 79.99),
    (14, 7, 15, 1, 44.99),
    (15, 8, 13, 1, 49.99),
    (16, 8, 16, 2, 18.99),
    (17, 9,  7, 2, 69.99),
    (18, 9, 21, 1, 22.99),
    (19,10, 11, 1, 54.99),
    (20,10, 24, 2,  9.99),
    (21,11,  1, 1, 89.99),
    (22,11, 14, 1, 42.99),
    (23,12,  8, 1, 94.99),
    (24,12, 23, 2, 18.99),
    (25,13,  4, 1,109.99),
    (26,13, 17, 2, 24.99),
    (27,14,  2, 1, 49.99),
    (28,14, 20, 1, 44.99),
    (29,15,  9, 1,129.99),
    (30,15, 25, 1, 27.99),
    (31,16,  5, 1, 79.99),
    (32,16, 12, 2, 35.99),
    (33,17,  6, 3, 29.99),
    (34,17, 16, 1, 18.99),
    (35,18,  3, 2, 34.99),
    (36,18, 22, 1, 32.99),
    (37,19, 15, 1, 44.99),
    (38,19, 10, 5, 19.99),
    (39,20,  7, 1, 69.99),
    (40,20, 18, 2, 14.99),
    (41,21,  1, 2, 89.99),
    (42,21,  3, 1, 34.99),
    (43,22,  9, 1,129.99),
    (44,22, 24, 3,  9.99),
    (45,23, 13, 1, 49.99),
    (46,23, 21, 2, 22.99),
    (47,24,  4, 1,109.99),
    (48,24, 19, 1, 39.99),
    (49,25,  8, 1, 94.99),
    (50,25, 11, 2, 54.99),
    (51,26, 16, 3, 18.99),
    (52,26, 23, 1, 18.99),
    (53,27,  2, 1, 49.99),
    (54,27, 17, 1, 24.99),
    (55,28,  5, 1, 79.99),
    (56,28, 20, 1, 44.99),
    (57,29, 14, 2, 42.99),
    (58,29, 25, 1, 27.99),
    (59,30,  6, 2, 29.99),
    (60,31, 12, 1, 35.99),
    (61,32,  7, 1, 69.99),
    (62,32, 10, 4, 19.99),
    (63,33,  1, 1, 89.99),
    (64,33, 22, 2, 32.99),
    (65,34,  9, 1,129.99),
    (66,34, 15, 1, 44.99),
    (67,35,  4, 1,109.99),
    (68,35,  3, 2, 34.99),
    (69,36, 13, 1, 49.99),
    (70,36, 16, 2, 18.99),
    (71,37,  8, 1, 94.99),
    (72,37, 24, 2,  9.99),
    (73,38,  2, 2, 49.99),
    (74,38, 21, 1, 22.99),
    (75,39,  5, 1, 79.99),
    (76,39, 11, 1, 54.99),
    (77,40,  6, 3, 29.99),
    (78,40, 18, 2, 14.99),
]


def init_db():
    """Create tables and seed sample data if the database doesn't exist yet."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    cursor = conn.cursor()

    for stmt in CREATE_TABLES:
        cursor.execute(stmt)
    conn.commit()

    # Only seed if tables are empty
    if cursor.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO customers VALUES (?,?,?,?,?,?)", CUSTOMERS
        )
        cursor.executemany(
            "INSERT INTO products VALUES (?,?,?,?,?)", PRODUCTS
        )
        cursor.executemany(
            "INSERT INTO orders VALUES (?,?,?,?)", ORDERS
        )
        cursor.executemany(
            "INSERT INTO order_items VALUES (?,?,?,?,?)", ORDER_ITEMS
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"✅ Database created at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    for table in ["customers", "products", "orders", "order_items"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"   {table}: {count} rows")
    conn.close()
