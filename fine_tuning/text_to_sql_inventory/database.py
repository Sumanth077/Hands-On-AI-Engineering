"""SQLite inventory database setup and safe query execution."""

import re
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "data" / "inventory.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    product_id INTEGER PRIMARY KEY,
    name TEXT,
    category TEXT,
    brand TEXT,
    unit_price REAL
);

CREATE TABLE IF NOT EXISTS stock_levels (
    stock_id INTEGER PRIMARY KEY,
    product_id INTEGER,
    warehouse_id INTEGER,
    quantity INTEGER,
    reorder_point INTEGER,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE IF NOT EXISTS warehouses (
    warehouse_id INTEGER PRIMARY KEY,
    name TEXT,
    location TEXT,
    capacity INTEGER
);

CREATE TABLE IF NOT EXISTS suppliers (
    supplier_id INTEGER PRIMARY KEY,
    name TEXT,
    contact_email TEXT,
    lead_time_days INTEGER
);

CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    product_id INTEGER,
    supplier_id INTEGER,
    quantity INTEGER,
    order_date TEXT,
    status TEXT,
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);
"""

SUPPLIERS = [
    (1, "AutoParts Global", "sales@autopartsglobal.com", 3),
    (2, "EuroBrake GmbH", "info@eurobrake.de", 5),
    (3, "Pacific Filters Inc", "orders@pacificfilters.com", 4),
    (4, "SparkTech Ltd", "contact@sparktech.co.uk", 7),
    (5, "ElectroDrive Systems", "support@electrodrive.com", 5),
    (6, "Nippon Auto Supply", "hello@nipponauto.jp", 10),
    (7, "Brasil Pecas", "vendas@brasilpecas.com.br", 14),
    (8, "Nordic Parts AB", "info@nordicparts.se", 8),
]

WAREHOUSES = [
    (1, "East Coast Hub", "Newark, NJ", 50000),
    (2, "Midwest Distribution", "Chicago, IL", 75000),
    (3, "West Coast Depot", "Los Angeles, CA", 60000),
    (4, "Southern Regional", "Atlanta, GA", 45000),
    (5, "Pacific Northwest", "Seattle, WA", 35000),
]

PRODUCTS = [
    (1, "Ceramic Brake Pad Set Front", "Brake Pads", "Bosch", 45.99),
    (2, "Ceramic Brake Pad Set Rear", "Brake Pads", "Bosch", 39.99),
    (3, "Semi-Metallic Brake Pads", "Brake Pads", "ACDelco", 28.50),
    (4, "Performance Brake Pads", "Brake Pads", "Brembo", 89.99),
    (5, "Economy Brake Pads", "Brake Pads", "Wagner", 19.99),
    (6, "Heavy Duty Truck Brake Pads", "Brake Pads", "Wagner", 65.00),
    (7, "Carbon Ceramic Brake Pads", "Brake Pads", "Brembo", 149.99),
    (8, "Organic Brake Pads", "Brake Pads", "Monroe", 32.00),
    (9, "Standard Oil Filter", "Oil Filters", "Fram", 8.99),
    (10, "Premium Oil Filter", "Oil Filters", "Fram", 14.99),
    (11, "Synthetic Oil Filter", "Oil Filters", "Mann", 18.50),
    (12, "Truck Oil Filter", "Oil Filters", "ACDelco", 22.00),
    (13, "Racing Oil Filter", "Oil Filters", "K&N", 24.99),
    (14, "Eco Oil Filter", "Oil Filters", "Fram", 11.50),
    (15, "Magnetic Oil Filter", "Oil Filters", "Denso", 16.99),
    (16, "Cartridge Oil Filter", "Oil Filters", "Champion", 9.99),
    (17, "Iridium Spark Plug Single", "Spark Plugs", "NGK", 12.99),
    (18, "Iridium Spark Plug 4-Pack", "Spark Plugs", "NGK", 45.99),
    (19, "Platinum Spark Plug", "Spark Plugs", "Denso", 8.99),
    (20, "Copper Spark Plug", "Spark Plugs", "ACDelco", 3.99),
    (21, "Double Platinum Spark Plug", "Spark Plugs", "Champion", 15.99),
    (22, "Racing Spark Plug", "Spark Plugs", "NGK", 22.50),
    (23, "Cold Start Spark Plug", "Spark Plugs", "Denso", 11.50),
    (24, "Motorcycle Spark Plug", "Spark Plugs", "Champion", 7.99),
    (25, "Standard Alternator 90A", "Alternators", "Bosch", 129.99),
    (26, "High Output Alternator 150A", "Alternators", "Denso", 189.99),
    (27, "Remanufactured Alternator", "Alternators", "ACDelco", 89.99),
    (28, "Performance Alternator 200A", "Alternators", "Valeo", 249.99),
    (29, "Truck Alternator 250A", "Alternators", "ACDelco", 299.99),
    (30, "Compact Alternator 70A", "Alternators", "Denso", 99.99),
    (31, "Front Strut Assembly", "Suspension", "Monroe", 89.99),
    (32, "Rear Shock Absorber", "Suspension", "Monroe", 45.00),
    (33, "Coil Spring Set", "Suspension", "Monroe", 65.00),
    (34, "Sway Bar Link Kit", "Suspension", "ACDelco", 28.99),
    (35, "Control Arm Bushing", "Suspension", "ACDelco", 18.50),
    (36, "Standard Battery 12V 60Ah", "Batteries", "ACDelco", 119.99),
    (37, "AGM Battery 12V 70Ah", "Batteries", "Denso", 189.99),
    (38, "Lithium Battery 12V 50Ah", "Batteries", "Valeo", 349.99),
    (39, "Truck Battery 12V 900CCA", "Batteries", "ACDelco", 229.99),
    (40, "Marine Battery 12V 100Ah", "Batteries", "Fram", 159.99),
    (41, "Water Pump Assembly", "Cooling", "ACDelco", 79.99),
    (42, "Electric Water Pump", "Cooling", "Denso", 149.99),
    (43, "Radiator Standard", "Cooling", "ACDelco", 189.99),
    (44, "Radiator Performance", "Cooling", "Bosch", 279.99),
    (45, "Thermostat 180F", "Cooling", "ACDelco", 12.99),
    (46, "Thermostat 195F", "Cooling", "ACDelco", 12.99),
    (47, "Serpentine Belt", "Belts", "ACDelco", 24.99),
    (48, "Timing Belt Kit", "Belts", "Bosch", 89.99),
    (49, "V-Belt Set", "Belts", "ACDelco", 18.99),
    (50, "Performance Drive Belt", "Belts", "NGK", 34.99),
    (51, "Engine Air Filter", "Air Filters", "Fram", 15.99),
    (52, "Cabin Air Filter", "Air Filters", "Fram", 12.99),
    (53, "Performance Air Filter", "Air Filters", "K&N", 49.99),
    (54, "Cold Air Intake Filter", "Air Filters", "K&N", 29.99),
    (55, "Windshield Wiper Blade 22in", "Wipers", "ACDelco", 14.99),
    (56, "Windshield Wiper Blade 18in", "Wipers", "ACDelco", 12.99),
]

STOCK_LEVELS = [
    (1, 1, 1, 450, 100),
    (2, 1, 2, 320, 100),
    (3, 2, 1, 280, 80),
    (4, 2, 3, 190, 80),
    (5, 3, 2, 600, 150),
    (6, 3, 4, 420, 150),
    (7, 4, 1, 85, 20),
    (8, 4, 5, 6, 20),
    (9, 5, 2, 800, 200),
    (10, 5, 3, 650, 200),
    (11, 6, 4, 120, 30),
    (12, 7, 1, 8, 10),
    (13, 8, 2, 340, 80),
    (14, 9, 1, 1200, 300),
    (15, 9, 3, 980, 300),
    (16, 10, 2, 750, 200),
    (17, 10, 4, 520, 200),
    (18, 11, 1, 430, 100),
    (19, 12, 5, 180, 50),
    (20, 13, 1, 65, 15),
    (21, 14, 3, 890, 200),
    (22, 15, 2, 310, 75),
    (23, 16, 4, 540, 120),
    (24, 17, 1, 2400, 500),
    (25, 17, 2, 1800, 500),
    (26, 18, 3, 320, 80),
    (27, 19, 2, 1500, 400),
    (28, 20, 4, 2200, 500),
    (29, 21, 1, 680, 150),
    (30, 22, 5, 95, 25),
    (31, 23, 3, 410, 100),
    (32, 24, 2, 760, 200),
    (33, 25, 1, 145, 40),
    (34, 25, 4, 88, 40),
    (35, 26, 2, 62, 20),
    (36, 27, 3, 210, 50),
    (37, 28, 1, 38, 10),
    (38, 29, 5, 55, 15),
    (39, 30, 2, 175, 45),
    (40, 31, 1, 220, 50),
    (41, 32, 3, 380, 80),
    (42, 33, 2, 145, 35),
    (43, 34, 4, 520, 100),
    (44, 35, 1, 890, 200),
    (45, 36, 2, 310, 75),
    (46, 37, 3, 125, 30),
    (47, 38, 1, 42, 10),
    (48, 39, 5, 78, 20),
    (49, 40, 2, 165, 40),
    (50, 41, 1, 195, 50),
    (51, 42, 3, 48, 12),
    (52, 43, 2, 88, 20),
    (53, 44, 1, 35, 8),
    (54, 45, 4, 720, 150),
    (55, 46, 2, 680, 150),
    (56, 47, 3, 950, 200),
    (57, 48, 1, 125, 30),
    (58, 49, 5, 440, 100),
    (59, 50, 2, 78, 20),
    (60, 51, 1, 1100, 250),
    (61, 52, 3, 870, 200),
    (62, 53, 2, 95, 25),
    (63, 54, 4, 210, 50),
    (64, 55, 1, 560, 120),
    (65, 56, 2, 480, 100),
]

ORDERS = [
    (1, 1, 2, 50, "2025-05-15", "delivered"),
    (2, 9, 3, 200, "2025-05-16", "delivered"),
    (3, 17, 4, 300, "2025-05-17", "delivered"),
    (4, 25, 5, 20, "2025-05-18", "delivered"),
    (5, 36, 1, 40, "2025-05-19", "delivered"),
    (6, 4, 2, 15, "2025-05-20", "delivered"),
    (7, 12, 1, 30, "2025-05-21", "shipped"),
    (8, 28, 5, 10, "2025-05-22", "shipped"),
    (9, 38, 5, 8, "2025-05-23", "pending"),
    (10, 7, 2, 12, "2025-05-24", "pending"),
    (11, 3, 1, 100, "2025-05-25", "delivered"),
    (12, 20, 1, 500, "2025-05-26", "delivered"),
    (13, 31, 1, 25, "2025-05-27", "delivered"),
    (14, 41, 1, 18, "2025-05-28", "shipped"),
    (15, 51, 3, 150, "2025-05-29", "shipped"),
    (16, 2, 2, 35, "2025-05-30", "pending"),
    (17, 10, 3, 80, "2025-06-01", "delivered"),
    (18, 22, 6, 20, "2025-06-02", "delivered"),
    (19, 44, 2, 5, "2025-06-03", "pending"),
    (20, 55, 1, 60, "2025-06-04", "shipped"),
    (21, 6, 1, 25, "2025-06-05", "delivered"),
    (22, 15, 5, 45, "2025-06-06", "delivered"),
    (23, 29, 1, 8, "2025-06-07", "pending"),
    (24, 47, 1, 90, "2025-06-08", "shipped"),
    (25, 53, 2, 15, "2025-06-09", "pending"),
]

SCHEMA_CONTEXT = """
Tables:
- products(product_id, name, category, brand, unit_price)
- stock_levels(stock_id, product_id, warehouse_id, quantity, reorder_point)
- warehouses(warehouse_id, name, location, capacity)
- suppliers(supplier_id, name, contact_email, lead_time_days)
- orders(order_id, product_id, supplier_id, quantity, order_date, status)
"""


def get_connection() -> sqlite3.Connection:
    """Return a connection to the inventory database, creating it if needed."""
    if not DB_PATH.exists():
        init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(force: bool = False) -> None:
    """Create the database schema and populate with sample data."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists() and not force:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM products")
        count = cursor.fetchone()[0]
        conn.close()
        if count >= 50:
            return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    if force or DB_PATH.exists():
        cursor.executescript("""
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS stock_levels;
            DROP TABLE IF EXISTS products;
            DROP TABLE IF EXISTS warehouses;
            DROP TABLE IF EXISTS suppliers;
        """)

    cursor.executescript(SCHEMA)

    cursor.executemany(
        "INSERT INTO suppliers (supplier_id, name, contact_email, lead_time_days) VALUES (?, ?, ?, ?)",
        SUPPLIERS,
    )
    cursor.executemany(
        "INSERT INTO warehouses (warehouse_id, name, location, capacity) VALUES (?, ?, ?, ?)",
        WAREHOUSES,
    )
    cursor.executemany(
        "INSERT INTO products (product_id, name, category, brand, unit_price) VALUES (?, ?, ?, ?, ?)",
        PRODUCTS,
    )
    cursor.executemany(
        "INSERT INTO stock_levels (stock_id, product_id, warehouse_id, quantity, reorder_point) "
        "VALUES (?, ?, ?, ?, ?)",
        STOCK_LEVELS,
    )
    cursor.executemany(
        "INSERT INTO orders (order_id, product_id, supplier_id, quantity, order_date, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ORDERS,
    )

    conn.commit()
    conn.close()


def execute_query(sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
    """
    Execute a read-only SQL query safely.

    Returns column names and rows. Raises sqlite3.Error on failure.
    """
    sql_stripped = sql.strip().rstrip(";").strip()
    upper = sql_stripped.upper()

    if not upper.startswith("SELECT"):
        raise ValueError("Only SELECT queries are allowed.")

    forbidden = (
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "CREATE", "TRUNCATE", "REPLACE", "ATTACH", "DETACH",
    )
    for keyword in forbidden:
        if re.search(rf"\b{keyword}\b", upper):
            raise ValueError(f"Query contains forbidden keyword: {keyword}")

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(sql_stripped)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()
        return columns, [tuple(row) for row in rows]
    finally:
        conn.close()


def get_schema_context() -> str:
    """Return schema description for prompt building."""
    return SCHEMA_CONTEXT.strip()


if __name__ == "__main__":
    init_db(force=True)
    print(f"Database initialized at {DB_PATH}")
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM products")
    print(f"Products: {cursor.fetchone()[0]}")
    cursor.execute("SELECT COUNT(*) FROM stock_levels")
    print(f"Stock levels: {cursor.fetchone()[0]}")
    conn.close()
