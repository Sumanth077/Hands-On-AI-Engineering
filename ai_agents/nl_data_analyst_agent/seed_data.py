"""Create a deterministic fictional sales database for the demo."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path


def seed_database(path: Path) -> None:
    """Create the local demo database without overwriting an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing database: {path}")
    rng = random.Random(42)
    products = [
        (1, "Starter", "Software", 39),
        (2, "Team", "Software", 99),
        (3, "Enterprise", "Software", 249),
        (4, "Setup", "Services", 120),
    ]
    regions = ["North", "South", "East", "West"]
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                region TEXT NOT NULL,
                signup_date TEXT NOT NULL
            );
            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                unit_price INTEGER NOT NULL
            );
            CREATE TABLE orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                product_id INTEGER NOT NULL REFERENCES products(product_id),
                order_date TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                status TEXT NOT NULL
            );
            CREATE INDEX idx_orders_date ON orders(order_date);
            CREATE INDEX idx_orders_customer ON orders(customer_id);
            """
        )
        connection.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", products)
        start = date(2025, 1, 1)
        customers = [
            (customer_id, f"Customer {customer_id:03d}", regions[(customer_id - 1) % 4],
             (start + timedelta(days=rng.randrange(180))).isoformat())
            for customer_id in range(1, 121)
        ]
        connection.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", customers)
        orders = [
            (order_id, rng.randrange(1, 121), rng.randrange(1, 5),
             (start + timedelta(days=rng.randrange(365))).isoformat(),
             rng.randrange(1, 5), "completed" if rng.random() > 0.08 else "cancelled")
            for order_id in range(1, 1201)
        ]
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)
        connection.commit()
    finally:
        connection.close()


if __name__ == "__main__":
    database = Path(__file__).resolve().parent / "data" / "demo.sqlite"
    seed_database(database)
    print(f"Created {database.resolve()} with 120 customers and 1,200 orders.")
