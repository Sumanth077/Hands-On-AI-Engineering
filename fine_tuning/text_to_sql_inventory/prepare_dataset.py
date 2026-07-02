"""Prepare training dataset from custom inventory examples."""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_PATH = DATA_DIR / "dataset.jsonl"

INSTRUCTION_PREFIX = (
    "Using the inventory database with tables: products, stock_levels, warehouses, suppliers, orders. "
    "Question: "
)

CUSTOM_PAIRS = [
    ("How many products are in each category?", "SELECT category, COUNT(*) AS product_count FROM products GROUP BY category;"),
    ("List all brake pad products with their prices.", "SELECT name, brand, unit_price FROM products WHERE category = 'Brake Pads';"),
    ("Which products have stock below their reorder point?", "SELECT p.name, p.brand, sl.quantity, sl.reorder_point FROM products p JOIN stock_levels sl ON p.product_id = sl.product_id WHERE sl.quantity < sl.reorder_point;"),
    ("What is the total stock quantity for each warehouse?", "SELECT w.name, SUM(sl.quantity) AS total_stock FROM warehouses w JOIN stock_levels sl ON w.warehouse_id = sl.warehouse_id GROUP BY w.warehouse_id, w.name;"),
    ("Show all pending orders with product names.", "SELECT o.order_id, p.name, o.quantity, o.order_date FROM orders o JOIN products p ON o.product_id = p.product_id WHERE o.status = 'pending';"),
    ("Which supplier has the most orders?", "SELECT s.name, COUNT(o.order_id) AS order_count FROM suppliers s JOIN orders o ON s.supplier_id = o.supplier_id GROUP BY s.supplier_id, s.name ORDER BY order_count DESC LIMIT 1;"),
    ("List the top 5 most expensive products.", "SELECT name, brand, unit_price FROM products ORDER BY unit_price DESC LIMIT 5;"),
    ("What is the average unit price by category?", "SELECT category, AVG(unit_price) AS avg_price FROM products GROUP BY category;"),
    ("How many orders were placed in May 2025?", "SELECT COUNT(*) AS order_count FROM orders WHERE order_date LIKE '2025-05-%';"),
    ("Show all oil filters and their total stock across warehouses.", "SELECT p.name, p.brand, SUM(sl.quantity) AS total_quantity FROM products p JOIN stock_levels sl ON p.product_id = sl.product_id WHERE p.category = 'Oil Filters' GROUP BY p.product_id, p.name, p.brand;"),
    ("Which warehouses are in California?", "SELECT name, location, capacity FROM warehouses WHERE location LIKE '%CA%';"),
    ("List all suppliers with lead time under 7 days.", "SELECT name, contact_email, lead_time_days FROM suppliers WHERE lead_time_days < 7;"),
    ("What is the total inventory value at each warehouse?", "SELECT w.name, SUM(p.unit_price * sl.quantity) AS inventory_value FROM warehouses w JOIN stock_levels sl ON w.warehouse_id = sl.warehouse_id JOIN products p ON sl.product_id = p.product_id GROUP BY w.warehouse_id, w.name;"),
    ("Show products ordered from AutoParts Global.", "SELECT DISTINCT p.name, p.brand, p.category, p.unit_price FROM products p JOIN orders o ON p.product_id = o.product_id JOIN suppliers s ON o.supplier_id = s.supplier_id WHERE s.name = 'AutoParts Global';"),
    ("How many spark plugs are in stock at the East Coast Hub?", "SELECT SUM(sl.quantity) AS total_spark_plugs FROM stock_levels sl JOIN products p ON sl.product_id = p.product_id JOIN warehouses w ON sl.warehouse_id = w.warehouse_id WHERE p.category = 'Spark Plugs' AND w.name = 'East Coast Hub';"),
    ("List all alternators with stock below 50 units.", "SELECT p.name, p.brand, sl.quantity, w.name AS warehouse FROM products p JOIN stock_levels sl ON p.product_id = sl.product_id JOIN warehouses w ON sl.warehouse_id = w.warehouse_id WHERE p.category = 'Alternators' AND sl.quantity < 50;"),
    ("What orders are currently shipped?", "SELECT o.order_id, p.name, o.quantity, o.order_date, s.name AS supplier FROM orders o JOIN products p ON o.product_id = p.product_id JOIN suppliers s ON o.supplier_id = s.supplier_id WHERE o.status = 'shipped';"),
    ("Count products in the Suspension category.", "SELECT COUNT(*) AS suspension_count FROM products WHERE category = 'Suspension';"),
    ("Show the cheapest product in each category.", "SELECT category, name, MIN(unit_price) AS min_price FROM products GROUP BY category;"),
    ("Which products need reordering at the Midwest Distribution warehouse?", "SELECT p.name, sl.quantity, sl.reorder_point FROM products p JOIN stock_levels sl ON p.product_id = sl.product_id JOIN warehouses w ON sl.warehouse_id = w.warehouse_id WHERE w.name = 'Midwest Distribution' AND sl.quantity < sl.reorder_point;"),
    ("List all battery products and their order suppliers.", "SELECT DISTINCT p.name, p.brand, p.unit_price, s.name AS supplier FROM products p JOIN orders o ON p.product_id = o.product_id JOIN suppliers s ON o.supplier_id = s.supplier_id WHERE p.category = 'Batteries';"),
    ("What is the total quantity ordered for brake pads?", "SELECT SUM(o.quantity) AS total_ordered FROM orders o JOIN products p ON o.product_id = p.product_id WHERE p.category = 'Brake Pads';"),
    ("Show warehouse capacity and current stock utilization.", "SELECT w.name, w.capacity, SUM(sl.quantity) AS current_stock FROM warehouses w JOIN stock_levels sl ON w.warehouse_id = sl.warehouse_id GROUP BY w.warehouse_id, w.name, w.capacity;"),
    ("Find all products with unit price over 100 dollars.", "SELECT name, brand, category, unit_price FROM products WHERE unit_price > 100;"),
    ("List delivered orders from June 2025.", "SELECT o.order_id, p.name, o.quantity, o.order_date FROM orders o JOIN products p ON o.product_id = p.product_id WHERE o.status = 'delivered' AND o.order_date LIKE '2025-06-%';"),
    ("How many unique categories of products exist?", "SELECT COUNT(DISTINCT category) AS category_count FROM products;"),
    ("Which brake pads have less than 10 units in stock?", "SELECT p.name, p.brand, sl.quantity, w.name AS warehouse FROM products p JOIN stock_levels sl ON p.product_id = sl.product_id JOIN warehouses w ON sl.warehouse_id = w.warehouse_id WHERE p.category = 'Brake Pads' AND sl.quantity < 10;"),
    ("Which warehouse has the most total stock?", "SELECT w.name, SUM(sl.quantity) AS total_stock FROM warehouses w JOIN stock_levels sl ON w.warehouse_id = sl.warehouse_id GROUP BY w.warehouse_id, w.name ORDER BY total_stock DESC LIMIT 1;"),
    ("List all cooling system products.", "SELECT name, brand, unit_price FROM products WHERE category = 'Cooling';"),
    ("What is the average order quantity?", "SELECT AVG(quantity) AS avg_order_qty FROM orders;"),
    ("Show suppliers and the number of orders they have fulfilled.", "SELECT s.name, s.lead_time_days, COUNT(o.order_id) AS order_count FROM suppliers s LEFT JOIN orders o ON s.supplier_id = o.supplier_id GROUP BY s.supplier_id, s.name, s.lead_time_days;"),
    ("Find products with no orders.", "SELECT p.name, p.brand FROM products p LEFT JOIN orders o ON p.product_id = o.product_id WHERE o.order_id IS NULL;"),
    ("What is the total value of pending orders?", "SELECT SUM(p.unit_price * o.quantity) AS pending_value FROM orders o JOIN products p ON o.product_id = p.product_id WHERE o.status = 'pending';"),
    ("List all air filter products sorted by price.", "SELECT name, brand, unit_price FROM products WHERE category = 'Air Filters' ORDER BY unit_price;"),
    ("How many stock entries exist per product on average?", "SELECT AVG(stock_count) AS avg_entries FROM (SELECT product_id, COUNT(*) AS stock_count FROM stock_levels GROUP BY product_id);"),
    ("Show the West Coast Depot stock for wiper products.", "SELECT p.name, sl.quantity FROM products p JOIN stock_levels sl ON p.product_id = sl.product_id JOIN warehouses w ON sl.warehouse_id = w.warehouse_id WHERE w.name = 'West Coast Depot' AND p.category = 'Wipers';"),
    ("Which products have stock in all warehouses?", "SELECT p.name FROM products p JOIN stock_levels sl ON p.product_id = sl.product_id GROUP BY p.product_id, p.name HAVING COUNT(DISTINCT sl.warehouse_id) = (SELECT COUNT(*) FROM warehouses);"),
    ("List orders for alternators.", "SELECT o.order_id, p.name, o.quantity, o.status FROM orders o JOIN products p ON o.product_id = p.product_id WHERE p.category = 'Alternators';"),
    ("What is the maximum stock quantity for any product?", "SELECT MAX(quantity) AS max_stock FROM stock_levels;"),
    ("Show products ordered from EuroBrake GmbH.", "SELECT DISTINCT p.name, p.brand, p.unit_price FROM products p JOIN orders o ON p.product_id = o.product_id JOIN suppliers s ON o.supplier_id = s.supplier_id WHERE s.name = 'EuroBrake GmbH';"),
    ("Count orders by status.", "SELECT status, COUNT(*) AS order_count FROM orders GROUP BY status;"),
    ("List belt products with their stock at Chicago warehouse.", "SELECT p.name, sl.quantity FROM products p JOIN stock_levels sl ON p.product_id = sl.product_id JOIN warehouses w ON sl.warehouse_id = w.warehouse_id WHERE p.category = 'Belts' AND w.location LIKE '%Chicago%';"),
    ("What is the total number of products?", "SELECT COUNT(*) AS total_products FROM products;"),
    ("Show stock entries with the lowest quantities.", "SELECT p.name, w.name AS warehouse, sl.quantity, sl.reorder_point FROM stock_levels sl JOIN products p ON sl.product_id = p.product_id JOIN warehouses w ON sl.warehouse_id = w.warehouse_id ORDER BY sl.quantity ASC LIMIT 10;"),
    ("Find warehouses with capacity over 50000.", "SELECT name, location, capacity FROM warehouses WHERE capacity > 50000;"),
    ("List spark plug products under 10 dollars.", "SELECT name, brand, unit_price FROM products WHERE category = 'Spark Plugs' AND unit_price < 10;"),
    ("What is the total stock of brake pads across all warehouses?", "SELECT SUM(sl.quantity) AS total_brake_pads FROM stock_levels sl JOIN products p ON sl.product_id = p.product_id WHERE p.category = 'Brake Pads';"),
    ("Show order details for order ID 5.", "SELECT o.order_id, p.name, s.name AS supplier, o.quantity, o.order_date, o.status FROM orders o JOIN products p ON o.product_id = p.product_id JOIN suppliers s ON o.supplier_id = s.supplier_id WHERE o.order_id = 5;"),
    ("Which category has the highest average price?", "SELECT category, AVG(unit_price) AS avg_price FROM products GROUP BY category ORDER BY avg_price DESC LIMIT 1;"),
    ("List all products with their brand.", "SELECT name, brand, category, unit_price FROM products ORDER BY brand, name;"),
]


def _expand_custom_pairs() -> list[dict]:
    """Generate 200 custom inventory training pairs."""
    pairs = []
    for question, sql in CUSTOM_PAIRS:
        pairs.append({
            "instruction": INSTRUCTION_PREFIX + question,
            "output": sql,
        })

    categories = [
        "Brake Pads", "Oil Filters", "Spark Plugs", "Alternators",
        "Suspension", "Batteries", "Cooling", "Belts", "Air Filters", "Wipers",
    ]
    warehouses = [
        "East Coast Hub", "Midwest Distribution", "West Coast Depot",
        "Southern Regional", "Pacific Northwest",
    ]
    statuses = ["pending", "shipped", "delivered"]
    brands = ["Bosch", "ACDelco", "Fram", "NGK", "Denso", "Monroe", "Brembo", "Wagner"]

    idx = len(pairs)
    while len(pairs) < 200:
        cat = categories[idx % len(categories)]
        wh = warehouses[idx % len(warehouses)]
        status = statuses[idx % len(statuses)]
        brand = brands[idx % len(brands)]

        templates = [
            (
                f"How many {cat.lower()} products do we have?",
                f"SELECT COUNT(*) AS count FROM products WHERE category = '{cat}';",
            ),
            (
                f"What is the total stock of {cat.lower()} at {wh}?",
                f"SELECT SUM(sl.quantity) AS total FROM stock_levels sl JOIN products p ON sl.product_id = p.product_id JOIN warehouses w ON sl.warehouse_id = w.warehouse_id WHERE p.category = '{cat}' AND w.name = '{wh}';",
            ),
            (
                f"List {cat.lower()} products that need reordering.",
                f"SELECT p.name, p.brand, sl.quantity, sl.reorder_point FROM products p JOIN stock_levels sl ON p.product_id = sl.product_id WHERE p.category = '{cat}' AND sl.quantity < sl.reorder_point;",
            ),
            (
                f"Show {status} orders for {cat.lower()} products.",
                f"SELECT o.order_id, p.name, o.quantity FROM orders o JOIN products p ON o.product_id = p.product_id WHERE p.category = '{cat}' AND o.status = '{status}';",
            ),
            (
                f"What is the average price of {cat.lower()} products?",
                f"SELECT AVG(unit_price) AS avg_price FROM products WHERE category = '{cat}';",
            ),
            (
                f"List {cat.lower()} products from brand {brand}.",
                f"SELECT name, brand, unit_price FROM products WHERE category = '{cat}' AND brand = '{brand}';",
            ),
        ]

        question, sql = templates[idx % len(templates)]
        pairs.append({
            "instruction": INSTRUCTION_PREFIX + question,
            "output": sql,
        })
        idx += 1

    return pairs[:200]


def prepare_dataset() -> None:
    """Build and save the custom inventory training dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    custom_examples = _expand_custom_pairs()
    assert len(custom_examples) == 200, f"Expected 200 custom pairs, got {len(custom_examples)}"

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for example in custom_examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Saved {len(custom_examples)} custom inventory pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    prepare_dataset()
