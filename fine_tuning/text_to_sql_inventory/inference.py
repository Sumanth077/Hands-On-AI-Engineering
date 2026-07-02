"""Hugging Face Transformers Text-to-SQL inference for inventory queries."""

import os
import re
from functools import lru_cache

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

MODEL_NAME = "Hecodes/text-to-sql-inventory"

SCHEMA = (
    "Tables: products(product_id, name, category, brand, unit_price), "
    "stock_levels(stock_id, product_id, warehouse_id, quantity, reorder_point), "
    "warehouses(warehouse_id, name, location, capacity), "
    "suppliers(supplier_id, name, contact_email, lead_time_days), "
    "orders(order_id, product_id, supplier_id, quantity, order_date, status)."
)


@lru_cache(maxsize=1)
def load_model():
    """Load tokenizer and model once (cached at module level)."""
    token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        token=token,
    )
    return tokenizer, model


def extract_sql(text: str) -> str:
    """Extract SQL from model response, handling code blocks."""
    text = text.strip()

    code_block = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip()

    select_match = re.search(r"(SELECT\b.*?;?)", text, re.DOTALL | re.IGNORECASE)
    if select_match:
        sql = select_match.group(1).strip()
        if not sql.endswith(";"):
            sql += ";"
        return sql

    return text.strip()


def generate_sql(question: str, tokenizer, model) -> str:
    """Generate a SQLite SELECT query from a natural language question."""
    messages = [
        {
            "role": "system",
            "content": """You are a SQLite expert. Generate ONLY a single SQL SELECT query using ONLY these exact table names, nothing else:
- products (columns: product_id, name, category, brand, unit_price)
- stock_levels (columns: stock_id, product_id, warehouse_id, quantity, reorder_point)
- warehouses (columns: warehouse_id, name, location, capacity)
- suppliers (columns: supplier_id, name, contact_email, lead_time_days)
- orders (columns: order_id, product_id, supplier_id, quantity, order_date, status)

Example: Question: How many products are in each category?
SQL: SELECT category, COUNT(*) FROM products GROUP BY category;

Example: Question: Which products have stock below their reorder point?
SQL: SELECT p.name, s.quantity, s.reorder_point FROM products p JOIN stock_levels s ON p.product_id = s.product_id WHERE s.quantity < s.reorder_point;

Only output SQL. No explanation. No markdown. No other table names."""
        },
        {
            "role": "user",
            "content": f"Question: {question}\nSQL:"
        }
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=False,
        temperature=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.3,
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response = response.strip()
    if ";" in response:
        response = response.split(";")[0] + ";"
    return response.strip()
