# SQL Skill — E-Commerce Database

## Database Schema

```
customers   (id, name, email, city, country, signup_date)
products    (id, name, category, price, stock)
orders      (id, customer_id, order_date, status)
order_items (id, order_id, product_id, quantity, unit_price)
```

### Column notes
- `orders.status` — one of: `pending`, `processing`, `shipped`, `delivered`, `cancelled`
- `products.category` — one of: `Electronics`, `Clothing`, `Books`, `Home & Garden`, `Sports`
- All dates are stored as ISO strings: `YYYY-MM-DD`
- Revenue for an item = `quantity * unit_price`

---

## Common Query Patterns

### Total revenue per order
```sql
SELECT order_id, SUM(quantity * unit_price) AS total
FROM order_items
GROUP BY order_id;
```

### Top customers by spend
```sql
SELECT c.name, SUM(oi.quantity * oi.unit_price) AS total_spend
FROM customers c
JOIN orders o ON c.id = o.customer_id
JOIN order_items oi ON o.id = oi.order_id
WHERE o.status != 'cancelled'
GROUP BY c.id, c.name
ORDER BY total_spend DESC
LIMIT 10;
```

### Best-selling products (by quantity)
```sql
SELECT p.name, SUM(oi.quantity) AS units_sold
FROM products p
JOIN order_items oi ON p.id = oi.product_id
GROUP BY p.id, p.name
ORDER BY units_sold DESC;
```

### Revenue by category
```sql
SELECT p.category, SUM(oi.quantity * oi.unit_price) AS revenue
FROM products p
JOIN order_items oi ON p.id = oi.product_id
JOIN orders o ON oi.order_id = o.id
WHERE o.status != 'cancelled'
GROUP BY p.category
ORDER BY revenue DESC;
```

### Orders in a date range
```sql
SELECT * FROM orders
WHERE order_date BETWEEN '2024-01-01' AND '2024-03-31';
```

### Orders by status count
```sql
SELECT status, COUNT(*) AS count
FROM orders
GROUP BY status
ORDER BY count DESC;
```

---

## SQLite Syntax Rules

1. **String literals** use single quotes: `WHERE city = 'London'`
2. **Date functions**: Use `strftime('%Y', order_date)` to extract year; `DATE('now')` for today
3. **LIKE wildcards**: `%` matches any sequence, `_` matches one character — `WHERE name LIKE '%phone%'`
4. **No ILIKE**: SQLite `LIKE` is case-insensitive for ASCII by default
5. **Integer division**: Cast explicitly — `CAST(a AS REAL) / b` for decimal results
6. **No RIGHT JOIN**: Rewrite as LEFT JOIN with tables swapped
7. **LIMIT always at end**: Place `LIMIT` after `ORDER BY`
8. **Subqueries**: Supported — wrap in parentheses with an alias: `FROM (SELECT ...) AS sub`

---

## Safety Rules

- Only write `SELECT` queries — never `INSERT`, `UPDATE`, `DELETE`, or `DROP`
- Always alias aggregated columns for clarity: `SUM(...) AS revenue`
- Check column names against the schema — never assume a column exists
