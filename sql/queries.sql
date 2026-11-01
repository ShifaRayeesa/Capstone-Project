
--Q1
SELECT
    customer_id,
    SUM(order_amount) AS total_amount
    FROM orders
    WHERE status = "completed" AND order_amount > 500
    GROUP BY customer_id
    HAVING SUM(order_amount) > 8000;


--Q2
SELECT
    c.city,
    p.category,
    SUM(o.order_amount),
    SUM(SUM(o.order_amount)) OVER() AS grand_total_revenue
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
    WHERE status = "completed"
    GROUP BY c.city, p.category;


--Q3
SELECT
    product_id,
    product_name,
    price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);


--Q4
SELECT
    c.customer_id
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL;


--Q5
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    cs.total_spend
FROM customer_spend cs
CROSS JOIN average_spend as_
WHERE cs.total_spend > as_.avg_spend;


--Q6
WITH customer_total AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city)
SELECT
    customer_id,
    city,
    total_spend,
    RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS rnk,
    DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rnk,
    ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_num
    FROM customer_total;


--Q7
WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    day,
    revenue,
    SUM(revenue) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue
FROM daily_revenue;
