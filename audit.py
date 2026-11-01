import numpy as np
import pandas as pd
import sqlite3
import os

#TODO: 1. Load the three raw tables
df_orders = pd.read_csv("data/raw/orders.csv")
df_customers = pd.read_csv("data/raw/customers.csv")
df_tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
df_products = pd.read_csv("data/raw/products.csv")


def print_table_info(name, df, is_tickets=False):

    print(f"TABLE SUMMARY: {name}")
    print("\n")
    print(f"Shape (Rows, Columns): {df.shape}\n")

    print("Data Types and Missing Value Analysis of each column:")
    report_data = []
    for col in df.columns:
        dtype = df[col].dtype
        #missing calculation for tickets text column to match constraints
        if is_tickets:
            # If keep_default_na=False, missing text means it's an empty string
            missing_count = (df[col] == "").sum()
        else:
            missing_count = df[col].isna().sum()

        missing_percentage = (missing_count / len(df)) * 100
        report_data.append([col, dtype, missing_count, f"{missing_percentage:.2f}%"])

    summary_df = pd.DataFrame(
        report_data,
        columns=["Column Name", "Data Type", "Missing Count", "Missing Percentage"],
    )
    print(summary_df.to_string(index=False))
    print("\n")


# Table report
print_table_info("orders.csv", df_orders)
print_table_info("customers.csv", df_customers)
print_table_info("tickets.csv", df_tickets, is_tickets=True)

print("=" * 50)
print("SPECIFIC AUDIT FIGURES CALCULATIONS")
print("=" * 50)

# Metrics for orders.csv
raw_orders_rows = len(df_orders)
exact_duplicates_count = df_orders.duplicated().sum()  # Standard drop count (119)
pending_rows = (df_orders["status"] == "pending").sum()
distinct_orders = df_orders["order_id"].nunique()

# Non-numeric conversion check for order_amount
converted_amounts = pd.to_numeric(df_orders["order_amount"], errors="coerce")
non_numeric_amounts = converted_amounts.isna().sum()

# Delivery minutes checks
converted_delivery = pd.to_numeric(df_orders["delivery_minutes"], errors="coerce")
negative_delivery = (converted_delivery < 0).sum()
blank_delivery = df_orders["delivery_minutes"].isna().sum()

# Metrics for customers.csv
blank_age = df_customers["age"].isna().sum()
converted_age = pd.to_numeric(df_customers["age"], errors="coerce")
age_outside_range = ((converted_age < 10) | (converted_age > 100)).sum()

# City check
distinct_cities = df_customers["city"].nunique(dropna=True)

# Gender checks
blank_gender = df_customers["gender"].isna().sum()
# Filter out actual NaNs and empty string representations if any exist
non_blank_genders = df_customers["gender"].dropna()
distinct_genders = non_blank_genders.nunique()

print(f"raw orders.csv rows: {raw_orders_rows}")
print(f"exact duplicate rows: {exact_duplicates_count}")
print(f"status pending rows: {pending_rows}")
print(f"distinct order_id: {distinct_orders}")
print(f"non-numeric amounts: {non_numeric_amounts}")
print(f"negative delivery: {negative_delivery}")
print(f"blank delivery: {blank_delivery}")
print(f"blank age: {blank_age}")
print(f"age outside 10-100: {age_outside_range}")
print(f"distinct cities: {distinct_cities}")
print(f"distinct non-blank genders: {distinct_genders}")
print(f"blank gender: {blank_gender}")

#TODO: CLEAN CUSTOMERS
def clean_customers():
    # (a) Standardise city by removing spaces and applying title case
    df_customers['city'] = df_customers['city'].astype(str).str.strip().str.title()

    # (b) Map every gender spelling to male, female, or unknown
    # Standardise text to lowercase to catch variants like 'Male', 'MALE', 'f', 'm', etc.
    gender_map = {
        'male': 'male',
        'm': 'male',
        'female': 'female',
        'f': 'female'
    }
    df_customers['gender'] = df_customers['gender'].astype(str).str.strip().str.lower()
    df_customers['gender'] = df_customers['gender'].map(gender_map).fillna('unknown')
    # Additionally handle actual empty/blank strings that were converted or existed
    df_customers.loc[df_customers['gender'].isin(['', 'nan', 'none']), 'gender'] = 'unknown'

    # (c) Treat ages outside 10–100 as missing, then fill with the city's median age
    df_customers['age'] = pd.to_numeric(df_customers['age'], errors='coerce')
    df_customers.loc[(df_customers['age'] < 10) | (df_customers['age'] > 100), 'age'] = np.nan

    # Calculate and fill missing values using groupby and transform
    df_customers['age'] = df_customers['age'].fillna(
        df_customers.groupby('city')['age'].transform('median')
    )

    # (d) Convert signup_date to a datetime
    df_customers['signup_date'] = pd.to_datetime(df_customers['signup_date'], errors='coerce')
    return df_customers

new_df_cust = clean_customers()
print((new_df_cust['city']=="Pune").sum())
print((new_df_cust['city']=="Delhi").sum())
print((new_df_cust['city']=="Bengaluru").sum())
print((new_df_cust['city']=="Chennai").sum())
print((new_df_cust['city']=="Hyderabad").sum())
print((new_df_cust['city']=="Mumbai").sum())
print((new_df_cust['gender']=="female").sum())
print((new_df_cust['gender']=="male").sum())
print((new_df_cust['gender']=="unknown").sum())
print(new_df_cust['age'].isna().sum())
print(new_df_cust['age'].count())
print(((new_df_cust['age']>18) & (new_df_cust['age']<60)).count())

df_products_categories = df_products['category'].unique()
#TODO: CLEAN ORDERS
def clean_orders(df_orders):
    df_orders['order_ts'] = pd.to_datetime(df_orders['order_ts'], errors="coerce")
    df_orders = df_orders.drop_duplicates()
    df_orders = df_orders.sort_values(by='order_ts', ascending=True).drop_duplicates(subset=['order_id'], keep='last')
    df_orders['order_amount'] = pd.to_numeric(df_orders['order_amount'], errors='coerce')
    df_orders = df_orders.merge(df_products[['product_id','category','price_inr']], on='product_id', how='left')
    # print(df_orders.shape[0],"\n",merge_df.shape[0])
    df_orders['order_amount'] = df_orders['order_amount'].fillna(df_orders['price_inr'] * df_orders['quantity'] * (1 - df_orders['discount_pct'] / 100))
    df_orders.loc[df_orders['delivery_minutes'] < 0, 'delivery_minutes'] = np.nan
    completed_orders = df_orders['status'] == 'completed'
    df_orders.loc[completed_orders, 'delivery_minutes'] =  df_orders.loc[completed_orders, 'delivery_minutes'].fillna(df_orders.groupby('category')['delivery_minutes'].transform('median'))
    return df_orders

new_df_orders = clean_orders(df_orders)
print(f"rows  ------->   {new_df_orders.shape[0]}")
print(new_df_orders['order_id'].nunique())
print((new_df_orders['status']=='completed').sum())
print((new_df_orders['status']=="cancelled").sum())
print((new_df_orders['status']=="pending").sum())
print(new_df_orders['order_amount'].isna().sum())
print(new_df_orders.loc[new_df_orders['status']=='completed',['delivery_minutes']].isna().sum())
print(new_df_orders.loc[new_df_orders['status']=='cancelled',['delivery_minutes']].isna().sum())
print(new_df_orders.loc[new_df_orders['status']=='completed',['order_amount']].sum().round(2))


#convert clean files to csv and store in sqlite
df_tickets.to_csv("data/clean/tickets.csv", index=False)
new_df_cust.to_csv("data/clean/customers.csv", index=False)
new_df_orders['order_ts'] = pd.to_datetime(new_df_orders['order_ts']).dt.strftime('%Y-%m-%d %H:%M:%S')
new_df_orders.to_csv('data/clean/orders.csv', index=False, encoding='utf-8')
df_products.to_csv('data/clean/products.csv', index=False, encoding='utf-8')

conn = sqlite3.connect('data/instamart.db')
df_tickets.to_sql('tickets', conn, if_exists='replace', index=False)
new_df_cust.to_sql('customers', conn, if_exists='replace', index=False)
new_df_orders.to_sql('orders', conn, if_exists='replace', index=False)
df_products.to_sql('products', conn, if_exists='replace', index=False)
conn.close()

# check products.csv for null or missing valuies
for col in df_products.columns:
    print(df_products[col].isna().sum())

#TODO:VALIDATION SCRIPT

# Checkpoint 1: Verify exactly 6 unique cities exist in customers
assert new_df_cust['city'].nunique() == 6, "Failed: City standardisation did not yield exactly 6 cities."

# Checkpoint 2: Verify specific city distribution counts match targets exactly
city_counts = new_df_cust['city'].value_counts()
assert city_counts.get('Pune', 0) == 227, "Failed: Pune count does not match 227."
assert city_counts.get('Delhi', 0) == 210, "Failed: Delhi count does not match 210."
assert city_counts.get('Bengaluru', 0) == 197, "Failed: Bengaluru count does not match 197."
assert city_counts.get('Chennai', 0) == 191, "Failed: Chennai count does not match 191"
assert city_counts.get('Hyderabad', 0) == 191, "Failed: Hyderabad count does not match 191"
assert city_counts.get('Mumbai', 0) == 184, "Failed: Mumbai count does not match 184"


# Checkpoint 3: Verify gender counts exactly match reference values
gender_counts = new_df_cust['gender'].value_counts()
assert gender_counts.get('female', 0) == 610, "Failed: Female count does not match 610."
assert gender_counts.get('male', 0) == 560, "Failed: Male count does not match 560."
assert gender_counts.get('unknown', 0) == 30, "Failed: Unknown gender count does not match 30."

# Checkpoint 4: Verify zero missing ages remain in the customer table
assert new_df_cust['age'].isna().sum() == 0, "Failed: Missing customer ages found."

# Checkpoint 5: Verify all customer ages are between 18 and 60
assert new_df_cust['age'].between(18, 60).all(), "Failed: Ages exist outside the 18-60 boundary."

# Checkpoint 6: Verify exact row duplicates were completely removed from orders
assert new_df_orders.duplicated().sum() == 0, "Failed: Exact duplicate rows still present in orders."

# Checkpoint 7: Verify order_amount calculation worked and contains zero missing entries
assert new_df_orders['order_amount'].isna().sum() == 0, "Failed: Missing values remain in order_amount."

# Checkpoint 8: Verify delivery_minutes has zero missing values for completed orders
completed_missing_delivery = new_df_orders.loc[new_df_orders['status'] == 'completed', 'delivery_minutes'].isna().sum()
assert completed_missing_delivery == 0, "Failed: Completed orders still have missing delivery minutes."

print("ALL CHECKPOINTS PASSED")


# TODO: SQL QUERIES ================================================
# 1. Create directory and write the SQL file
os.makedirs("sql", exist_ok=True)
with open("sql/queries.sql", "w") as f:
    f.write("""
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
""")

# 2. Function to read and separate individual queries from the file
with open("sql/queries.sql", "r") as f:
    # Split queries by header blocks
    raw_queries = f.read().split("\n\n\n")

# Clean up whitespace and keep non-empty text chunks
queries = [q.strip() for q in raw_queries if q.strip()]

# 3. Connect to database and run each query
conn = sqlite3.connect("data/instamart.db")

for i, query_text in enumerate(queries, 1):
    # Extract the query code
    print(f"--- Running Q{i} ---")
    df = pd.read_sql_query(query_text, conn)

    # Process or display your references/metrics here
    print(f"Row count: {len(df)}")
    print(df.head(2))
    print(df.tail(2))# preview data

conn.close()

conn = sqlite3.connect("data/instamart.db")
# TODO: PART (a): Q2 Pivot Table Validation & Named Aggregations

# Fetch the raw completed orders dataset joining orders, customers, and products
query_raw = """
SELECT 
    c.city, 
    p.category, 
    o.order_amount,
    o.discount_pct,
    o.quantity,
    o.delivery_minutes
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = 'completed';
"""
df_completed = pd.read_sql_query(query_raw, conn)

# 1. Rebuild Q2 with pd.pivot_table and margins=True
pivot_q2 = pd.pivot_table(
    df_completed,
    values="order_amount",
    index="city",
    columns="category",
    aggfunc="sum",
    margins=True,
    margins_name="Total"
)

# Fetch the original SQL aggregate total for absolute proof
sql_total = pd.read_sql_query(
    "SELECT SUM(order_amount) FROM orders WHERE status = 'completed';", conn
).iloc[0, 0]

# Display validation checks
print("--- PART (a) VALIDATION ---")
print(f"SQL Grand Total: INR {sql_total:,.2f}")
print(f"Pandas Pivot Grand Total: INR {pivot_q2.loc['Total', 'Total']:,.2f}")
print(f"Matches Exactly: {np.isclose(sql_total, pivot_q2.loc['Total', 'Total'])}\n")

# 2. Groupby and Named Aggregation per City
city_summary = df_completed.groupby("city").agg(
    sum_order_amount=pd.NamedAgg(column="order_amount", aggfunc="sum"),
    mean_order_amount=pd.NamedAgg(column="order_amount", aggfunc="mean"),
    count_orders=pd.NamedAgg(column="order_amount", aggfunc="count")
)
print("City Summary Breakdown:")
print(city_summary.loc[["Bengaluru", "Mumbai"]].round(2))
print("-" * 50 + "\n")

# TODO:PART (b):
print("--- PART (b) METRICS ---")
amount = df_completed["order_amount"]

# Statistical Metrics
mean_val = amount.mean()
median_val = amount.median()
skew_val = amount.skew()
q1 = amount.quantile(0.25)
q3 = amount.quantile(0.75)
iqr = q3 - q1

lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr

outliers_below = (amount < lower_fence).sum()
outliers_above = (amount > upper_fence).sum()

print(f"Mean: {mean_val:.2f} | Median: {median_val:.2f}")
print(f"Skewness: {skew_val:.2f} (Right-Skewed)")
print(f"Quartiles -> Q1: {q1:.2f} | Q3: {q3:.2f} | IQR: {iqr:.2f}")
print(f"Fences -> Lower: {lower_fence:.2f} | Upper: {upper_fence:.2f}")
print(f"Outliers -> Below Lower Fence: {outliers_below} | Above Upper Fence: {outliers_above}\n")

# Correlation Matrix Analysis
cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr_matrix = df_completed[cols].corr()
print("Correlation Matrix:")
print(corr_matrix.round(4))

conn.close()
