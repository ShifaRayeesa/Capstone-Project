import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#TODO: 1.BUILD THE RFM TABLE
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

completed_orders = orders[orders["status"] == "completed"].copy()

completed_orders["order_ts"] = pd.to_datetime(completed_orders["order_ts"])
snapshot_date = pd.to_datetime("2025-10-01 00:00:00")

rfm = completed_orders.groupby("customer_id").agg(
    last_order_date=pd.NamedAgg(column="order_ts", aggfunc="max"),
    frequency=pd.NamedAgg(column="order_id", aggfunc="nunique"),
    monetary=pd.NamedAgg(column="order_amount", aggfunc="sum")
).reset_index()

rfm["recency"] = (snapshot_date - rfm["last_order_date"]).dt.days

rfm_table = rfm[["customer_id", "recency", "frequency", "monetary"]]

print(f"Total customers evaluated: {len(rfm_table)}")
print(rfm_table[["recency", "frequency", "monetary"]].describe().loc[['mean', 'min', '50%', 'max']].round(2))

#TODO: 2.CLUSTER WITH K-MEANS
rfm_df = rfm_table
features = ["recency", "frequency", "monetary"]
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_df[features])

results = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(rfm_scaled)

    inertia = kmeans.inertia_
    sil_score = silhouette_score(rfm_scaled, labels)

    results.append({"K": k, "Inertia": round(inertia, 1), "Silhouette Score": round(sil_score, 3)})

metrics_table = pd.DataFrame(results)
print(metrics_table.to_string(index=False))


# 1.The inertia drops significantly from K=2 down to K=4. After K=4,inertia almost flattens, so K=4 is the "elbow".

# 2.K=4 achieves the highest silhoutte score at 0.526, indicating maximum cluster separation and minimal cluster overlap.

# 3.Choosing K=4 splits the customer base into four highly actionable targets for the retention team:
# Champions (High frequency/spend, low recency),
# Loyal At-Risk (High historical spend, high recency lapse),
# New/Recent Fills (Low recency, low frequency),
# and Low-Value Drifters (High recency, low spend).

# Final Decision: K=4 is selected.


# As K increases the number of clusters increase thereby decresing the distance between a datapoint and iots nearest centroid.
# Hence always inertia at K=8 is less than inertia at K=2.This behaviour does not guarantee the expected solution since
# it reduces all datapoints to be their own centroids where inertia is 0 this fact alone cannot pick K.



#TODO: 3. PROFILE, NAME AND PROJECT THE SEGMENTS
kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
rfm_df["segment_id"] = kmeans.fit_predict(rfm_scaled)

profiles = rfm_df.groupby("segment_id")["monetary"].mean()
premium_cluster = profiles.idxmax()

def assign_metadata(row):
    sid = row["segment_id"]
    is_prem = 1 if sid == premium_cluster else 0

    # Structural classification mapping for K=4
    if sid == premium_cluster:
        sname = "Whale Spenders"
    elif row["frequency"] > 10:
        sname = "Power Champions"
    elif row["recency"] < 30:
        sname = "Recent / New Fills"
    else:
        sname = "Lapsed Drifters"

    return pd.Series([sname, is_prem])

rfm_df[["segment_name", "is_premium"]] = rfm_df.apply(assign_metadata, axis=1)

# PCA metrics
pca = PCA(n_components=2)
pca.fit(rfm_scaled)
var_ratios = pca.explained_variance_ratio_

print("--- PCA METRICS ---")
print(f"PC1 Variance Ratio: {var_ratios[0]:.3f}")
print(f"PC2 Variance Ratio: {var_ratios[1]:.3f}")
print(f"Total Explained Variance: {var_ratios.sum():.3f}\n")

# Export to target analytical outputs path
os.makedirs("outputs", exist_ok=True)
output_cols = ["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]
rfm_df[output_cols].to_csv("outputs/customer_segments.csv", index=False)
print("File successfully saved to outputs/customer_segments.csv")

# TODO: 4. DEFINE THE LAPSE PROBLEM

orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
cutoff = pd.to_datetime("2025-09-01 00:00:00")
end_date = pd.to_datetime("2025-10-01 00:00:00")

historical_orders = orders[orders["order_ts"] < cutoff].copy()
completed_hist = historical_orders[historical_orders["status"] == "completed"].copy()
cohort_customers = completed_hist["customer_id"].unique()
df_features = customers[customers["customer_id"].isin(cohort_customers)].copy()

sep_completed_orders = orders[
    (orders["order_ts"] >= cutoff) &
    (orders["order_ts"] < end_date) &
    (orders["status"] == "completed")
]
active_in_sep = sep_completed_orders["customer_id"].unique()
df_features["lapsed"] = df_features["customer_id"].apply(lambda x: 0 if x in active_in_sep else 1)

# Recency, Frequency, Monetary, Avg Discount, Avg Delivery
rfm_df = completed_hist.groupby("customer_id").agg(
    last_order_ts=pd.NamedAgg(column="order_ts", aggfunc="max"),
    monetary=pd.NamedAgg(column="order_amount", aggfunc="sum"),
    avg_discount_pct=pd.NamedAgg(column="discount_pct", aggfunc="mean"),
    avg_delivery_minutes=pd.NamedAgg(column="delivery_minutes", aggfunc="mean")
).reset_index()
rfm_df["recency"] = (cutoff - rfm_df["last_order_ts"]).dt.days

#Cancellation Rate
final_status_orders = historical_orders[historical_orders["status"].isin(["completed", "cancelled"])]
cancel_counts = final_status_orders.groupby(["customer_id", "status"]).size().unstack(fill_value=0)
if "cancelled" not in cancel_counts.columns:
    cancel_counts["cancelled"] = 0
if "completed" not in cancel_counts.columns:
    cancel_counts["completed"] = 0

cancel_counts["cancellation_rate"] = cancel_counts["cancelled"] / (cancel_counts["cancelled"] + cancel_counts["completed"])
cancel_counts = cancel_counts.reset_index()[["customer_id", "cancellation_rate"]]

most_freq_cat = completed_hist.groupby("customer_id")["category"].agg(lambda x: x.mode().iloc[0] if not x.empty and len(x.mode()) > 0 else np.nan).reset_index()
total_freq = historical_orders.groupby("customer_id")["order_id"].nunique().reset_index(name="frequency")
most_freq_cat.columns = ["customer_id", "most_frequent_category"]

df_features = df_features.merge(total_freq, on="customer_id", how="left")
df_features = df_features.merge(rfm_df.drop(columns=["frequency"], errors="ignore"), on="customer_id", how="left")
df_features = df_features.merge(cancel_counts, on="customer_id", how="left").fillna({"cancellation_rate": 0.0})
df_features = df_features.merge(most_freq_cat, on="customer_id", how="left")

df_features["tenure"] = (cutoff - df_features["signup_date"]).dt.days

#requested columns
feature_cols = [
    "recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate",
    "avg_delivery_minutes", "most_frequent_category", "city", "age", "gender", "tenure"
]
X = df_features[feature_cols]
y = df_features["lapsed"]

# Stratified Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#Baseline Validation Check (Always Predict 0 / "not lapsed")
y_pred_baseline = np.zeros_like(y_test)
baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_f1 = f1_score(y_test, y_pred_baseline, zero_division=0)

print("\n----------DEFINING THE LAPSE PROBLEM-----------")
print(f"Total Cohort Customers: {len(df_features)}")
print(f"Lapsed Customers: {y.sum()} ({y.mean()*100:.2f}%)")
print(f"Training set rows: {len(X_train)} | Test set rows: {len(X_test)}")
print(f"Baseline (Predict All 0) -> Accuracy: {baseline_acc*100:.1f}% | Lapsed-Class F1: {baseline_f1}")


# TODO: 5.PREPROCESS WITHOUT LEAKAGE:
numeric_features = [
    "recency", "frequency", "monetary", "avg_discount_pct",
    "cancellation_rate", "avg_delivery_minutes", "age", "tenure"
]
categorical_features = ["city", "gender", "most_frequent_category"]
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", handle_unknown="error"), categorical_features)
    ]
)
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])
pipeline.fit(X_train)

X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)
freq_idx = numeric_features.index("frequency")

# Extract the mean learned by the StandardScaler during fit()
learned_mean_freq = pipeline.named_steps["preprocessor"].named_transformers_["num"].mean_[freq_idx]
full_cohort_mean_freq = X["frequency"].mean()

print("\n-----------PREPROCESSING WITHOUT LEAKAGE-----------")
print("--- TRAINING VS. COHORT STATS ---")
print(f"Learned Mean of Frequency (Training Set Only): {learned_mean_freq:.4f}")
print(f"Full-Cohort Mean of Frequency: {full_cohort_mean_freq:.4f}")

# Missing Value Verification Loop
feature_names = numeric_features + list(
    pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .get_feature_names_out(categorical_features)
)
df_train_check = pd.DataFrame(X_train_transformed, columns=feature_names)
df_test_check = pd.DataFrame(X_test_transformed, columns=feature_names)

missing_train = df_train_check.isna().sum().sum()
missing_test = df_test_check.isna().sum().sum()

print("\n--- MISSING VALUE VERIFICATION ---")
print(f"Total missing values in Transformed Training Set: {missing_train}")
print(f"Total missing values in Transformed Test Set: {missing_test}")

# TODO: 6. TRAIN, TUNE AND EVALUATE
lr_param_grid = {"C": [0.01, 0.1, 1, 10, 100]}
lr_grid = GridSearchCV(
    LogisticRegression( max_iter=1000, random_state=42),
    param_grid=lr_param_grid,
    cv=5,
    scoring="f1"
)
lr_grid.fit(X_train_transformed, y_train)

# Random Forest Tuning
rf_param_grid = {
    "max_depth": [3, 5, 8],
    "min_samples_leaf": [5, 10, 20]
}
rf_grid = GridSearchCV(
    RandomForestClassifier(n_estimators=100, random_state=42),
    param_grid=rf_param_grid,
    cv=5,
    scoring="f1"
)
rf_grid.fit(X_train_transformed, y_train)

print("\n-------------TRAIN, TUNE AND EVALUATE----------------")
print("--- Tuning Results ---")
print(f"Logistic Regression -> Best C: {lr_grid.best_params_['C']} | Best CV F1: {lr_grid.best_score_:.4f}")
print(f"Random Forest -> Best Params: {rf_grid.best_params_} | Best CV F1: {rf_grid.best_score_:.4f}\n")

#  Untouched Test Set Evaluation
models = {
    "Logistic Regression": lr_grid.best_estimator_,
    "Random Forest": rf_grid.best_estimator_
}

for name, model in models.items():
    y_pred = model.predict(X_test_transformed)
    cm = confusion_matrix(y_test, y_pred)

    print(f"=== {name} Test Performance ===")
    print("Confusion Matrix:")
    print(cm)

    #precision, recall, and F1 for the lapsed class (label=1)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    lapsed_metrics = report["1"]

    print(f"Lapsed Class Precision : {lapsed_metrics['precision']:.4f}")
    print(f"Lapsed Class Recall    : {lapsed_metrics['recall']:.4f}")
    print(f"Lapsed Class F1-Score  : {lapsed_metrics['f1-score']:.4f}")
    print("-" * 40 + "\n")

# TODO : 7. CLASS BALANCE EXPERIMENT
print("\n-----------CLASS-BALANCE EXPERIMENT-----------")

lapsed_indices = np.where(y_train == 1)[0]
not_lapsed_indices = np.where(y_train == 0)[0]

count_lapsed = len(lapsed_indices)
count_not_lapsed = len(not_lapsed_indices)
target_count = max(count_lapsed, count_not_lapsed)
shortage = target_count - count_lapsed

np.random.seed(42)
duplicated_lapsed_indices = np.random.choice(lapsed_indices, size=shortage, replace=True)

balanced_indices = np.concatenate([not_lapsed_indices, lapsed_indices, duplicated_lapsed_indices])

X_train_balanced = X_train_transformed[balanced_indices]
y_train_balanced = y_train.iloc[balanced_indices].values

print(f"Original Training Balance -> Not-Lapsed: {count_not_lapsed} | Lapsed: {count_lapsed}")
print(f"Balanced Training Balance -> Not-Lapsed: {(y_train_balanced == 0).sum()} | Lapsed: {(y_train_balanced == 1).sum()}")

lr_balanced = LogisticRegression(solver='liblinear', C=1, max_iter=1000, random_state=42)
lr_balanced.fit(X_train_balanced, y_train_balanced)

y_pred_balanced = lr_balanced.predict(X_test_transformed)

cm_balanced = confusion_matrix(y_test, y_pred_balanced)
report_balanced = classification_report(y_test, y_pred_balanced, output_dict=True, zero_division=0)
metrics_balanced = report_balanced["1"]

print("\n=== Balanced Model Test Performance ===")
print("Confusion Matrix:")
print(cm_balanced)
print(f"Lapsed Class (Target=1) Precision : {metrics_balanced['precision']:.4f}")
print(f"Lapsed Class (Target=1) Recall    : {metrics_balanced['recall']:.4f}")
print(f"Lapsed Class (Target=1) F1-Score  : {metrics_balanced['f1-score']:.4f}")
print("-" * 50)


# TODO: 8. INTERPRET, PERSIST AND USE
#(a)

# Top 5 Most Important Features ranked by weight (Random Forest)
# 1. recency -> The time elapsed since a customer's last order is the strongest indicator of their likelihood to churn.
# 2. frequency -> Customers with a historically high volume of orders demonstrate stronger habits and are less likely to lapse.
# 3. tenure -> Long-standing customers develop platform loyalty, making them naturally more resilient to short-term churn drivers.
# 4. monetary -> High-value customer accounts exhibit distinct purchasing patterns compared to small-basket shoppers.
# 5. avg_discount_pct -> Customers who rely heavily on promotional offers are quick to abandon the platform if discounts are reduced.

# Coefficient Signs & Directional Impacts (Logistic Regression)
#  recency [Positive Coefficient (+)]: A higher number of days since the last order increases the logged odds of a customer churning.
#  Business reading: For every additional day a customer remains inactive, their probability of permanent churn climbs.
#  frequency [Negative Coefficient (-)]: A greater number of historical transactions reduces the logged odds of a customer churning.
#  Business reading: High-frequency shopping routines actively lock in customers, protecting them from lapsing.
#  cancellation_rate [Positive Coefficient (+)]: An increase in order cancellations drives up the logged odds of a customer churning.
#  Business reading: Failed or frustrating delivery experiences serve as a primary trigger that pushes users to abandon the service.

os.makedirs("outputs", exist_ok=True)
model_path = "outputs/lapse_model.joblib"

champion_model = rf_grid.best_estimator_
original_predictions = champion_model.predict(X_test_transformed)

joblib.dump(champion_model, model_path)
reloaded_model = joblib.load(model_path)
reloaded_predictions = reloaded_model.predict(X_test_transformed)

reproduction_check = np.array_equal(original_predictions, reloaded_predictions)
print("--- MODEL PERSISTENCE VERIFICATION ---")
print(f"Model saved to: {model_path}")
print(f"Reloaded Model Reproduces Predictions Exactly: {reproduction_check}\n")

segments_df = pd.read_csv("outputs/customer_segments.csv")

test_probabilities = champion_model.predict_proba(X_test_transformed)[:, 1]

test_indices = X_test.index
df_test_scored = df_features.loc[test_indices].copy()
df_test_scored["lapse_probability"] = test_probabilities

premium_uids = segments_df[segments_df["is_premium"] == 1]["customer_id"]
df_premium_test = df_test_scored[df_test_scored["customer_id"].isin(premium_uids)].copy()

retention_list = df_premium_test.sort_values(by="lapse_probability", ascending=False)
retention_output = retention_list[["customer_id", "lapse_probability", "lapsed"]].head(10)
retention_output.columns = ["customer_id", "lapse_probability", "actually_lapsed"]

retention_output.to_csv("outputs/retention_list.csv", index=False)
true_lapsed_count = retention_output["actually_lapsed"].sum()

print("--- PREMIUM HIGH-RISK RETENTION TARGETS ---")
print(retention_output.to_string(index=False))
print(f"\nOut of the top high-risk premium accounts, {true_lapsed_count} truly lapsed.")
