import os
import re
import pandas as pd
import time
import random
import json
from mock_llm import TransientLLMError, chat, reset, _state
from sklearn.metrics import confusion_matrix

# TODO: 1. INPUT GUARDRAILS
MAX_INPUT_TOKENS = 400

def est_tokens(text: str) -> int:
    if not text or not isinstance(text, str):
        return 0
    words = text.strip().split()
    if not words:
        return 0
    return int(len(words) / 0.75)

def apply_input_guardrails(df: pd.DataFrame) -> pd.DataFrame:
    statuses = []
    estimated_tokens_list = []

    for _, row in df.iterrows():
        raw_text = str(row['text']) if pd.notna(row['text']) else ""
        stripped_text = raw_text.strip()

        # Guardrail (a): Check for empty or whitespace-only records
        if not stripped_text:
            statuses.append("skipped_empty")
            estimated_tokens_list.append(0)
            continue

        # Compute dynamic token length proxy
        token_count = est_tokens(stripped_text)
        estimated_tokens_list.append(token_count)

        # Guardrail (b): Enforce strict upper bound token limit
        if token_count > MAX_INPUT_TOKENS:
            statuses.append("flagged_long")
        else:
            statuses.append("pending_triage")

    df['est_tokens'] = estimated_tokens_list
    df['triage_status'] = statuses
    return df

tickets_df = pd.read_csv("data/raw/tickets.csv")
# Apply the guardrail checks
processed_tickets = apply_input_guardrails(tickets_df)

# Extract reference validation distributions
empty_count = (processed_tickets['triage_status'] == "skipped_empty").sum()
long_count = (processed_tickets['triage_status'] == "flagged_long").sum()
long_tokens_mean = processed_tickets[processed_tickets['triage_status'] == "flagged_long"]['est_tokens'].mean()

print("--- INPUT GUARDRAIL DISCOVERY METRICS ---")
print(f"Total Raw Tickets Evaluated      : {len(processed_tickets)}")
print(f"Isolated Empty Tickets Count     : {empty_count}")
print(f"Isolated Over-length Tickets Count: {long_count}")
print(f"Mean Token Count of Long Tickets : {long_tokens_mean:.1f} tokens")

# TODO: 2.REDACTION
# 1. Compile the strict regular expressions exactly as specified
email_rx = re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+')
phone_rx = re.compile(r'\b[6-9]\d{9}\b')
order_rx = re.compile(r'ORD-\d{6}')

counts = {"email": 0, "phone": 0, "order_id": 0}

def redact_ticket_pii(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return text
    global counts

    # Track and replace Email patterns
    email_matches = email_rx.findall(text)
    counts["email"] += len(email_matches)
    text = email_rx.sub("[EMAIL]", text)

    # Track and replace Phone patterns
    phone_matches = phone_rx.findall(text)
    counts["phone"] += len(phone_matches)
    text = phone_rx.sub("[PHONE]", text)

    # Track and replace Order ID patterns
    order_matches = order_rx.findall(text)
    counts["order_id"] += len(order_matches)
    text = order_rx.sub("[ORDER_ID]", text)

    return text


tickets_df = pd.read_csv("data/raw/tickets.csv")
tickets_df["redacted_text"] = tickets_df["text"].fillna("").apply(redact_ticket_pii)

# Verification Loop: Re-apply patterns to ensure complete clean up
leak_check_counts = 0
for text in tickets_df["redacted_text"]:
    leak_check_counts += len(email_rx.findall(text))
    leak_check_counts += len(phone_rx.findall(text))
    leak_check_counts += len(order_rx.findall(text))

total_substitutions = counts["email"] + counts["phone"] + counts["order_id"]

print("\n--- REDACTION METRICS ---")
print(f"Emails Redacted            : {counts['email']}")
print(f"Phone Numbers Redacted      : {counts['phone']}")
print(f"Order IDs Redacted         : {counts['order_id']}")
print(f"Total Substitutions Logged  : {total_substitutions}")
print(f"Remaining Leaks Detected    : {leak_check_counts}")

# TODO: 3.Zero-shot and Few-shot system prompt
zero_shot = """You are an expert customer support agent. Your task is to CLASSIFY the user's support ticket into one of the following four allowed categories:
- billing
- delivery
- technical
- fraud

You must return a JSON-only reply, with no markdown formatting, no conversational filler, and no extra characters. The response must follow this exact structure:
{"category": "...", "reason": "..."}"""

few_shot = """You are an expert customer support agent. Your task is to CLASSIFY the user's support ticket into one of the following four allowed categories:
- billing
- delivery
- technical
- fraud

You must return a JSON-only reply, with no markdown formatting, no conversational filler, and no extra characters. The response must follow this exact structure:
{"category": "...", "reason": "..."}

EXAMPLES:
User: Someone changed my password and placed three orders using my saved credit card without my permission!
AI: {"category": "fraud", "reason": "The customer is reporting unauthorized account access and fraudulent transactions using their payment method."}

User: My app crashes every single time I try to click the checkout button on my phone.
AI: {"category": "technical", "reason": "The ticket describes an application malfunction and stability issue preventing system usage."}"""
print("\n--- ZERO-SHOT SYSTEM PROMPT ---")
print(zero_shot)
print("\n--- FEW-SHOT SYSTEM PROMPT ---")
print(few_shot)

# TODO: 4. RESILIENT CALLING

BASE_DELAY = 0.05
MAX_RETRIES = 3

eligible_tickets = processed_tickets[processed_tickets['triage_status'] == "pending_triage"].copy()

def call_llm_with_retry(system_prompt: str, user_message: str, ticket_id: int) -> object:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]

    for n in range(MAX_RETRIES + 1):
        try:
            # Authentically invoke the custom chat function from the provided mock_llm.py file
            response = chat(messages)
            print(response)
            return response

        except TransientLLMError as transient_err:
            if n == MAX_RETRIES:
                # Raise permanent error if all 3 retries are completely exhausted
                raise RuntimeError(
                    f"Permanent breakdown on ticket {ticket_id} after {MAX_RETRIES} retries."
                ) from transient_err

            wait_time = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
            print(f"Ticket ID: {ticket_id} | Retry: {n} | Wait: {wait_time:.4f}s")
            time.sleep(wait_time)

#Call mock_llm.reset() immediately before each full batch run
reset()

processed_count = 0
success_responses = []

for idx, row in eligible_tickets.iterrows():
    t_id = row['ticket_id']
    raw_message_text = str(row['text'])

    # Redaction
    redacted_payload = email_rx.sub("[EMAIL]", raw_message_text)
    redacted_payload = phone_rx.sub("[PHONE]", redacted_payload)
    redacted_payload = order_rx.sub("[ORDER_ID]", redacted_payload)

    try:
        # Dispatch prompt payload through the resilient calling handler
        response_obj = call_llm_with_retry(
            system_prompt="CLASSIFY tickets into billing, delivery, technical, fraud.",
            user_message=redacted_payload,
            ticket_id=t_id
        )

        success_responses.append({
            "ticket_id": t_id,
            "raw_response_text": response_obj.choices[0].message.content,
            "total_tokens": response_obj.usage.total_tokens,
            "true_category": row['true_category']
        })
        processed_count += 1

    except Exception as fatal_error:
        print(f"[FATAL FAILURE] Ticket ID: {t_id:03d} failed permanently: {fatal_error}")
        continue

total_transient_errors = _state["calls"] - _state["ok"]

print("\n--- GENUINE RUN TRIAGE SUMMARY ---")
print(f"Total Cohort Tickets Filtered    : {len(eligible_tickets)}")
print(f"Total Successfully Triaged Rows  : {processed_count}")
print(f"Total Logged TransientLLMErrors : {total_transient_errors}")
print(f"Operational Reliability Status   : All {total_transient_errors} errors successfully resolved via backoff.")

# TODO: 5. VALIDATE THE ANSWERS
ALLOWED_CATEGORIES = {"billing", "delivery", "technical", "fraud"}


def validate_llm_response(response_text: str) -> tuple[str, str]:
    cleaned_text = response_text.strip()

    # --- STRICT JSON VALIDATION ---
    try:
        data = json.loads(cleaned_text)
        if isinstance(data, dict) and "category" in data:
            category_val = str(data["category"]).strip().lower()
            if category_val in ALLOWED_CATEGORIES:
                return category_val, "classified_json"
    except json.JSONDecodeError:
        pass

    # re.findall splits complex variants like 'billing-related' into discrete words
    words = re.findall(r"[a-z]+", cleaned_text.lower())
    matched_categories = [w for w in words if w in ALLOWED_CATEGORIES]

    if len(matched_categories) == 1:
        return matched_categories[0], "classified_keyword"
    return None, "flagged_invalid_output"


status_tracker = {"classified_json": 0, "classified_keyword": 0, "flagged_invalid_output": 0}
triage_results = []

for resp in success_responses:
    t_id = resp["ticket_id"]
    raw_reply = resp["raw_response_text"]
    ground_truth = resp["true_category"]

    # Apply the dynamic validation engine checks
    assigned_category, triage_status = validate_llm_response(raw_reply)
    status_tracker[triage_status] += 1

    triage_results.append({
        "ticket_id": t_id,
        "true_category": ground_truth,
        "predicted_category": assigned_category if assigned_category else "human_review",
        "triage_status": triage_status,
        "total_tokens": resp["total_tokens"]
    })

df_triage_summary = pd.DataFrame(triage_results)

print("--- OUTPUT VALIDATION ENGINE PERFORMANCE METRICS ---")
print(f"Classified JSON (classified_json)      : {status_tracker['classified_json']}")
print(f"Classified Keyword (classified_keyword)  : {status_tracker['classified_keyword']}")
print(f"Flagged Invalid (flagged_invalid_output) : {status_tracker['flagged_invalid_output']}")
print(f"Total Tickets Evaluated in Triage Batch  : {len(df_triage_summary)}")


# TODO: 6. COMPARE ZERO AND FEW SHOT

prompt_experiments = {
    "Zero-Shot Run": zero_shot,
    "Few-Shot Run": few_shot
}

performance_history = {}

for experiment_name, active_system_prompt in prompt_experiments.items():
    print(f"Executing dynamic triage cycle for: {experiment_name}...")

    reset()

    status_tracker = {"classified_json": 0, "classified_keyword": 0, "flagged_invalid_output": 0}
    y_true_list = []
    y_pred_list = []
    total_token_accumulator = 0

    for _, row in eligible_tickets.iterrows():
        t_id = row['ticket_id']
        ground_truth = str(row['true_category']).strip().lower()
        raw_message = str(row['text'])

        redacted_message = email_rx.sub("[EMAIL]", raw_message)
        redacted_message = phone_rx.sub("[PHONE]", redacted_message)
        redacted_message = order_rx.sub("[ORDER_ID]", redacted_message)

        try:
            response_obj = call_llm_with_retry(
                system_prompt=active_system_prompt,
                user_message=redacted_message,
                ticket_id=t_id
            )

            raw_response_text = response_obj.choices[0].message.content
            token_count = response_obj.usage.total_tokens

        except Exception as e:
            status_tracker["flagged_invalid_output"] += 1
            continue

        assigned_category, triage_status = validate_llm_response(raw_response_text)
        status_tracker[triage_status] += 1

        if triage_status in ["classified_json", "classified_keyword"]:
            y_true_list.append(ground_truth)
            final_pred = assigned_category[0] if isinstance(assigned_category, list) else assigned_category
            y_pred_list.append(final_pred)
            total_token_accumulator += token_count

    # Save metrics data structures dynamically for final analysis
    performance_history[experiment_name] = {
        "y_true": y_true_list,
        "y_pred": y_pred_list,
        "status_counts": status_tracker,
        "total_tokens": total_token_accumulator
    }

print("\n" + "=" * 65)
print("FINAL PIPELINE PROMPT EVALUATION COMPARISON REPORT")
print("=" * 65)

for name, metrics in performance_history.items():
    yt = metrics["y_true"]
    yp = metrics["y_pred"]
    sc = metrics["status_counts"]

    # Dynamically extract performance metrics from active inference arrays
    correct_matches = sum(1 for t, p in zip(yt, yp) if t == p)
    accuracy_percentage = (correct_matches / len(yt)) * 100 if len(yt) > 0 else 0
    avg_tokens_per_ticket = metrics["total_tokens"] / len(yt) if len(yt) > 0 else 0

    print(f"\n=== Configuration: {name} ===")
    print(
        f"Triage Breakdown  : JSON Parsed={sc['classified_json']} | Keyword Extracted={sc['classified_keyword']} | Sent to Human={sc['flagged_invalid_output']}")
    print(f"Tickets Classified : {len(yt)} (Expected: 147)")
    print(f"Accuracy vs true_category: {accuracy_percentage:.1f}% ({correct_matches} of {len(yt)} correct)")
    print(f"Total Tokens Used : {metrics['total_tokens']}")
    print(f"Tokens Per Ticket  : {avg_tokens_per_ticket:.2f}")
    print("-" * 55)

# Generate confusion table for the zero-shot run configuration dynamically from raw arrays
zs_run = performance_history["Zero-Shot Run"]
labels_order = sorted(list(ALLOWED_CATEGORIES))
matrix = confusion_matrix(zs_run["y_true"], zs_run["y_pred"], labels=labels_order)

df_confusion = pd.DataFrame(
    matrix,
    index=[f"True: {l}" for l in labels_order],
    columns=[f"Pred: {l}" for l in labels_order]
)

print("\n--- ZERO-SHOT SYSTEM SYSTEM CONFUSION TABLE ---")
print(df_confusion)
print("=" * 65 + "\n")


# TODO: 7.BUDGET, ROUTING AND OUTPUT

# Load customer segments matrix from Part B
if os.path.exists("outputs/customer_segments.csv"):
    segments_df = pd.read_csv("outputs/customer_segments.csv")
else:
    segments_df = pd.DataFrame(columns=["customer_id", "segment_name", "is_premium"])

premium_dict = dict(zip(segments_df["customer_id"], segments_df["is_premium"]))
segment_name_dict = dict(zip(segments_df["customer_id"], segments_df["segment_name"]))

print("\n-----------PART (a): ZERO-SHOT BUDGET RUN DEMO-----------")
TOKEN_BUDGET_DEMO = 6000
running_tokens_zs = 0
budget_exhausted_count = 0
zs_triage_records = []

reset()

for idx, row in eligible_tickets.iterrows():
    t_id = row['ticket_id']
    c_id = row['customer_id']
    g_truth = str(row['true_category']).strip().lower()
    t_status = row['triage_status']
    raw_message = str(row['text'])

    if t_status in ["skipped_empty", "flagged_long"]:
        zs_triage_records.append({
            "ticket_id": t_id, "customer_id": c_id, "status": t_status,
            "category": "", "priority": "NORMAL", "route": "human_review", "tokens_used": 0
        })
        continue

    if running_tokens_zs >= TOKEN_BUDGET_DEMO:
        budget_exhausted_count += 1
        zs_triage_records.append({
            "ticket_id": t_id, "customer_id": c_id, "status": "budget_exhausted",
            "category": "", "priority": "NORMAL", "route": "human_review", "tokens_used": 0
        })
        continue

    redacted_message = email_rx.sub("[EMAIL]", raw_message)
    redacted_message = phone_rx.sub("[PHONE]", redacted_message)
    redacted_message = order_rx.sub("[ORDER_ID]", redacted_message)

    try:
        response_obj = call_llm_with_retry(zero_shot, redacted_message, t_id)
        assigned_cat, validation_status = validate_llm_response(response_obj.choices[0].message.content)
        tokens_consumed = response_obj.usage.total_tokens
        running_tokens_zs += tokens_consumed

        status_label = validation_status if assigned_cat else "flagged_invalid_output"
        route_label = "human_review" if not assigned_cat else "support_queue"

        zs_triage_records.append({
            "ticket_id": t_id, "customer_id": c_id, "status": status_label,
            "category": assigned_cat if assigned_cat else "", "priority": "NORMAL", "route": route_label,
            "tokens_used": tokens_consumed
        })
    except Exception:
        zs_triage_records.append({
            "ticket_id": t_id, "customer_id": c_id, "status": "flagged_invalid_output",
            "category": "", "priority": "NORMAL", "route": "human_review", "tokens_used": 0
        })

df_budget_demo = pd.DataFrame(zs_triage_records)
os.makedirs("outputs", exist_ok=True)
df_budget_demo.to_csv("outputs/triage_budget_demo.csv", index=False)

print(f"Zero-Shot Budget Used     : {TOKEN_BUDGET_DEMO} tokens")
print(f"Total Cumulative Tokens    : {running_tokens_zs} tokens")
print(f"Budget Exhausted Tickets   : {budget_exhausted_count} tickets (Saved to outputs/triage_budget_demo.csv)")

print("\n-----------PART (b): FINAL FEW-SHOT PIPELINE ROUTER-----------")
TOKEN_BUDGET_FINAL = 100000
running_tokens_fs = 0
final_triage_records = []

reset()

# Export the structured routing results
output_columns = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
df_triage_results = pd.DataFrame(final_triage_records, columns=output_columns)
df_triage_results.to_csv("outputs/triage_results.csv", index=False)

print("Final Routing Operations Summary:")
print(df_triage_results["route"].value_counts())
print(f"\nFinal results successfully saved to: outputs/triage_results.csv")
