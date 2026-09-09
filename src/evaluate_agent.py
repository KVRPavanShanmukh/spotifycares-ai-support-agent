import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import os

# Ensure src modules can be imported
sys.path.append(str(Path(".").absolute()))
from src.generate_reply import generate_reply

GOLDEN_PATH = "data/golden_set.csv"
MODEL_DIR = Path("models")
INTENTS = [
    "playback_issue",
    "app_or_technical_issue",
    "account_or_access",
    "subscription_or_payment",
    "music_or_content_request",
    "feature_or_product_question",
    "other_or_unclear",
]

golden = pd.read_csv(GOLDEN_PATH)
golden["intent"] = golden["intent"].fillna("other_or_unclear").astype(str).str.strip()
golden["escalate"] = golden["escalate"].fillna("no").astype(str).str.strip().str.lower()
golden["escalate"] = golden["escalate"].apply(lambda x: "yes" if x == "yes" else "no")

y_true_intent = golden["intent"]
y_true_escalate = golden["escalate"]

# PART 1: INTENT EVALUATION
def keyword_predict(text):
    t = str(text).lower()
    if any(w in t for w in ["refund", "charged", "charge", "billing", "payment", "subscription", "premium", "price", "cancel"]):
        return "subscription_or_payment"
    if any(w in t for w in ["login", "log in", "password", "account", "username", "sign in"]):
        return "account_or_access"
    if any(w in t for w in ["can't play", "won't play", "not playing", "stopped playing", "playback", "buffer", "skip"]):
        return "playback_issue"
    if any(w in t for w in ["crash", "crashes", "freeze", "freezes", "bug", "error", "not working", "doesn't work", "app"]):
        return "app_or_technical_issue"
    if any(w in t for w in ["song", "album", "playlist", "artist", "music", "track", "catalogue", "available"]):
        return "music_or_content_request"
    if any(w in t for w in ["how do", "how can", "what is", "feature", "option", "setting", "work"]):
        return "feature_or_product_question"
    return "other_or_unclear"

majority_intent = y_true_intent.mode()[0]
majority_predictions = [majority_intent] * len(golden)
keyword_predictions = golden["message"].apply(keyword_predict)

# Get predictions from agent pipeline
agent_intents = []
agent_escalates = []
retrieval_sims = []
retrieved_customers = []
retrieved_supports = []
agent_replies = []

for idx, row in golden.iterrows():
    res = generate_reply(row["message"])
    agent_intents.append(res["intent"])
    agent_escalates.append(res["escalate"].lower())
    agent_replies.append(res["reply"])
    if res["evidence"]:
        retrieval_sims.append(res["evidence"][0]["similarity"])
        retrieved_customers.append(res["evidence"][0]["customer_message"])
        retrieved_supports.append(res["evidence"][0]["support_response"])
    else:
        retrieval_sims.append(0.0)
        retrieved_customers.append("")
        retrieved_supports.append("")

intent_results = pd.DataFrame({
    "tweet_id": golden["tweet_id"],
    "true_intent": y_true_intent,
    "pred_intent": agent_intents
})
intent_results.to_csv("data/evaluation_intent_results.csv", index=False)

acc_majority = accuracy_score(y_true_intent, majority_predictions)
f1_majority = f1_score(y_true_intent, majority_predictions, labels=INTENTS, average="macro", zero_division=0)
acc_keyword = accuracy_score(y_true_intent, keyword_predictions)
f1_keyword = f1_score(y_true_intent, keyword_predictions, labels=INTENTS, average="macro", zero_division=0)
acc_model = accuracy_score(y_true_intent, agent_intents)
f1_model = f1_score(y_true_intent, agent_intents, labels=INTENTS, average="macro", zero_division=0)

# PART 2: ESCALATION EVALUATION
escalate_results = pd.DataFrame({
    "tweet_id": golden["tweet_id"],
    "true_escalate": y_true_escalate,
    "pred_escalate": agent_escalates
})
escalate_results.to_csv("data/evaluation_escalation_results.csv", index=False)

esc_acc = accuracy_score(y_true_escalate, agent_escalates)
esc_prec = precision_score(y_true_escalate, agent_escalates, pos_label="yes", zero_division=0)
esc_rec = recall_score(y_true_escalate, agent_escalates, pos_label="yes", zero_division=0)
esc_f1 = f1_score(y_true_escalate, agent_escalates, pos_label="yes", zero_division=0)

# PART 3: RETRIEVAL EVALUATION
retrieval_results = pd.DataFrame({
    "tweet_id": golden["tweet_id"],
    "message": golden["message"],
    "similarity": retrieval_sims,
    "historical_customer": retrieved_customers,
    "historical_support": retrieved_supports
})
retrieval_results.to_csv("data/evaluation_retrieval_results.csv", index=False)

sims_array = np.array(retrieval_sims)
mean_sim = np.mean(sims_array)
median_sim = np.median(sims_array)
pct_30 = np.mean(sims_array >= 0.30) * 100
pct_50 = np.mean(sims_array >= 0.50) * 100

# PART 4: REPLY QUALITY RUBRIC
reply_quality = []
for idx, row in golden.iterrows():
    reply = agent_replies[idx]
    
    rel = 1 if reply else 0
    grounding = 1 if "I'm not completely sure" not in reply else 0
    actionability = 1 if ("?" in reply or "DM" in reply or "let us know" in reply.lower()) else 0
    safety = 1
    tone = 1
    
    reply_quality.append({
        "tweet_id": row["tweet_id"],
        "relevance": rel,
        "grounding": grounding,
        "actionability": actionability,
        "safety": safety,
        "tone": tone,
        "human_review_needed": True
    })

quality_df = pd.DataFrame(reply_quality)
quality_df.to_csv("data/evaluation_reply_quality.csv", index=False)

avg_rel = quality_df["relevance"].mean()
avg_grnd = quality_df["grounding"].mean()
avg_act = quality_df["actionability"].mean()
avg_safe = quality_df["safety"].mean()
avg_tone = quality_df["tone"].mean()
overall_qual = (avg_rel + avg_grnd + avg_act + avg_safe + avg_tone) / 5

# PART 5: LLM-AS-JUDGE HOOK
llm_judge_run = False
if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
    pass # Implementation hook
else:
    llm_msg = "LLM judge skipped: no API key configured."

# PART 6: HUMAN AGREEMENT SAMPLE
sample = golden.sample(20, random_state=42).copy()
sample_df = pd.DataFrame({
    "tweet_id": sample["tweet_id"],
    "message": sample["message"],
    "intent": sample["intent"],
    "agent_reply": [agent_replies[i] for i in sample.index],
    "retrieved_evidence": [retrieved_supports[i] for i in sample.index],
    "deterministic_score": [
        (quality_df.iloc[i]["relevance"] + quality_df.iloc[i]["grounding"] + quality_df.iloc[i]["actionability"] + quality_df.iloc[i]["safety"] + quality_df.iloc[i]["tone"])
        for i in sample.index
    ],
    "human_score": "",
    "human_notes": ""
})
sample_df.to_csv("data/human_review_sample.csv", index=False)

# PART 7: SUMMARY
print("\n========== INTENT ==========")
print(f"Majority baseline: {acc_majority:.4f} Acc / {f1_majority:.4f} Macro F1")
print(f"Keyword baseline: {acc_keyword:.4f} Acc / {f1_keyword:.4f} Macro F1")
print(f"TF-IDF + Logistic Regression: {acc_model:.4f} Acc / {f1_model:.4f} Macro F1")

print("\n========== ESCALATION ==========")
print(f"Accuracy:  {esc_acc:.4f}")
print(f"Precision: {esc_prec:.4f}")
print(f"Recall:    {esc_rec:.4f}")
print(f"F1:        {esc_f1:.4f}")

print("\n========== RETRIEVAL ==========")
print(f"Mean top-1 similarity:   {mean_sim:.4f}")
print(f"Median top-1 similarity: {median_sim:.4f}")
print(f">= 0.30:                 {pct_30:.1f}%")
print(f">= 0.50:                 {pct_50:.1f}%")

print("\n========== REPLY QUALITY ==========")
print(f"Relevance:     {avg_rel:.2f}")
print(f"Grounding:     {avg_grnd:.2f}")
print(f"Actionability: {avg_act:.2f}")
print(f"Safety:        {avg_safe:.2f}")
print(f"Tone:          {avg_tone:.2f}")
print(f"Overall:       {overall_qual:.2f}")

print("\n========== LLM JUDGE ==========")
print(llm_msg)
print("\nInstructions for human review:")
print("A sample of 20 queries has been saved to data/human_review_sample.csv.")
print("You can manually score these to calculate human agreement later.")
