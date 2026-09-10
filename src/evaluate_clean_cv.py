import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import StratifiedKFold

GOLDEN = Path("data/golden_set.csv")

INTENTS = [
    "playback_issue",
    "app_or_technical_issue",
    "account_or_access",
    "subscription_or_payment",
    "music_or_content_request",
    "feature_or_product_question",
    "other_or_unclear",
]

def keyword_predict(text):
    t = str(text).lower()
    if any(w in t for w in ["refund", "charged", "charge", "billing", "payment", "subscription", "premium", "price", "cancel", "cancellation"]):
        return "subscription_or_payment"
    if any(w in t for w in ["login", "log in", "password", "account", "username", "sign in", "signin"]):
        return "account_or_access"
    if any(w in t for w in ["can't play", "cannot play", "won't play", "not playing", "stopped playing", "playback", "buffer", "buffering", "skip", "skipping"]):
        return "playback_issue"
    if any(w in t for w in ["crash", "crashes", "crashed", "freeze", "freezes", "frozen", "bug", "error", "not working", "doesn't work", "app"]):
        return "app_or_technical_issue"
    if any(w in t for w in ["song", "album", "playlist", "artist", "music", "track", "catalogue", "catalog", "available"]):
        return "music_or_content_request"
    if any(w in t for w in ["how do", "how can", "what is", "feature", "option", "setting", "work"]):
        return "feature_or_product_question"
    return "other_or_unclear"

print("Loading data...")
golden = pd.read_csv(GOLDEN)
golden["message"] = golden["message"].fillna("").astype(str)
golden["intent"] = golden["intent"].fillna("").astype(str).str.strip()
golden = golden[golden["intent"].isin(INTENTS)].copy()

X = golden["message"].values
y = golden["intent"].values

# Ensure >=5 examples per class
class_counts = pd.Series(y).value_counts()
if class_counts.min() < 5:
    print("WARNING: Some classes have fewer than 5 examples. Removing them for 5-fold CV to avoid splits crashing.")
    valid_classes = class_counts[class_counts >= 5].index
    mask = pd.Series(y).isin(valid_classes)
    X = X[mask]
    y = y[mask]

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = []
majority_metrics = []
keyword_metrics = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    # 1. Train Model (Pipeline ensures TF-IDF only sees train)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1, max_features=10000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
    ])
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    fold_metrics.append({"accuracy": acc, "macro_f1": f1})
    
    # 2. Majority Baseline
    majority_class = pd.Series(y_train).mode()[0]
    y_pred_maj = [majority_class] * len(y_test)
    maj_acc = accuracy_score(y_test, y_pred_maj)
    maj_f1 = f1_score(y_test, y_pred_maj, average="macro", zero_division=0)
    majority_metrics.append({"accuracy": maj_acc, "macro_f1": maj_f1})
    
    # 3. Keyword Baseline
    y_pred_kw = [keyword_predict(text) for text in X_test]
    kw_acc = accuracy_score(y_test, y_pred_kw)
    kw_f1 = f1_score(y_test, y_pred_kw, average="macro", zero_division=0)
    keyword_metrics.append({"accuracy": kw_acc, "macro_f1": kw_f1})

def summarize(metrics):
    accs = [m["accuracy"] for m in metrics]
    f1s = [m["macro_f1"] for m in metrics]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_macro_f1": float(np.mean(f1s)),
        "std_macro_f1": float(np.std(f1s))
    }

clean_results = {
    "LogisticRegression_TFIDF": summarize(fold_metrics),
    "Keyword_Baseline": summarize(keyword_metrics),
    "Majority_Baseline": summarize(majority_metrics),
}

print("\n========== CLEAN 5-FOLD CV EVALUATION ==========")
print(f"Majority Baseline: Acc = {clean_results['Majority_Baseline']['mean_accuracy']:.4f} ± {clean_results['Majority_Baseline']['std_accuracy']:.4f}")
print(f"Keyword Baseline:  Acc = {clean_results['Keyword_Baseline']['mean_accuracy']:.4f} ± {clean_results['Keyword_Baseline']['std_accuracy']:.4f}")
print(f"CLEAN HEADLINE RESULT (TFIDF+LR):")
print(f"Accuracy = {clean_results['LogisticRegression_TFIDF']['mean_accuracy']:.4f} ± {clean_results['LogisticRegression_TFIDF']['std_accuracy']:.4f}")
print(f"Macro F1 = {clean_results['LogisticRegression_TFIDF']['mean_macro_f1']:.4f} ± {clean_results['LogisticRegression_TFIDF']['std_macro_f1']:.4f}")

Path("reports").mkdir(exist_ok=True)
with open("reports/clean_cv_results.json", "w") as f:
    json.dump(clean_results, f, indent=4)
print("\nResults saved to reports/clean_cv_results.json")
