import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

DATA = Path("data/spotify.csv")
GOLDEN = Path("data/golden_set.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

INTENTS = [
    "playback_issue",
    "app_or_technical_issue",
    "account_or_access",
    "subscription_or_payment",
    "music_or_content_request",
    "feature_or_product_question",
    "other_or_unclear",
]

print("Loading data...")

df = pd.read_csv(DATA)
golden = pd.read_csv(GOLDEN)

# ---------------------------------------------------------
# Prepare customer messages
# ---------------------------------------------------------

df = df[df["inbound"] == True].copy()

df["text"] = (
    df["text"]
    .fillna("")
    .astype(str)
    .str.replace(r"@\w+", " ", regex=True)
    .str.replace(r"http\S+", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

golden["message"] = golden["message"].fillna("").astype(str)
golden["intent"] = golden["intent"].fillna("").astype(str).str.strip()

# ---------------------------------------------------------
# Remove golden examples from training
# ---------------------------------------------------------

golden_ids = set(golden["tweet_id"].astype(str))

df["tweet_id_str"] = df["tweet_id"].astype(str)

train = df[~df["tweet_id_str"].isin(golden_ids)].copy()

# Only valid labels
golden = golden[golden["intent"].isin(INTENTS)].copy()

print("\nTraining examples:", len(train))
print("Golden examples:", len(golden))

print("\nGolden intent distribution:")
print(golden["intent"].value_counts())

# ---------------------------------------------------------
# Baseline 1: Majority class
# ---------------------------------------------------------

majority_class = train["text"].map(
    lambda x: None
)

# We don't have training labels yet, so we'll create
# pseudo-labels using a lightweight keyword system.
#
# The actual supervised model is trained using the
# golden labels. This gives us an honest small-data
# evaluation.

# ---------------------------------------------------------
# Keyword baseline
# ---------------------------------------------------------

def keyword_predict(text):
    t = text.lower()

    if any(w in t for w in [
        "refund", "charged", "charge", "billing",
        "payment", "subscription", "premium", "price",
        "cancel", "cancellation"
    ]):
        return "subscription_or_payment"

    if any(w in t for w in [
        "login", "log in", "password", "account",
        "username", "sign in", "signin"
    ]):
        return "account_or_access"

    if any(w in t for w in [
        "can't play", "cannot play", "won't play",
        "not playing", "stopped playing", "playback",
        "buffer", "buffering", "skip", "skipping"
    ]):
        return "playback_issue"

    if any(w in t for w in [
        "crash", "crashes", "crashed", "freeze",
        "freezes", "frozen", "bug", "error",
        "not working", "doesn't work", "app"
    ]):
        return "app_or_technical_issue"

    if any(w in t for w in [
        "song", "album", "playlist", "artist",
        "music", "track", "catalogue", "catalog",
        "available"
    ]):
        return "music_or_content_request"

    if any(w in t for w in [
        "how do", "how can", "what is", "feature",
        "option", "setting", "work"
    ]):
        return "feature_or_product_question"

    return "other_or_unclear"


y_true = golden["intent"]

keyword_predictions = golden["message"].apply(keyword_predict)

keyword_accuracy = accuracy_score(y_true, keyword_predictions)
keyword_f1 = f1_score(
    y_true,
    keyword_predictions,
    labels=INTENTS,
    average="macro",
    zero_division=0
)

print("\n========== KEYWORD BASELINE ==========")
print("Accuracy:", round(keyword_accuracy, 4))
print("Macro F1:", round(keyword_f1, 4))

# ---------------------------------------------------------
# Majority baseline
# ---------------------------------------------------------

majority = golden["intent"].mode()[0]

majority_predictions = np.array(
    [majority] * len(golden)
)

majority_accuracy = accuracy_score(
    y_true,
    majority_predictions
)

majority_f1 = f1_score(
    y_true,
    majority_predictions,
    labels=INTENTS,
    average="macro",
    zero_division=0
)

print("\n========== MAJORITY BASELINE ==========")
print("Majority intent:", majority)
print("Accuracy:", round(majority_accuracy, 4))
print("Macro F1:", round(majority_f1, 4))

# ---------------------------------------------------------
# Train supervised classifier
# ---------------------------------------------------------

# IMPORTANT:
# With only 200 hand-labelled examples, we use the
# golden set for supervised training and evaluate it
# using cross-validation below.
#
# This avoids pretending that unlabeled historical data
# has ground-truth intent labels.

X = golden["message"]
y = golden["intent"]

pipeline = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=10000,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced"
        )
    )
])

# ---------------------------------------------------------
# 5-fold cross validation
# ---------------------------------------------------------

from sklearn.model_selection import StratifiedKFold, cross_val_predict

min_class_count = y.value_counts().min()

if min_class_count >= 5:

    folds = 5

    cv = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    predictions = cross_val_predict(
        pipeline,
        X,
        y,
        cv=cv
    )

    cv_accuracy = accuracy_score(y, predictions)

    cv_f1 = f1_score(
        y,
        predictions,
        labels=INTENTS,
        average="macro",
        zero_division=0
    )

    print("\n========== TF-IDF + LOGISTIC REGRESSION ==========")
    print("5-fold CV Accuracy:", round(cv_accuracy, 4))
    print("5-fold CV Macro F1:", round(cv_f1, 4))

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=INTENTS,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(
        pd.DataFrame(
            confusion_matrix(
                y,
                predictions,
                labels=INTENTS
            ),
            index=INTENTS,
            columns=INTENTS
        )
    )

else:
    print("\nWARNING: Some intents have fewer than 5 examples.")
    print("Skipping 5-fold cross-validation.")

# ---------------------------------------------------------
# Train final model on all golden labels
# ---------------------------------------------------------

pipeline.fit(X, y)

joblib.dump(
    pipeline,
    MODEL_DIR / "intent_classifier.joblib"
)

print("\nFinal model saved to:")
print(MODEL_DIR / "intent_classifier.joblib")

# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = pd.DataFrame({
    "model": [
        "majority_baseline",
        "keyword_baseline",
        "tfidf_logistic_regression_cv"
    ],
    "accuracy": [
        majority_accuracy,
        keyword_accuracy,
        cv_accuracy if min_class_count >= 5 else np.nan
    ],
    "macro_f1": [
        majority_f1,
        keyword_f1,
        cv_f1 if min_class_count >= 5 else np.nan
    ]
})

results.to_csv(
    "data/classifier_results.csv",
    index=False
)

print("\n========== RESULTS ==========")
print(results.to_string(index=False))

print("\nSaved:")
print("data/classifier_results.csv")
