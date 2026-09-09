import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

INPUT = "data/spotify.csv"

df = pd.read_csv(INPUT)

# Customer messages only
customers = df[df["inbound"] == True].copy()

# Clean text
def clean(text):
    text = str(text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

customers["clean_text"] = customers["text"].apply(clean)

# Remove extremely short messages
customers = customers[customers["clean_text"].str.len() >= 15]

print("Customer messages:", len(customers))

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2),
    min_df=3
)

X = vectorizer.fit_transform(customers["clean_text"])

# Cluster into 8 candidate topics
kmeans = KMeans(
    n_clusters=8,
    random_state=42,
    n_init=10
)

customers["cluster"] = kmeans.fit_predict(X)

terms = vectorizer.get_feature_names_out()

print("\n========== CANDIDATE INTENTS ==========")

for cluster in range(8):
    center = kmeans.cluster_centers_[cluster]
    top_indices = center.argsort()[-15:][::-1]
    keywords = [terms[i] for i in top_indices]

    examples = customers[
        customers["cluster"] == cluster
    ]["clean_text"].sample(
        min(8, (customers["cluster"] == cluster).sum()),
        random_state=42
    )

    print(f"\nCLUSTER {cluster}")
    print("Keywords:", ", ".join(keywords))

    print("Examples:")
    for example in examples:
        print(" -", example)

print("\nDone.")
