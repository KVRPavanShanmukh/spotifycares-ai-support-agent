import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path("data/spotify.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

print("Loading data for retrieval...")
df = pd.read_csv(DATA)

# 1. Identify customer tweets
customers = df[df["inbound"] == True].copy()
support = df[df["inbound"] == False].copy()

# Ensure IDs are strings
customers["tweet_id"] = customers["tweet_id"].astype(str)
customers["in_response_to_tweet_id"] = customers["in_response_to_tweet_id"].astype("Int64").astype(str)

support["tweet_id"] = support["tweet_id"].astype(str)
support["in_response_to_tweet_id"] = support["in_response_to_tweet_id"].astype("Int64").astype(str)

# 2. For each customer tweet, use in_response_to_tweet_id to find the corresponding SpotifyCares response tweet.
# Based on instructions, we match Customer's `in_response_to_tweet_id` to Support's `tweet_id`.
# 3. Keep only customer-response pairs where both sides exist.
support_dict = pd.Series(support["text"].values, index=support["tweet_id"]).to_dict()

pairs = []
for _, row in customers.iterrows():
    resp_id = row["in_response_to_tweet_id"]
    if resp_id in support_dict:
        pairs.append({
            "customer_tweet_id": row["tweet_id"],
            "support_tweet_id": resp_id,
            "customer_text": row["text"],
            "support_text": support_dict[resp_id]
        })

pairs_df = pd.DataFrame(pairs)
print(f"Found {len(pairs_df)} customer-response pairs.")

# 4. Clean text by removing URLs, @mentions, and excessive whitespace.
def clean(text):
    text = str(text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

pairs_df["clean_customer_text"] = pairs_df["customer_text"].apply(clean)

# 5. Build a TF-IDF vectorizer over the historical customer messages.
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
tfidf_matrix = vectorizer.fit_transform(pairs_df["clean_customer_text"])

# 8. Save the retrieval index/vectorizer
joblib.dump({
    "vectorizer": vectorizer,
    "tfidf_matrix": tfidf_matrix,
    "pairs_df": pairs_df
}, MODEL_DIR / "retriever.joblib")

print("Retriever saved to models/retriever.joblib\n")

# Global variables for retrieve_similar function
RETRIEVER_DATA = {
    "vectorizer": vectorizer,
    "tfidf_matrix": tfidf_matrix,
    "pairs_df": pairs_df
}

# 6. Use cosine similarity to retrieve the top-k most similar historical customer messages.
def retrieve_similar(query, top_k=3):
    clean_query = clean(query)
    query_vec = RETRIEVER_DATA["vectorizer"].transform([clean_query])
    
    sim_scores = cosine_similarity(query_vec, RETRIEVER_DATA["tfidf_matrix"]).flatten()
    top_indices = sim_scores.argsort()[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            "similarity": sim_scores[idx],
            "customer_message": RETRIEVER_DATA["pairs_df"].iloc[idx]["customer_text"],
            "support_response": RETRIEVER_DATA["pairs_df"].iloc[idx]["support_text"],
            "customer_tweet_id": RETRIEVER_DATA["pairs_df"].iloc[idx]["customer_tweet_id"],
            "support_tweet_id": RETRIEVER_DATA["pairs_df"].iloc[idx]["support_tweet_id"]
        })
    return results

# 10. Add a simple test at the bottom using 3 example customer queries from the dataset.
if __name__ == "__main__":
    test_queries = [
        "Spotify keeps crashing on my iPhone when I try to play a song",
        "I was double charged for premium this month",
        "How do I cancel my subscription?"
    ]

    for q in test_queries:
        print(f"=== TEST QUERY: {q} ===")
        results = retrieve_similar(q, top_k=3)
        for i, res in enumerate(results, 1):
            print(f"Rank {i} (Sim: {res['similarity']:.4f})")
            print(f"Customer [{res['customer_tweet_id']}]: {res['customer_message']}")
            print(f"Support  [{res['support_tweet_id']}]: {res['support_response']}")
            print("-" * 40)
        print("\n")
