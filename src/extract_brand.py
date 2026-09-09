import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from pathlib import Path

def extract_brand_conversations(df, brand):
    """
    Extracts customer-support conversation pairs for the given brand.
    Returns a DataFrame with ['customer_text', 'support_text'].
    """
    support = df[df["author_id"] == brand].copy()
    
    if len(support) == 0:
        return pd.DataFrame(), 0, 0, 0
    
    # IDs of support tweets
    support_ids = set(support["tweet_id"].astype(str))
    
    # Customer tweets directly responding to the brand
    # Use Int64 to avoid float conversion (e.g. 123.0 -> '123.0')
    customer = df[
        df["in_response_to_tweet_id"]
        .astype("Int64")
        .astype(str)
        .isin(support_ids)
    ].copy()
    
    # Create matching strings to merge on
    support["tweet_id_str"] = support["tweet_id"].astype(str)
    customer["in_response_to_str"] = customer["in_response_to_tweet_id"].astype("Int64").astype(str)
    
    merged = pd.merge(
        customer,
        support,
        left_on="in_response_to_str",
        right_on="tweet_id_str",
        suffixes=("_cust", "_supp")
    )
    
    pairs_df = pd.DataFrame({
        "customer_text": merged["text_cust"],
        "support_text": merged["text_supp"],
        "support_author_id": merged["author_id_supp"]
    })
    
    return pairs_df, len(df), len(support), len(customer)

if __name__ == "__main__":
    INPUT = Path("data/twcs/twcs.csv")
    OUTPUT = Path("data/spotify.csv")
    
    print("Loading dataset...")
    df = pd.read_csv(INPUT)
    
    pairs_df, total_rows, support_rows, customer_rows = extract_brand_conversations(df, "SpotifyCares")
    
    print("\n=== FINAL SPOTIFY DATASET ===")
    print("Total tweets:", total_rows)
    print("SpotifyCares tweets:", support_rows)
    print("Customer replies:", customer_rows)
    print("Conversation pairs:", len(pairs_df))
    
    # Re-create spotify.csv format to avoid breaking old scripts
    support = df[df["author_id"] == "SpotifyCares"].copy()
    support_ids = set(support["tweet_id"].astype(str))
    customer = df[df["in_response_to_tweet_id"].astype("Int64").astype(str).isin(support_ids)].copy()
    spotify = pd.concat([support, customer], ignore_index=True)
    spotify = spotify.drop_duplicates(subset="tweet_id")
    spotify["created_at_dt"] = pd.to_datetime(spotify["created_at"], errors="coerce", utc=True)
    spotify = spotify.sort_values("created_at_dt").drop(columns=["created_at_dt"])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    spotify.to_csv(OUTPUT, index=False)
    print(f"Saved to: {OUTPUT}")
