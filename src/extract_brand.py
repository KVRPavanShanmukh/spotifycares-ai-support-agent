import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from pathlib import Path

INPUT = Path("data/twcs/twcs.csv")
OUTPUT = Path("data/spotify.csv")

print("Loading dataset...")
df = pd.read_csv(INPUT)

print("Total tweets:", len(df))

# Spotify support account
brand = "SpotifyCares"

# Tweets written by SpotifyCares
support = df[df["author_id"] == brand].copy()

print("SpotifyCares tweets:", len(support))

# IDs of SpotifyCares tweets
support_ids = set(support["tweet_id"].astype(str))

# Customer tweets directly responding to SpotifyCares
# in_response_to_tweet_id might be parsed as float, so cast to Int64 first.
customer = df[
    df["in_response_to_tweet_id"]
    .astype("Int64")
    .astype(str)
    .isin(support_ids)
].copy()

print("Customer replies:", len(customer))

# Combine support + customer messages
spotify = pd.concat([support, customer], ignore_index=True)

# Remove duplicates
spotify = spotify.drop_duplicates(subset="tweet_id")

# Sort chronologically
spotify["created_at_dt"] = pd.to_datetime(
    spotify["created_at"],
    errors="coerce",
    utc=True
)

spotify = spotify.sort_values("created_at_dt")

# Remove helper column
spotify = spotify.drop(columns=["created_at_dt"])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
spotify.to_csv(OUTPUT, index=False)

print("\n=== FINAL SPOTIFY DATASET ===")
print("Rows:", len(spotify))
print("Customers:", int(spotify["inbound"].sum()))
print("Support:", int((~spotify["inbound"]).sum()))
print("Saved to:", OUTPUT)

print("\n=== EXAMPLE CUSTOMER MESSAGES ===")
print(
    spotify[spotify["inbound"] == True]["text"]
    .head(20)
    .to_string(index=False)
)
