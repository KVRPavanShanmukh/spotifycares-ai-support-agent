import pandas as pd
import re

INPUT = "data/spotify.csv"
OUTPUT = "data/golden_set.csv"

df = pd.read_csv(INPUT)

# Customer messages only
df = df[df["inbound"] == True].copy()

# Clean text
def clean(text):
    text = str(text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["message"] = df["text"].apply(clean)

# Remove very short conversational messages
df = df[df["message"].str.len() >= 20]

# Sample 200 messages
golden = df.sample(
    n=200,
    random_state=42
)[["tweet_id", "message"]].copy()

# Empty labels for manual annotation
golden["intent"] = ""
golden["escalate"] = ""
golden["label_notes"] = ""

golden.to_csv(OUTPUT, index=False)

print("Golden set created!")
print("Rows:", len(golden))
print("File:", OUTPUT)
print("\nOpen this file and fill the intent column.")
