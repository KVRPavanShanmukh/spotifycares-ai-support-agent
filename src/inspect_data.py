import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from pathlib import Path

DATA = Path("data/twcs/twcs.csv")

print("Loading dataset...")
df = pd.read_csv(DATA)

print("\n=== DATASET SHAPE ===")
print(df.shape)

print("\n=== COLUMNS ===")
print(df.columns.tolist())

print("\n=== SAMPLE ROWS ===")
print(df.head(10).to_string())

print("\n=== INBOUND COUNTS ===")
print(df["inbound"].value_counts())

print("\n=== UNIQUE AUTHORS ===")
print("Unique authors:", df["author_id"].nunique())

print("\n=== MOST FREQUENT AUTHORS ===")
print(df["author_id"].value_counts().head(30))

print("\n=== MISSING VALUES ===")
print(df.isnull().sum())

print("\n=== DATE RANGE ===")
print(df["created_at"].min(), "to", df["created_at"].max())
