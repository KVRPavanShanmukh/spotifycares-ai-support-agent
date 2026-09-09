import sys
sys.stdout.reconfigure(encoding='utf-8')
import argparse
import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import from src
sys.path.append(str(Path(".").absolute()))
from src.extract_brand import extract_brand_conversations

TWCS_DATA = Path("data/twcs/twcs.csv")
MODEL_DIR = Path("models")

def clean(text):
    text = str(text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def contains_case_specific_facts(text):
    if re.search(r'(https?://|www\.)', text): return True
    if re.search(r'\d+', text): return True
    if "reputation" in text.lower() or "taylor swift" in text.lower(): return True
    return False

def format_reply(text):
    text = re.sub(r"[\/\-\^]\s*[a-zA-Z]{1,3}$", "", text)
    return clean(text)

class BrandSupportAgent:
    def __init__(self, brand):
        self.brand = brand
        self.retriever_data = None
        self.classifier = None
        
    def setup(self):
        print(f"[{self.brand}] Loading generic intent classifier...")
        try:
            self.classifier = joblib.load(MODEL_DIR / "intent_classifier.joblib")
        except Exception as e:
            print(f"Error loading intent classifier: {e}")
            sys.exit(1)
            
        print(f"[{self.brand}] Reading full TWCS dataset (this may take a moment)...")
        if not TWCS_DATA.exists():
            print(f"Error: Could not find raw dataset at {TWCS_DATA}")
            sys.exit(1)
            
        df = pd.read_csv(TWCS_DATA)
        
        print(f"[{self.brand}] Filtering brand conversations...")
        pairs_df, total_rows, support_rows, customer_rows = extract_brand_conversations(df, self.brand)
        
        print(f"\nBrand:\n{self.brand}")
        print(f"\nTotal full-dataset rows:\n{total_rows}")
        print(f"\nRows belonging to brand/support author:\n{support_rows}")
        print(f"\nInbound customer tweets:\n{customer_rows}")
        
        if len(pairs_df) > 0:
            pairs_df["customer_text"] = pairs_df["customer_text"].fillna("").apply(clean)
            pairs_df["support_text"] = pairs_df["support_text"].fillna("").apply(clean)
            pairs_df = pairs_df[pairs_df["customer_text"].str.len() >= 15]
            pairs_df = pairs_df.drop_duplicates(subset=["customer_text"]).reset_index(drop=True)
            
        print(f"\nCustomer/support pairs:\n{len(pairs_df)}")
        
        if len(pairs_df) == 0:
            print(f"\nInvalid Brand Handling: Brand '{self.brand}' was not found or has no valid customer-support pairs.")
            print("The historical retriever cannot be built for this brand.")
            sys.exit(0)
            
        print(f"[{self.brand}] Building brand-specific retriever...")
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2)
        tfidf_matrix = vectorizer.fit_transform(pairs_df["customer_text"])
        
        self.retriever_data = {
            "vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "pairs_df": pairs_df
        }
        print(f"[{self.brand}] Initialization complete.\n")

    def generate_reply(self, message):
        clean_msg = clean(message)
        clean_lower = clean_msg.lower()
        
        intent = self.classifier.predict([clean_msg])[0]
        
        sub_keywords = ["subscription", "premium", "charged", "charge", "billing", "payment", "paid", "refund", "cancel subscription", "renewal", "renew", "student discount", "cancel"]
        if any(kw in clean_lower for kw in sub_keywords):
            intent = "subscription_or_payment"
            
        ack_keywords = ["thanks", "thank you", "thx", "appreciate", "much appreciated", "sent", "just sent", "okay", "ok", "got it", "solved", "worked", "it works"]
        is_ack = False
        clean_stripped = clean_lower.strip('!. ')
        if len(clean_stripped.split()) <= 6 and any(kw in clean_stripped for kw in ack_keywords):
            intent = "other_or_unclear"
            is_ack = True

        query_vec = self.retriever_data["vectorizer"].transform([clean_msg])
        sim_scores = cosine_similarity(query_vec, self.retriever_data["tfidf_matrix"]).flatten()
        top_indices = sim_scores.argsort()[-3:][::-1]
        
        evidence = []
        for idx in top_indices:
            author = self.retriever_data["pairs_df"].iloc[idx]["support_author_id"]
            if author != self.brand:
                raise ValueError(f"Brand isolation mismatch! Retrieved example authored by {author}, expected {self.brand}")
            evidence.append({
                "similarity": sim_scores[idx],
                "customer_message": self.retriever_data["pairs_df"].iloc[idx]["customer_text"],
                "support_response": self.retriever_data["pairs_df"].iloc[idx]["support_text"]
            })
            
        top_sim = evidence[0]["similarity"] if evidence else 0.0
        
        raw_response = evidence[0]["support_response"] if evidence else ""
        draft_reply = format_reply(raw_response)
        
        escalate = "No"
        reason = "None"
        
        if is_ack:
            draft_reply = "You're welcome! Let us know if you need anything else."
            escalate = "No"
        else:
            if top_sim < 0.35:
                escalate = "Yes"
                reason = "Low retrieval similarity (evidence is weak)."
                draft_reply = "I'm not completely sure how to help with this. Let me transfer you to a human agent who can investigate further."
            elif intent in ["account_or_access", "subscription_or_payment"]:
                escalate = "Yes"
                reason = f"Intent '{intent}' requires secure handling or human judgement."
                draft_reply = f"Since this looks like an {intent.replace('_', ' ')} issue, I'm escalating this to a human specialist to ensure your account details are handled securely."
            elif intent == "other_or_unclear":
                escalate = "Yes"
                reason = "Message appears genuinely ambiguous and requires human judgment."
                draft_reply = "I'm escalating this to a human specialist to assist you further."
            else:
                if contains_case_specific_facts(raw_response):
                    draft_reply = "We're looking into this! Could you send us a DM with more details about the issue so we can help?"
            
        return {
            "intent": intent,
            "reply": draft_reply,
            "evidence": evidence,
            "confidence": top_sim,
            "escalate": escalate,
            "escalation_reason": reason
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the AI Support Agent for a specific brand.")
    parser.add_argument("--brand", type=str, required=True, help="The brand Twitter handle")
    parser.add_argument("--demo", action="store_true", help="Run a demo query and exit")
    args = parser.parse_args()

    agent = BrandSupportAgent(args.brand)
    agent.setup()

    if args.demo:
        print("=" * 80)
        print(f"[{args.brand}] Running Demo Query...")
        
        demo_queries = {
            "SpotifyCares": "My app keeps crashing when playing songs.",
            "AmazonHelp": "Where is my package? It says delivered but I don't have it.",
            "AppleSupport": "My iPhone battery drains too fast after the update."
        }
        q = demo_queries.get(args.brand, "How do I get a refund?")
        
        print(f"\nCustomer Message: {q}")
        res = agent.generate_reply(q)
        print(f"[{args.brand} Agent] INTENT: {res['intent']} (Sim: {res['confidence']:.4f})")
        if res['escalate'] == "Yes":
            print(f"[{args.brand} Agent] ESCALATING to Human: {res['escalation_reason']}")
        print(f"[{args.brand} Agent] REPLY: {res['reply']}")
        print(f"[{args.brand} Agent] EVIDENCE:")
        for i, ev in enumerate(res['evidence']):
            print(f"  {i+1}. Sim: {ev['similarity']:.4f}")
            print(f"     Historical Cust: {ev['customer_message']}")
            print(f"     Historical Supp: {ev['support_response']}")
        
        sys.exit(0)

    print("=" * 80)
    print(f"[{args.brand}] AI Support Agent Ready")
    print("Type your message below (or type 'quit' to exit).")
    print("=" * 80)

    while True:
        try:
            user_input = input("\nCustomer Message: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Exiting...")
                break
            if not user_input.strip():
                continue
                
            res = agent.generate_reply(user_input)
            
            print(f"\n[{args.brand} Agent] INTENT: {res['intent']} (Sim: {res['confidence']:.4f})")
            
            if res['escalate'] == "Yes":
                print(f"[{args.brand} Agent] ESCALATING to Human: {res['escalation_reason']}")
                
            print(f"[{args.brand} Agent] REPLY: {res['reply']}")
            
            print(f"[{args.brand} Agent] EVIDENCE:")
            for i, ev in enumerate(res['evidence']):
                print(f"  {i+1}. Sim: {ev['similarity']:.4f}")
                print(f"     Historical Cust: {ev['customer_message']}")
                print(f"     Historical Supp: {ev['support_response']}")
                
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
