import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib
import re
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

MODEL_DIR = Path("models")

# Load models
print("Loading models...")
try:
    classifier = joblib.load(MODEL_DIR / "intent_classifier.joblib")
    retriever_data = joblib.load(MODEL_DIR / "retriever.joblib")
except Exception as e:
    print(f"Error loading models: {e}")
    sys.exit(1)

def clean(text):
    text = str(text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def contains_case_specific_facts(text):
    # Very simple heuristic to detect specific names, dates, urls, quotes, or numbers
    # which implies we shouldn't blindly regurgitate it.
    if re.search(r'(https?://|www\.)', text): return True
    if re.search(r'\d+', text): return True
    if "reputation" in text.lower() or "taylor swift" in text.lower(): return True
    return False

def format_reply(text):
    # Remove agent initials like /RC, -AC, ^JS
    text = re.sub(r"[\/\-\^]\s*[a-zA-Z]{1,3}$", "", text)
    # Clean standard elements
    return clean(text)

def generate_reply(message):
    clean_msg = clean(message)
    clean_lower = clean_msg.lower()
    
    # 1. Predict intent
    intent = classifier.predict([clean_msg])[0]
    
    # 1. Subscription/payment detection override
    sub_keywords = ["subscription", "premium", "charged", "charge", "billing", "payment", "paid", "refund", "cancel subscription", "renewal", "renew", "student discount", "cancel"]
    if any(kw in clean_lower for kw in sub_keywords):
        intent = "subscription_or_payment"
        
    # 2. Acknowledgements
    ack_keywords = ["thanks", "thank you", "thx", "sent", "just sent", "okay", "ok", "got it", "solved", "worked", "it works"]
    is_ack = False
    if clean_lower in ack_keywords or clean_lower.strip('!.') in ack_keywords:
        intent = "other_or_unclear"
        is_ack = True

    # 2. Retrieve top 3
    query_vec = retriever_data["vectorizer"].transform([clean_msg])
    sim_scores = cosine_similarity(query_vec, retriever_data["tfidf_matrix"]).flatten()
    top_indices = sim_scores.argsort()[-3:][::-1]
    
    evidence = []
    for idx in top_indices:
        evidence.append({
            "similarity": sim_scores[idx],
            "customer_message": retriever_data["pairs_df"].iloc[idx]["customer_text"],
            "support_response": retriever_data["pairs_df"].iloc[idx]["support_text"]
        })
        
    top_sim = evidence[0]["similarity"] if evidence else 0.0
    
    # 3/4/5/6. Deterministic fallback reply generation
    raw_response = evidence[0]["support_response"] if evidence else ""
    draft_reply = format_reply(raw_response)
    
    # Escalation Rules
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
            # 4. Historical grounding filter
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
    test_queries = [
        "How do I cancel my subscription?",
        "Thanks!",
        "I was double charged for premium this month",
        "Spotify keeps crashing on my iPhone when I try to play a song",
        "Can you add Taylor Swift's new album?"
    ]
    
    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"CUSTOMER: {q}")
        print(f"{'='*50}")
        
        result = generate_reply(q)
        
        print(f"INTENT: {result['intent']}")
        print(f"ESCALATE: {result['escalate']} ({result['escalation_reason']})")
        print(f"\nDRAFT REPLY:\n{result['reply']}")
        
        print(f"\nTOP EVIDENCE (Sim: {result['evidence'][0]['similarity']:.4f}):")
        print(f"Historical Customer: {result['evidence'][0]['customer_message']}")
        print(f"Historical Support: {result['evidence'][0]['support_response']}")
