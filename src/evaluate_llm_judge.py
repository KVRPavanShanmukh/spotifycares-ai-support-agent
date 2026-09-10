import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
import json
import time
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from pydantic import BaseModel, Field

# Ensure src modules can be imported
sys.path.append(str(Path(".").absolute()))
from src.generate_reply import generate_reply

class JudgeResult(BaseModel):
    relevance: int = Field(description="Score 1-5")
    grounding: int = Field(description="Score 1-5")
    actionability: int = Field(description="Score 1-5")
    safety: int = Field(description="Score 1-5")
    tone: int = Field(description="Score 1-5")
    reasoning: str = Field(description="Brief explanation for the scores")

def evaluate_llm_judge():
    print("========== GEMINI LLM-AS-JUDGE EVALUATION ==========")
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("LLM judge skipped: GEMINI_API_KEY environment variable not configured.")
        print("To run the Gemini LLM judge, please export GEMINI_API_KEY and rerun this script.")
        return

    print("GEMINI_API_KEY found. Initializing Gemini client...")
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key)
    except ImportError:
        print("Error: google-genai library not installed. Please 'pip install google-genai'.")
        return

    sample_file = Path("data/human_review_sample.csv")
    if not sample_file.exists():
        print("Sample file not found.")
        return
        
    df = pd.read_csv(sample_file).head(20)
    
    results = []
    for idx, row in df.iterrows():
        # Re-run generation to get the escalation info not saved in the sample CSV
        reply_res = generate_reply(row['message'], exclude_tweet_id=row['tweet_id'])
        escalation_decision = reply_res.get('escalate', 'Unknown')
        escalation_reason = reply_res.get('escalation_reason', 'None')

        prompt = f"""
You are an expert customer service evaluator. 
Evaluate the following AI agent reply based on the customer message, retrieved historical evidence, and escalation decision.

Historical evidence shows how the brand previously handled similar issues, but is NOT automatically current truth.
Penalize unsupported current facts, invented policies, invented URLs, dates, or artist details.
Reward useful next steps.
Reward safe and appropriate escalation when needed. Penalize failure to escalate sensitive issues.
Penalize generic replies that do not address the customer's issue.

Customer Message: {row['message']}
Retrieved Evidence (Historical): {row['retrieved_evidence']}
Agent Escalation Decision: {escalation_decision}
Agent Escalation Reason: {escalation_reason}
Agent Reply: {row['agent_reply']}

Score the reply from 1 to 5 on these dimensions:
1. Relevance: Does it address the customer's actual issue?
2. Grounding: Is it supported by the historical evidence without inventing or leaking outdated facts?
3. Actionability: Does it provide a concrete next step or useful question?
4. Safety: Is it professional and safe? Did it escalate appropriately?
5. Tone: Is it empathetic and on-brand?
"""
        
        judge_status = "Success"
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=JudgeResult,
                ),
            )
            parsed = json.loads(response.text)
            print(f"Evaluated {idx+1}/{len(df)} (Tweet ID: {row['tweet_id']})")
            
            # Simple rate limiting for free tier
            time.sleep(2)
        except Exception as e:
            print(f"Error evaluating row {idx}: {e}")
            judge_status = "Error"
            parsed = {}
            
        relevance = parsed.get("relevance", 0)
        grounding = parsed.get("grounding", 0)
        actionability = parsed.get("actionability", 0)
        safety = parsed.get("safety", 0)
        tone = parsed.get("tone", 0)
        
        try:
            overall_score = (int(relevance) + int(grounding) + int(actionability) + int(safety) + int(tone)) / 5
        except ValueError:
            overall_score = 0
            
        results.append({
            "tweet_id": row["tweet_id"],
            "message": row["message"],
            "reply": row["agent_reply"],
            "relevance": relevance,
            "grounding": grounding,
            "actionability": actionability,
            "safety": safety,
            "tone": tone,
            "overall_score": overall_score,
            "judge_model": "gemini-2.5-flash",
            "judge_status": judge_status
        })
        
    print("\nLLM Judge Execution Complete.")
    out_df = pd.DataFrame(results)
    out_df.to_csv("data/llm_judge_results.csv", index=False)
    print("Saved detailed judge results to data/llm_judge_results.csv")
    
    # Summary
    valid = out_df[out_df["judge_status"] == "Success"]
    if len(valid) > 0:
        print("\n--- LLM Judge Summary ---")
        print(f"Relevance: {pd.to_numeric(valid['relevance'], errors='coerce').mean():.2f}")
        print(f"Grounding: {pd.to_numeric(valid['grounding'], errors='coerce').mean():.2f}")
        print(f"Actionability: {pd.to_numeric(valid['actionability'], errors='coerce').mean():.2f}")
        print(f"Safety: {pd.to_numeric(valid['safety'], errors='coerce').mean():.2f}")
        print(f"Tone: {pd.to_numeric(valid['tone'], errors='coerce').mean():.2f}")
        print(f"Overall Score: {pd.to_numeric(valid['overall_score'], errors='coerce').mean():.2f}")
    
    print("\n========== HUMAN AGREEMENT STUDY ==========")
    if sample_file.exists():
        df = pd.read_csv(sample_file)
        human_scores_col = pd.to_numeric(df["human_score"], errors="coerce")
        valid_scores = human_scores_col.notna()
        
        if valid_scores.sum() > 0:
            print(f"Found {valid_scores.sum()} genuinely reviewed examples.")
            
            llm_res_file = Path("data/llm_judge_results.csv")
            if llm_res_file.exists():
                llm_df = pd.read_csv(llm_res_file)
                df_merged = df.merge(llm_df, on="tweet_id", how="inner")
                h_score = pd.to_numeric(df_merged["human_score"], errors="coerce")
                l_score = pd.to_numeric(df_merged["overall_score"], errors="coerce")
                
                mask = h_score.notna() & l_score.notna()
                hs = h_score[mask]
                ls = l_score[mask]
                
                if len(hs) > 1:
                    print(f"Comparison N: {len(hs)}")
                    print(f"Mean Human Score: {hs.mean():.2f}")
                    print(f"Mean LLM Judge Score: {ls.mean():.2f}")
                    
                    mean_abs_diff = (hs - ls).abs().mean()
                    print(f"Mean Absolute Difference: {mean_abs_diff:.2f}")
                    
                    exact_match = (hs.round() == ls.round()).mean()
                    print(f"Exact Agreement (rounded): {exact_match:.2f}")
                    
                    if len(hs) >= 3:
                        corr_p, p_val_p = pearsonr(hs, ls)
                        print(f"Pearson Correlation: {corr_p:.3f} (p={p_val_p:.3f})")
                        corr_s, p_val_s = spearmanr(hs, ls)
                        print(f"Spearman Correlation: {corr_s:.3f} (p={p_val_s:.3f})")
                else:
                    print("Not enough paired scores for LLM vs Human correlation.")
            else:
                print("LLM Judge results not found. Run the judge to compare with human scores.")
        else:
            print("Human scores not yet available; judge-human agreement not calculated.")
            print("\nTo conduct the manual review:")
            print("1. Open data/human_review_sample.csv")
            print("2. For each row, provide a 'human_score' from 1-5 (1=Poor, 5=Excellent).")
            print("3. Consider Relevance, Grounding, Actionability, Safety, and Tone.")
            print("4. Add brief 'human_notes'.")
            print("5. Save and rerun this evaluation.")
    else:
        print("Sample file not found.")

if __name__ == "__main__":
    evaluate_llm_judge()

