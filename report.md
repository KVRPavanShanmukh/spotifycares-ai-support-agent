# SpotifyCares AI Support Agent - Project Report

## 1. Project Overview

The **SpotifyCares AI Support Agent** is designed to build an automated support assistant for the SpotifyCares Twitter handle. The core functionalities of the agent include:

- **Intent Classification:** Understands what the customer needs based on their message, mapping it to 7 distinct support intents.
- **Historical Retrieval:** Grounds its responses by finding the top 3 historically similar SpotifyCares interactions using TF-IDF and Cosine Similarity.
- **Drafting Responses:** Uses a deterministic fallback generation to craft safe, relevant replies, while stripping out internal initials and URLs.
- **Escalation Decisions:** Applies rule-based decision logic to escalate sensitive (e.g., billing) or low-confidence interactions, ensuring safety first.

The pipeline is evaluated using a clean 5-fold Stratified Cross-Validation on a subset of data containing 43,265 SpotifyCares tweets to avoid data leakage.

## 2. Recent Updates (What else we did)

In addition to the core features described above, we performed the following updates and fixes:

### LLM Judge Model Migration
- **Model Upgrade:** We upgraded the LLM-as-a-judge system from `gemini-2.5-flash` to the newer `gemini-3.6-flash` model, ensuring compatibility for new users on the Google GenAI platform.
- **Codebase Adjustments:** The evaluation script `src/evaluate_llm_judge.py` was modified to point to `gemini-3.6-flash` in the API calls.
- **Documentation Updates:** The project documentation in `README.md` was accurately updated to reflect the new `Gemini 3.6 Flash` usage.

### Execution & Environment Troubleshooting
- We performed environment checks to ensure the `google-genai` SDK successfully utilizes the `GEMINI_API_KEY` strictly from the environment variables, avoiding any hardcoded keys.
- We tested the integration and identified an environment isolation issue that initially prevented the automated script from executing. This has since been resolved, and the 20-example evaluation script is now successfully executing locally.

## 3. Current Status
The project successfully bridges deterministic machine learning evaluation metrics (TF-IDF + Logistic Regression) with modern LLM capabilities (Gemini 3.6 Flash as a judge). The focus remains on transparent evidence, safe escalation handling, and rigorous failure analysis.
