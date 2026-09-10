# SpotifyCares AI Support Agent

## 1. Problem

The goal of this project is to build an automated AI support agent for the SpotifyCares Twitter handle. The agent must perform four core tasks:
- **Classify customer intent:** Understand what the customer needs based on their message.
- **Retrieve historically similar SpotifyCares interactions:** Ground responses in real past resolutions.
- **Draft a historically grounded response:** Generate a safe, relevant reply without inventing arbitrary policies.
- **Decide whether to escalate:** Identify sensitive or ambiguous cases that require human intervention.

## 2. What I Built

The project is built around a transparent, deterministic pipeline:

```text
Customer message
       ↓
Intent classifier
       ↓
Historical retrieval
       ↓
Evidence
       ↓
Reply generation
       ↓
Escalation decision
       ↓
Final response
```

**Implementation Details:**
- **Intent Classification:** TF-IDF + Logistic Regression
- **Retrieval:** TF-IDF with Cosine Similarity to find the top 3 historical matches.
- **Reply Generation:** Deterministic fallback logic that leverages the retrieved historical responses, stripping out internal initials and URLs.
- **Escalation:** Rule-based decision logic ensuring sensitive intents (e.g. billing) and low-confidence retrievals are immediately escalated.

## 3. Dataset

The agent was built using the Customer Support on Twitter dataset.

- Approximately 2.8M tweets exist in the downloaded dataset.
- SpotifyCares was selected as the single brand.
- Extracted 43,265 SpotifyCares support tweets.
- Built 15,096 customer-response conversation pairs.
- Created 200 hand-labelled golden examples for honest evaluation.

*(Note: the full dataset was not required for every experiment. The pipeline isolates and operates entirely on the SpotifyCares subset.)*

## 4. Intent Taxonomy

Customer messages were mapped to 7 distinct intents:

- `playback_issue`
- `app_or_technical_issue`
- `account_or_access`
- `subscription_or_payment`
- `music_or_content_request`
- `feature_or_product_question`
- `other_or_unclear`

*Note: Device and OS information (e.g. "iPhone 6", "iOS 11") were explicitly treated as metadata rather than an intent, as they describe the context of a technical or playback issue rather than the issue itself.*

## 5. Evaluation

The pipeline was evaluated against two simple baselines using the 200 golden examples, and now features a **clean 5-fold Stratified Cross-Validation** to prevent data leakage.

**Majority Baseline (5-fold):**
46.19% (±0.79%) accuracy 

**Keyword Baseline (5-fold):**
62.45% (±4.24%) accuracy

**Clean HEADLINE RESULT: TF-IDF + Logistic Regression (5-fold CV):**
67.46% (±5.44%) accuracy / 0.5301 (±0.0586) Macro F1

*(The previously reported 94% accuracy and 99.5% retrieval similarity results from the final harness are NOT used as headline results because the same golden examples were used to train the final classifier and populate the retrieval index, causing massive data leakage. The new retrieval evaluation explicitly excludes the tested customer tweet from the index to prevent exact self-retrieval.)*

## 6. What Is Misleading About My Headline Number?

It is crucial to acknowledge the limitations in the evaluation harness results:

- **Original Contamination:** The previous 94% was not a held-out test score. The golden examples were reused in training the final classifier. We have now fixed this using 5-fold CV.
- **Self-Retrieval Fixed:** The previous 99.5% retrieval similarity was completely invalid due to exact self-retrieval. The new leakage-safe retrieval evaluation yields a much more realistic **Mean top-1 similarity of 0.5060**.
- **Data Imbalance:** The golden set is small and heavily imbalanced.
- **Metric Limitations:** The deterministic reply quality rubric is a rigid proxy and is not equivalent to human or LLM-judge judgements. An LLM judge rubric is implemented but pending an API key to execute.

**The defensible classifier headline result is the clean 5-fold CV result of 67.46% accuracy / 0.5301 Macro F1.**

## 7. Escalation Results

- **Accuracy:** 53.5%
- **Precision:** 18.4%
- **Recall:** 100%
- **F1:** 31.1%

The policy deliberately favored recall over precision. The agent guarantees that ambiguous or sensitive account matters are escalated, resulting in massive over-escalation but ensuring safe failure modes.

## 8. Reply Quality

Using an internal deterministic rubric:
- **Relevance:** 1.00
- **Grounding:** 0.99
- **Actionability:** 0.42
- **Safety:** 1.00
- **Tone:** 1.00
- **Overall:** 0.88

*(This is an internal deterministic rubric, NOT a human or LLM-judge score.)*

Actionability was the weakest dimension, as the fallback generation often produces safe but generic replies instead of providing concrete next steps.

## 9. Top 5 Failure Modes

1. **Retrieval finds lexically similar but operationally different cases.**
   *Actual example:* Customer: "A song page loads, such as Wolves and then... nothing." retrieved "A white page loads..."
   *Why it failed:* TF-IDF relies on keyword overlap without semantic understanding.
   *Proposed fix:* Upgrade retrieval to dense semantic embeddings.

2. **Specific historical facts can leak into replies.**
   *Actual example:* A content request retrieved a historical response about "Reputation" by Taylor Swift, which is unsafe for a current generic content request.
   *Why it failed:* The deterministic reply logic sometimes grabs highly specific historical context.
   *Proposed fix:* Add an LLM generation step strictly prompted to abstract specific entities.

3. **Intent imbalance (Keyword override confusion).**
   *Actual example:* Customer: "Different issues: occasionally when I try to play a playlist I have, it either won't play or tells me to get premium."
   *Why it failed:* The system intended `playback_issue` but hardcoded rules override to `subscription_or_payment` because of the word "premium".
   *Proposed fix:* Trust the ML model's confidence or use an LLM for multi-intent handling.

4. **Over-escalation for ambiguous/other messages.**
   *Actual example:* AmazonHelp Demo Customer: "Where is my package? It says delivered but I don't have it."
   *Why it failed:* System categorized as `other_or_unclear` and immediately escalated with reason "Message appears genuinely ambiguous".
   *Proposed fix:* Implement a confidence threshold on the classifier before defaulting to an escalation route, and expand the taxonomy for other brands.

5. **Generic replies can have low actionability.**
   *Actual example:* Customer says the web app is BAD. The agent replies with a safe but generic: "We're looking into this! Could you send us a DM with more details about the issue so we can help?"
   *Why it failed:* The deterministic fallback produces safe but generic replies instead of providing concrete next steps when the top evidence is low similarity.
   *Proposed fix:* Use an LLM to generate actionable steps grounded in the retrieved evidence.

## 10. Non-obvious Decisions

1. **Selected SpotifyCares** because it had enough high-quality support interactions.
2. **Used a single brand** rather than mixing brands to maintain consistent support policies.
3. **Used seven intents** instead of raw unsupervised clusters to align with actionable support domains.
4. **Treated device/OS/version as metadata** rather than unique intents.
5. **Included `other_or_unclear`** to avoid forced classification of banter or thank-yous.
6. **Used TF-IDF** for interpretability, speed, and immediate feedback.
7. **Used Logistic Regression** as a lightweight, explainable baseline classifier.
8. **Used cosine similarity** for straightforward retrieval scoring.
9. **Retrieved three historical examples** to provide robust grounding context.
10. **Avoided fine-tuning** because of time/data costs within the scope of this assignment.
11. **Avoided a frontend** because rigorous evaluation mattered more than a UI.
12. **Used deterministic reply generation** as a fallback.
13. **Escalated payment/account-sensitive cases conservatively** to prioritize safety.
14. **Removed Twitter handles and internal metadata** from customer replies.
15. **Explicitly disclosed evaluation leakage** rather than reporting inflated numbers as production performance.
16. **Built a free-tier LLM-as-judge integration** using Google Gemini 2.5 Flash via the `google-genai` SDK and the `GEMINI_API_KEY` environment variable, ensuring zero cost and avoiding API keys committed to the repository.

## 11. What I Did Not Build

To prioritize measurable proof and robust evaluation over feature bloat, the following were intentionally excluded:
- No frontend
- No deployment
- No fine-tuning
- No full-dataset training
- No multi-agent framework
- No production monitoring
- No live Spotify integration
- No real account/payment access

## 12. One-Week Next Steps

- **Day 1:** Create a clean train/dev/test split.
- **Day 2:** Improve intent labeling and class balance.
- **Day 3:** Replace TF-IDF retrieval with dense embeddings.
- **Day 4:** Improve grounded generation with an LLM.
- **Day 5:** Improve escalation calibration.
- **Day 6:** Expand human evaluation and measure judge-human agreement.
- **Day 7:** Error analysis, monitoring, and deployment preparation.

## 13. Reproduction

A fresh clone of this repository contains the prepared subset (`data/spotify.csv`) to easily reproduce the headline classifier result without downloading the full 2.8M-row raw dataset.

```bash
pip install -r requirements.txt
python src/train_classifier.py
```

- **Expected headline result:** 67.46% Accuracy / 0.5301 Macro F1 from 5-fold Stratified Cross-Validation.

**Running the LLM-as-Judge:**
The evaluation harness uses Google Gemini 2.5 Flash on the free tier to evaluate reply quality without incurring costs.
To use the judge:
1. Obtain a free Gemini API key from Google AI Studio.
2. Set the environment variable: `$env:GEMINI_API_KEY="your-key-here"` (in PowerShell) or `export GEMINI_API_KEY="your-key-here"` (in bash).
3. Run the evaluation script: `python src/evaluate_llm_judge.py`

*(Note: Human-review agreement requires genuine human scores in `data/human_review_sample.csv`. The script will output "Human scores not yet available" if they are blank.)*

The full raw Kaggle dataset is only needed if you wish to recreate `spotify.csv` from scratch (`inspect_data.py` and `extract_brand.py`). Do not attempt to run the downstream full agent evaluation (`generate_reply.py` or `evaluate_agent.py`) until you have actually generated the required models/retriever files.

## 14. Repository Structure

```text
Hiver/
├── data/
│   ├── archive.zip
│   ├── sample.csv
│   ├── spotify.csv                  # The extracted SpotifyCares dataset
│   ├── golden_set.csv               # The 200 hand-labelled evaluation queries
│   └── evaluation_*.csv             # Evaluation harness metrics and samples
├── models/
│   ├── intent_classifier.joblib     # Pickled TF-IDF + Logistic Regression model
│   └── retriever.joblib             # Pickled TF-IDF retrieval index
├── src/
│   ├── inspect_data.py              # Initial dataset exploration
│   ├── extract_brand.py             # Script to extract SpotifyCares interactions
│   ├── discover_intents.py          # Unsupervised clustering for intent discovery
│   ├── make_golden_set.py           # Evaluation set sampling script
│   ├── train_classifier.py          # Supervised intent classifier training
│   ├── build_retriever.py           # Retrieval index generation
│   ├── generate_reply.py            # The core agent pipeline (Routing & Fallback generation)
│   └── evaluate_agent.py            # Evaluation harness
├── README.md                        # Project documentation
└── requirements.txt                 # Dependencies
```

## 15. Limitations

- **Small Golden Set:** 200 examples is too small to capture all edge cases.
- **Class Imbalance:** Natural frequency causes severe minority class starvation.
- **Evaluation Leakage:** The final harness evaluates the classifier/retriever against data present in its training/index.
- **Deterministic Reply Evaluation:** Rule-based heuristics are used instead of proper human/LLM-judge evaluation.
- **No External LLM:** The pipeline currently uses deterministic generation fallbacks.
- **No Production Deployment.**

This project prioritizes transparent evidence and rigorous failure analysis over inflated benchmark numbers.
