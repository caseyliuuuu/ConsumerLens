# ConsumerLens V2 verification

Verified locally on Python 3.11.7 on 2026-09-16.

## Automated validation

- Created a separate temporary Python 3.11.7 environment without modifying the project's existing `.venv`.
- Installed every package in `requirements.txt` successfully. `pip check` reported no broken requirements.
- Compiled the application, analysis modules, evaluation scripts, and tests without syntax errors.
- Ran 12 regression and dashboard tests: all passed in 2.358 seconds.
- Started Streamlit from the clean environment and received `ok` from `/_stcore/health`.
- Inspected the rendered landing page, executive overview, relative priority language, recommendation mode label, Product/Pricing/Marketing cards, and expandable evidence in a browser.
- Ran the 400-row synthetic sample through the preferred path. Preprocessing retained 397 reviews, and the run completed with 12 displayed topic groups using `MiniLM + BERTopic` and the text sentiment model with no fallback warnings.
- Ran the same sample with model downloads disabled. The complete pipeline remained usable through the documented TF-IDF/KMeans and rating-proxy fallbacks.

## Reproducible evaluation results

`python evaluation/evaluate_system.py` regenerated `system_evaluation.json` and `system_evaluation.csv` from the saved semantic demo analysis.

| Measurement | Result |
|---|---:|
| Review-weighted lexical topic cohesion proxy | 0.6226636891 |
| Sentiment coverage | 100.0% |
| Non-outlier topic assignment coverage | 91.9395466% |
| BERTopic outlier rate | 8.0604534% (32 reviews) |
| Recommendation evidence citation rate | 100.0% |
| Deterministic brief reproducibility | Identical across two runs |
| Maximum impact-score recalculation difference | 5.1020299e-11 |
| Completed human validation rows | 0 of 50 |

These measurements are descriptive. Lexical cohesion is not topic accuracy, coverage is not sentiment correctness, and valid citations do not establish recommendation usefulness. The 50 human-validation rows remain intentionally unlabeled, so no accuracy claim is made.

## LLM-path validation

No paid endpoint was called. Automated tests verify the structured evidence payload, exact Product/Pricing/Marketing schema, same-topic citation enforcement, invalid review-ID rejection, numeric/revenue/market-wide claim rejection, timeout fallback, and deterministic fallback when no API key is present.

## Repository audit

- No API keys, private keys, local `.env`, logs, or credential-like values were found outside the ignored `.venv`.
- `.gitignore` excludes virtual environments, caches, model/download caches, local environment files, Streamlit secrets, coverage output, logs, and macOS metadata.
- The demo dataset and dashboard label synthetic data explicitly.
- The README describes the 0–10 score as an investigation-priority index and does not claim revenue, ROI, causal impact, or model accuracy.

## Manual checks remaining

1. Upload a private real-world CSV through the browser and confirm its column mapping and source-specific interpretation.
2. Save each export through the browser and review it in the intended spreadsheet tool.
3. If LLM mode will be shown, configure the intended provider and review a live response under that provider's data policy. No live paid-provider response is claimed here.
4. Complete the 50 human labels before reporting topic or sentiment accuracy.
5. Refresh `assets/dashboard_demo.png` after any later visual customization.
