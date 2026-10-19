# Article Category Suggester

A small tool that suggests a category for a WordPress / tech article title.
Built for the WPArena task *"Article Category Suggester"*.

## What it does
Give it a title, get the top-3 predicted categories with confidence scores.
Categories (5): **WordPress**, **Cybersecurity**, **AI & Machine Learning**, **Web Development**, **SEO & Marketing**.

## Data
- `data/train.csv` – 60 labelled titles (12 per category), written by me as realistic blog-style titles (no scraped or copyrighted data).
- `data/test.csv` – 15 separate, unseen titles (3 per category), never used for training or tuning.

## Model
TF-IDF features (word 1-2 grams + character 3-5 grams) fed into Logistic Regression (scikit-learn). Chosen because it is simple, fast, trains in under a second and is easy to explain.

## Setup
```bash
pip install -r requirements.txt
python evaluate.py                       # train + test on the 15 unseen titles
python classifier.py "Your title here"   # command-line prediction
streamlit run app.py                     # web UI
```

## How it was tested
Trained on the 60 titles, evaluated once on the 15 held-out titles.

**Result: 15/15 correct (100% accuracy).**

Please read this number carefully: with only 15 test titles the estimate is very noisy, and the titles contain fairly clear keywords. Real-world accuracy will be lower.

## Limitations
- Tiny dataset (60 titles); the model mostly learns keywords.
- Overlapping topics are ambiguous, e.g. "How to Use ChatGPT for SEO" scores AI/ML 28% vs SEO 25%; "Best AI tools for writing blog posts" leans to SEO & Marketing.
- Out-of-scope titles (e.g. "Top 10 gadgets of 2026") still get a forced guess with low confidence (~23%); there is no "unknown" class.
- English only; titles only, not article body.
- Next steps: collect several hundred more titles, add a confidence threshold for "unsure", try sentence embeddings.

## Deploy (free)
Push to GitHub, then create an app on [Streamlit Community Cloud](https://streamlit.io/cloud) pointing at `app.py`. No secrets or API keys are used.
