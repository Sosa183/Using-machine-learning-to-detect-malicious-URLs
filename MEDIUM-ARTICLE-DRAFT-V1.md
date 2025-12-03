# AI-Powered Phishing URL Detector 

> Teach your laptop to spot sketchy links before you click them.

Phishing links are still one of the easiest ways attackers steal passwords, money, and access to entire organizations. Instead of trying to memorize every “red flag” by hand, this project shows how to build a **machine learning URL detector** that can automatically classify links as **malicious** or **benign**.

This repository is a practical, reproducible implementation of an AI-powered phishing URL detector, built around and inspired by the public GitHub project **“Using machine learning to detect malicious URLs” by Faizan Ahmad (faizann24)**. It is designed to be beginner friendly, realistic, and something you can run and modify in an IDE like **PyCharm**. :contentReference[oaicite:0]{index=0}

You will see:

- How to set up a **basic ML pipeline** for malicious URL detection  
- What kinds of **features** help distinguish “normal” URLs from “phishy” ones  
- How to train and evaluate a **Logistic Regression** baseline  
- How to use the model to **classify your own list of URLs**  
- How this project connects to **modern research** and **industry phishing reports**  
- Where things broke during setup, and how those problems were actually where most of the learning happened  


---

## 1. Project Overview

This project sits at the intersection of **cybersecurity** and **machine learning**. The core question:

> Given only the URL string, can we guess whether it’s malicious before the user clicks?

A realistic AI-powered phishing detection system would consider many signals (sender reputation, email content, landing pages, user behavior, etc.). Here we focus on **one powerful building block**:

> A URL classifier that uses pattern-based features (length, character patterns, tokens) and classical ML algorithms to decide if a link looks suspicious.

### 1.1. Key Ideas

- We operate purely on **URL strings**, not page content.
- We use a **TF-IDF vectorizer** over URL tokens.
- Our baseline model is **Logistic Regression**, with options to plug in other classifiers.
- The project is designed to run in **PyCharm**, using **Python + scikit-learn**.
- The repo structure keeps things clean and modular, so you can experiment and extend it.

---

## 2. Repository Structure

Here is a suggested structure for the repo that this README belongs to:

```text
.
├── data/
│   └── data.csv                  # URL dataset with labels (malicious / benign)
│
├── models/
│   ├── url_vectorizer.pkl        # Saved TF-IDF vectorizer
│   └── url_classifier.pkl        # Saved trained ML model
│
├── notebooks/
│   └── 01_exploration.ipynb      # Optional: data exploration / experiments
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # Config values (paths, parameters)
│   ├── features.py               # URL tokenization & feature extraction
│   ├── train.py                  # Training script
│   └── predict.py                # CLI prediction script for custom URLs
│
├── app/
│   └── server.py                 # Optional: minimal Flask app for a web demo
│
├── requirements.txt              # Python dependencies
└── README.md                     # You are here
You can keep it simple and only use data/, models/, src/, and README.md if you want.

3. Background & Motivation
I chose phishing URL detection because:

Phishing is everywhere. Almost every modern breach report mentions phishing as an initial access vector.

It is practical and easy to visualize: everyone has seen a sketchy login link at some point.

It gives a concrete way to apply machine learning to a real security problem.

The baseline that inspired this repo, “Using machine learning to detect malicious URLs” by Faizan Ahmad, takes a classic ML approach:

Extract URL-based features (length, number of dots, special characters, suspicious tokens, etc.)

Vectorize with TF-IDF

Train with Logistic Regression / Random Forest / etc.

I picked this approach precisely because it is:

Understandable

Reproducible

Easy to run and tweak in PyCharm

My personal goals with this project:

Learn how a phishing URL dataset is structured

See how to turn raw URL strings into features

Train and evaluate at least one baseline model

Build a small prediction script I can point at my own test URLs

Connect the hands-on work to academic research and industry reports on phishing

4. Getting Started
4.1. Prerequisites
You will need:

A computer running Windows, macOS, or Linux

Python 3.x

PyCharm (Community Edition is enough)

Git installed
Or you can download the repository as a ZIP.

4.2. Clone the Repository
bash
Copy code
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
If you forked or renamed the project, use your own repo URL.

4.3. Create a Virtual Environment (Optional but Recommended)
Using venv:

bash
Copy code
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
4.4. Install Dependencies
You can install from requirements.txt:

bash
Copy code
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
If you prefer to install manually, the key packages are:

bash
Copy code
python -m pip install numpy pandas scikit-learn matplotlib flask
Whenever you hit an ImportError, install the missing package the same way:

bash
Copy code
python -m pip install <package-name>
4.5. Open in PyCharm
Open PyCharm.

Click “Open” and select the project folder.

Make sure the interpreter is set to your virtual environment (if you created one).

Let PyCharm index the project.

5. Dataset & Code Tour
5.1. The Dataset: data/data.csv
The dataset contains URLs and their labels, something like:

csv
Copy code
url,label
http://secure-login-paypal.com.verify-details.xyz/login,malicious
https://accounts.google.com/ServiceLogin,benign
https://www.amazon.com/,benign
...
Typical columns:

url: the full URL string

label: malicious or benign (or 1 / 0 in some datasets)

Open the CSV in PyCharm to see the actual structure. This step makes everything less mysterious.

5.2. Feature Extraction: src/features.py
This module handles:

Splitting URLs into tokens using separators like /, ., -, ?, =, _, &

Feeding these tokens into TfidfVectorizer

Keeping the training and inference preprocessing consistent

Example snippet:

python
Copy code
# src/features.py
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
import re

TOKEN_PATTERN = r"[A-Za-z0-9]+"

def tokenize_url(url: str) -> List[str]:
    # Split on common URL separators and keep alphanumeric chunks
    tokens = re.findall(TOKEN_PATTERN, url.lower())
    return tokens

def build_vectorizer():
    return TfidfVectorizer(
        tokenizer=tokenize_url,
        token_pattern=None,      # disable default token pattern
        ngram_range=(1, 2),      # unigrams + bigrams
        min_df=2
    )
5.3. Training Script: src/train.py
This script:

Loads the dataset

Splits into train/test

Builds the TF-IDF vectorizer

Trains a Logistic Regression model

Prints evaluation metrics

Saves the trained model and vectorizer into models/

python
Copy code
# src/train.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

from features import build_vectorizer

DATA_PATH = os.path.join("data", "data.csv")
VECTORIZER_PATH = os.path.join("models", "url_vectorizer.pkl")
MODEL_PATH = os.path.join("models", "url_classifier.pkl")

def main():
    # 1. Load data
    df = pd.read_csv(DATA_PATH)
    X = df["url"].astype(str).values
    y = df["label"].values

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Build vectorizer and fit
    vectorizer = build_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)

    # 4. Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    # 5. Evaluate
    X_test_vec = vectorizer.transform(X_test)
    y_pred = clf.predict(X_test_vec)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    # 6. Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(clf, MODEL_PATH)

    print(f"\nSaved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
Run it with:

bash
Copy code
python -m src.train
(or python src/train.py depending on how your project is set up).

5.4. Prediction Script: src/predict.py
This script loads the saved model and vectorizer and classifies a list of URLs you provide.

python
Copy code
# src/predict.py
import os
import joblib
import numpy as np

from features import tokenize_url  # ensures same tokenizer is used

VECTORIZER_PATH = os.path.join("models", "url_vectorizer.pkl")
MODEL_PATH = os.path.join("models", "url_classifier.pkl")

EXAMPLE_URLS = [
    "https://accounts.google.com/ServiceLogin",
    "http://login-secure-paypal.com.verify-details.xyz/login",
    "https://www.amazon.com/",
]

def load_artifacts():
    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)
    return vectorizer, model

def predict_urls(urls):
    vectorizer, model = load_artifacts()
    X_vec = vectorizer.transform(urls)
    preds = model.predict(X_vec)
    probs = model.predict_proba(X_vec)
    return preds, probs

def main():
    preds, probs = predict_urls(EXAMPLE_URLS)

    for url, label, prob in zip(EXAMPLE_URLS, preds, probs):
        confidence = np.max(prob)
        print(f"{url}")
        print(f"  -> predicted: {label} (confidence: {confidence:.2f})\n")

if __name__ == "__main__":
    main()
Run:

bash
Copy code
python -m src.predict
You should see obviously fake-looking domains labeled as malicious, and common domains like Google or Amazon labeled benign (depending on the training data).

6. How It Works (Under the Hood)
6.1. Feature Engineering
The key trick is that we never fetch the actual page contents. We only look at URL patterns:

URL length

Number of dots, slashes, and special characters

Suspicious substrings like login, secure, verify, update, bank in the wrong places

Token patterns such as paypal appearing as a subdomain of some random domain

These signals are captured by:

Custom URL tokenization with tokenize_url

A TF-IDF vectorizer that learns which tokens/ngrams matter

The classifier then learns which combinations are more likely in malicious vs benign URLs

6.2. Model Choice
This repo uses Logistic Regression as a baseline because:

It is fast and interpretable

It handles high-dimensional sparse TF-IDF features well

It gives a probability that a URL is malicious

You can easily swap in:

RandomForestClassifier

LinearSVC

Or more advanced models later

6.3. Evaluation
The training script prints:

Overall accuracy

Precision, recall, and F1-scores for each class

A basic sense of how often the model catches malicious URLs vs how often it cries wolf

For real security use, minimizing false negatives (missed malicious URLs) often matters more than raw accuracy, which is something you can explore by adjusting class weights and thresholds.

7. Research & Industry Context
To connect this small project to the bigger world:

Haq et al., 2024 present a deep learning approach using 1D-CNN on raw URL strings, skipping manual feature engineering entirely. Their model learns character-level patterns directly and achieves strong results on public datasets.

The Anti-Phishing Working Group (APWG) Phishing Activity Trends Report (Q2 2025) tracks over a million phishing attacks in a single quarter and shows how attackers constantly rotate domains and URLs to stay ahead of blocklists.

Together, they show that:

Classical ML pipelines like this repo provide a solid baseline and are easy to deploy.

Modern deep learning models promise better generalization but are more complex.

The real world keeps changing, so defences need continuous updates and retraining.

This project sits at the “baseline” layer: a readable, modifiable starting point you can extend toward more modern architectures and bigger systems.

8. Challenges & Problem-Solving (What Went Wrong)
This project was not “clone and run.” A lot of learning came from fixing problems:

8.1. Environment & Dependencies
On Windows, pip was not recognized at first.

The solution was to use python -m pip consistently.

Some library versions (especially scikit-learn) did not match the original project and caused errors.

Fix:

Install packages one by one using python -m pip install <name>.

If an error mentions a missing method/attribute, try pinning to a specific version in requirements.txt and reinstalling.

8.2. Understanding Feature Engineering
The original malicious-URL project used a custom tokenizer for URLs. At first it looked like a wall of code.

Fix:

Add print() statements inside the tokenizer to see what tokens come out for a single test URL.

This made it clear which patterns the model was actually seeing.

8.3. Training vs Inference Mismatch
A very common pitfall:

Training uses one preprocessing pipeline.

Prediction uses a slightly different one.

Result: weird errors or nonsense predictions.

Fix:

Reuse the same tokenize_url function in both train and predict.

Save and load the exact same TfidfVectorizer and model objects with joblib.

These problems were frustrating but also built a lot of confidence with:

Reading stack traces

Adjusting dependencies

Using AI tools (like ChatGPT) to debug carefully instead of copy-pasting random code

9. Future Work & Ideas
This repo is a foundation, not the final system. Some directions you can explore:

Implement a 1D-CNN model

Take inspiration from the Haq et al. paper.

Represent URLs as sequences of characters or tokens.

Compare performance vs Logistic Regression.

Add more features

Hand-crafted features like URL length, number of digits, suspicious TLDs, etc.

Combine them with TF-IDF features.

Build a simple web UI

Use Flask / FastAPI + basic HTML/JS.

Paste a URL into a form and see the prediction live.

Integrate into an email analysis pipeline

Parse URLs out of emails.

Run them through this classifier.

Combine predictions with other signals, such as sender reputation.

Move toward a Zero-Trust Architecture

Treat every URL as untrusted until scanned.

Use this model as one signal among many.

10. How to Use This in Your Portfolio
You can use this repo to demonstrate:

Practical machine learning skills (data handling, feature engineering, model training, evaluation)

Real cybersecurity context (phishing, malicious URLs, defences)

Ability to debug, read other people’s code, and work through dependency issues

A clear description of what you built, why it matters, and how you’d improve it

Resources and Links
My Work:
-https://github.com/Sosa183/Using-machine-learning-to-detect-malicious-URLs 

Software and Tools:
- Baseline tool GitHub:       https://github.com/Sosa183/Using-machine-learning-to-detect-malicious-URLs  
- PyCharm: https://www.jetbrains.com/pycharm/ 
- Python: https://www.python.org/downloads/ 

Research Sources:
-   Safi, A., Jhanjhi, N. Z., Humayun, M., & others. (2023). A systematic literature review on phishing website detection techniques. Journal of King Saud University – Computer and Information Sciences.
 Comprehensive survey of machine-learning methods for phishing website detection, especially URL and HTML-based features. https://www.sciencedirect.com/science/article/pii/S1319157823000034?utm_source=chatgpt.com

- Haq, Q. E., Sadiq, S., Shafique, A., & Farooq, M. (2024). Detecting phishing URLs based on a deep learning approach using 1D-CNN. Applied Sciences, 14(22), 10086. https://www.mdpi.com/2076-3417/14/22/10086?utm_source=chatgpt.com 
- https://www.youtube.com/watch?v=9sC3t-g3iJA
- https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs
Learning Resources:
- Scikit-learn documentation: https://scikit-learn.org/ 
- Basic phishing awareness resources from organisations like CISA and NIST
- AI tools used for debugging and explanation
