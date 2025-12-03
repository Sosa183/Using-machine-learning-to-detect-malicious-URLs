# Can My Laptop Catch Phishing Links? Building a Machine Learning URL Detector in PyCharm

> **Draft:** MEDIUM-ARTICLE-DRAFT-V1.md  
> **Project:** AI-Powered Phishing URL Detection (URL-based)  
> **Student:** Albert Sosa  

---

## 1. Introduction / Abstract (Hook)

The first time I saw a phishing link that looked almost identical to the real site, I wondered a scary question:

> *If I can’t tell the difference, how is a regular user supposed to stay safe?*

Modern phishing attacks hide behind look-alike domains, random subdomains, and URL tricks that are easy to miss when you’re half-awake checking email. I wanted to know whether a simple machine-learning model running on **my own laptop** could help spot these suspicious links automatically.

In this article, I walk through how I turned an open-source project into my own **AI-powered phishing URL detector** using Python, scikit-learn, and PyCharm on Windows. I’ll show how I:

- Set up the environment and fixed very real Windows issues (like `pip` not being recognized).
- Trained a Logistic Regression model on a dataset of labeled URLs.
- Built a tiny **Flask web app** that classifies URLs directly from the browser.
- Tested real-world-style URLs like `steamunlocked.org` and `google.com` and looked at how the model behaves.
- Connected my results to current phishing research and industry statistics.

By the end, you’ll see exactly how to reproduce this project yourself, what went wrong for me, and why URL-only models are powerful but also limited in defending against modern phishing.

[Screenshot: Hero image or banner of terminal + browser showing URL classification]

---

## 2. Purpose and Background

I chose this project for two reasons: curiosity and practicality.

First, phishing is **everywhere**. Most people think of phishing emails, but underneath almost every phishing campaign is at least one malicious URL. If a model can learn what “bad” URLs look like, it could be plugged into email gateways, proxies, or even browser extensions as one more layer of defense.

Second, I’m a cybersecurity student, and I wanted a project that:

- Uses **real code**, not just a box-and-arrow diagram.
- Gives me something to show on **GitHub** and later in job interviews.
- Fits nicely into my course’s Medium article assignment.

### What this tool is

At its core, my project is a **URL-based classifier**:

- Input: a string that looks like a URL (e.g., `https://login-paypal.com.verify-account.ru/login`)
- Output: a label such as `good` or `bad`, based on patterns learned from a dataset.

Under the hood, the model:

- Tokenizes URLs into chunks (characters or small n-grams).
- Converts them into numeric vectors using **TF-IDF**.
- Learns a decision boundary using **Logistic Regression**.

I started from Faizan Ahmad’s open-source repo, **“Using machine learning to detect malicious URLs”**, which already had a dataset and a basic training script. My contribution was to:

- Make it work reliably on **Windows / PyCharm**.
- Clean up and modernize the training and Flask code.
- Wrap it in better documentation and a Medium-style narrative.

### Who uses tools like this?

In the real world, similar techniques are used by:

- **Email security platforms** to pre-score URLs in messages.
- **Secure web gateways** to block suspicious destinations.
- **Browser extensions** and endpoint agents that warn users about risky links.

My project is obviously not production-grade, but it’s a miniature version of the ideas those systems use.

Emotionally, I was a mix of **excited** and **intimidated**. I expected “just pip install and run,” but my actual experience was wrestling with Python versions, missing libraries, and debugging Windows-specific problems. Ironically, fixing those problems taught me as much as the machine learning did.

---

## 3. Installation and Setup Tutorial (PyCharm on Windows)

In this section I’ll show exactly how I got the project running, step by step, including the issues I hit and how I fixed them. You can follow this as a mini-tutorial.

> All commands below are from **PowerShell** or the **PyCharm terminal** on Windows.

### 3.1 Clone the project

I started by cloning the original repo into my Downloads folder:

```bash
git clone https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs.git
cd Using-machine-learning-to-detect-malicious-URLs

