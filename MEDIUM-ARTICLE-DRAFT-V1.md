1. Put your screenshots in an images folder

In your project folder (Using-machine-learning-to-detect-malicious-URLs), create:

images/
    flask_console.png          # from screenshot of terminal with Flask + accuracy
    steamunlocked_good.png     # browser showing steamunlocked.org = good
    google_bad.png             # browser showing google.com = bad
    predict_route.png          # browser showing /predict = good
    pycharm_classifier.png     # PyCharm with my_url_classifier.py + data.csv
    pip_install.png            # terminal showing python -m pip install ...
    windows_folder.png         # Windows Explorer view of project files
    pycharm_project_tree.png   # PyCharm project tree + README


You can rename your actual PNGs to these names, or keep your names and just change the filenames in the README below.

2. Copy-paste this into README.md
# AI-Powered Phishing URL Detector

This repository is my hands-on project where I turned an open-source URL dataset
into a working **AI-powered phishing detector** using Python, scikit-learn, Flask,
and PyCharm on Windows.

I used the repo:

> https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs  

and then:

- Fixed Windows / Python / pip issues
- Trained the model locally
- Built a Flask app that lets me test URLs from the browser
- Documented every step with screenshots (see below)

---

## 🔍 What the Project Does

- Loads labeled URLs from `data/data.csv`  
- Vectorizes the URLs using **TF-IDF**  
- Trains a **Logistic Regression** classifier (`good` vs `bad`)  
- Exposes a **Flask web app** where any path gets classified  
- Shows a simple **entropy** score for the URL string  

Flask server running, training the model, and handling requests:

![Flask console](images/flask_console.png)

---

## 🧠 Tech Stack

- **Language:** Python 3.13  
- **Libraries:** `flask`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`  
- **Editor:** PyCharm on Windows 10  

PyCharm view of the project and classifier script:

![PyCharm with classifier + data](images/pycharm_classifier.png)

---

## 📂 Project Structure

```text
Using-machine-learning-to-detect-malicious-URLs/
├─ data/
│  ├─ data.csv
│  └─ ...
├─ AIserver.py
├─ my_url_classifier.py
├─ README.md
├─ REQUIREMENTS
└─ images/
   ├─ flask_console.png
   ├─ steamunlocked_good.png
   ├─ google_bad.png
   ├─ predict_route.png
   ├─ pycharm_classifier.png
   ├─ pip_install.png
   ├─ windows_folder.png
   └─ pycharm_project_tree.png


Windows Explorer view of the main files:

 Setup & Installation (What I Actually Did)
 Clone and open in PyCharm
git clone https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs.git
cd Using-machine-learning-to-detect-malicious-URLs


Open PyCharm → File → Open… and select this folder.

Overall PyCharm project tree:

 Fixing pip / Python and installing requirements

At first, PowerShell gave:

pip : The term 'pip' is not recognized as the name of a cmdlet...

So instead of pip, I used:

python -m pip install --upgrade pip
python -m pip install -r REQUIREMENTS
# plus anything missing:
python -m pip install flask pandas numpy scikit-learn matplotlib


Install output looked like this:

If PowerShell said “Python was not found…”, I used the Python that PyCharm configured in
C:\Users\alber\AppData\Local\Programs\Python\Python313\python.exe.

 Running the Model
Option A: Flask web demo (AIserver.py)

In PyCharm’s terminal:

python AIserver.py


What happens:

The script trains the Logistic Regression model on data/data.csv.

It prints test accuracy (around 0.98 in my run).

It starts a Flask server on http://127.0.0.1:5000.

You should see something like:

Testing URLs in the Browser
1. steamunlocked.org – predicted good

Visit:

http://127.0.0.1:5000/https://steamunlocked.org/


Result page:

You asked for: https://steamunlocked.org/

AI output (label): good
Entropy: 4.103909910282364


2. google.com (typed with double https) – predicted bad

I accidentally entered a weird URL:

http://127.0.0.1:5000/https://https://www.google.com/


The model said:

You asked for: https://https://www.google.com/

AI output (label): bad
Entropy: 3.6277635530073997


This is a nice example of how URL-only models can misclassify odd but
benign strings. Great thing to talk about under “limitations”.

3. Simple path /predict – predicted good

Visiting:

http://127.0.0.1:5000/predict


gave:

You asked for: predict

AI output (label): good
Entropy: 2.8073549220576046


 CLI Classifier (my_url_classifier.py)

I also created my_url_classifier.py which:

Loads data.csv

Trains TF-IDF + Logistic Regression

Prints accuracy and a classification report

Lets you type URLs and get a prediction

Run it with:

python my_url_classifier.py


You can then test:

Enter a URL to classify (or 'quit'): https://google.com
Enter a URL to classify (or 'quit'): http://login-paypal.com.verify-account.ru/login

🧷 Issues & Fixes (for future me / other students)

pip not recognized
→ Use python -m pip ... instead of plain pip.

Python launches Microsoft Store
→ Install Python from python.org and use PyCharm’s configured interpreter.

ModuleNotFoundError for flask, sklearn, etc.
→ Install them manually with python -m pip install flask scikit-learn pandas numpy.

Weird predictions (e.g., Google = bad)
→ Remember the model only sees URL strings and the dataset’s labels.
It’s a great talking point for limitations and future improvements.

🔮 Future Ideas

Try a 1D-CNN / deep learning model for URLs.

Use a fresher phishing dataset (e.g., from APWG feeds).

Expose a JSON /api/predict endpoint.

Combine URL signals with email metadata or page content as part of a
bigger AI-Powered Phishing Detection System.

This repo is mainly a learning project and a portfolio piece for my cybersecurity coursework, not a production security product.
All screenshots in this README are from my actual setup on Windows using PyCharm.
