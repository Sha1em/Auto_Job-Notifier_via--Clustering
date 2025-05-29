# 🎯 Auto Job Clustering & Matching App

Welcome to **Auto Job Clustering**, your AI-powered assistant that scrapes, clusters, and matches jobs based on your skills — with daily personalized email alerts!

🚀 Built with Python, Streamlit, and scikit-learn — this app helps job seekers discover the most relevant tech jobs from [karkidi.com](https://www.karkidi.com/).

---

## 🌟 Features

✅ **Daily Job Scraping** from Karkidi  
✅ **TF-IDF + KMeans** skill clustering for smart recommendations  
✅ **Skill-based Matching** using Cosine Similarity  
✅ **Streamlit Web UI** for live job matching  
✅ **📩 Daily Email Alerts** with top job matches  
✅ **GitHub Actions** automated pipeline  
✅ Clean, minimal UI — just enter your skills and go!

---

## 🖥️ Live App

Try it now 👉 [Job Matching App on Streamlit](https://autojob-notifiervia--clustering-ct7pjcdhbmgujvigb8pqtr.streamlit.app/)

---

## 📌 How It Works

1. **Scraper** runs daily via GitHub Actions  
2. Scraped jobs are vectorized + clustered with `TF-IDF` + `KMeans`  
3. User skills are matched using cosine similarity  
4. Top matches are shown instantly in the app or emailed to subscribers  
5. All fully automated, no manual effort needed!

---

## 📁 Project Structure

```bash
.
├── streamlit_app.py         # 🎨 Web interface (Streamlit)
├── scraper.py               # 🤖 Scrapes & clusters jobs daily
├── emailer.py               # 📬 Sends daily job match emails
├── subscribers.csv          # 📧 List of users & their skills
├── karkidi_jobs.csv         # 💼 Latest scraped job listings
├── tfidf_vectorizer.pkl     # 🔢 Saved TF-IDF model
├── kmeans_model.pkl         # 🔍 Saved KMeans clustering model
├── requirements.txt         # 📦 Python dependencies
└── .github/workflows/
    └── daily_scrape.yml     # ⚙️ GitHub Action for daily automation

--------------------------

⚙️ Setup Locally
# Clone the repo
git clone https://github.com/Sha1em/Auto_Job-Notifier_via--Clustering.git
cd Auto_Job-Notifier_via--Clustering

# Create and activate virtual env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py

