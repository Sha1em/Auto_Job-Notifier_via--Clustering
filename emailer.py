import os
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Config ----------
YOUR_EMAIL = os.environ.get("EMAIL")
YOUR_APP_PASSWORD = os.environ.get("APP_PASSWORD")
TOP_N = 5  # number of job matches per user

# ---------- Load Models & Data ----------
df = pd.read_csv(os.path.join(os.getcwd(), "karkidi_jobs.csv"))
vectorizer = joblib.load(os.path.join(os.getcwd(), "tfidf_vectorizer.pkl"))

# Preprocess job skills
df["Skills_proc"] = df["Skills"].fillna("").str.lower().str.replace(",", " ")

# Load subscribers
subs = pd.read_csv(os.path.join(os.getcwd(), "subscribers.csv"))

# ---------- Email Function ----------
def send_email(to_email, subject, body):
    msg = MIMEMultipart()
    msg["From"] = YOUR_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(YOUR_EMAIL, YOUR_APP_PASSWORD)
            server.send_message(msg)
        print(f"✅ Sent email to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")

# ---------- Match & Send ----------
for _, row in subs.iterrows():
    email = row["email"]
    skills = row["skills"].lower().replace(",", " ")
    uvec = vectorizer.transform([skills])
    jvecs = vectorizer.transform(df["Skills_proc"])

    sims = cosine_similarity(uvec, jvecs).flatten()
    df["Similarity"] = sims
    matches = df.sort_values("Similarity", ascending=False).head(TOP_N)

    if matches.empty:
        body = "No new matching jobs found today. Please check back later!"
    else:
        body = f"Hi,\n\nHere are your top {TOP_N} job matches for today:\n\n"
        for _, job in matches.iterrows():
            body += f"📌 {job['Title']} at {job['Company']} ({job['Location']})\n"
            body += f"🔗 {job['JobLink']}\n"
            body += f"🛠️ Skills: {job['Skills']}\n"
            body += f"🧠 Cluster: {job['Cluster']}\n"
            body += f"📈 Similarity: {job['Similarity']:.2f}\n\n"
        body += "🖥️ Visit the app: https://autojobclusteringapp-wvuyvdxafrkvq9dvw4appaq.streamlit.app/\n\n"
        body += "Best,\nJob Matcher Bot"

    send_email(email, "🎯 Your Daily Job Matches", body)
