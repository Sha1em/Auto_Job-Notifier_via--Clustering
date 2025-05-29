import streamlit as st
import pandas as pd
import joblib
import os
import csv
from sklearn.metrics.pairwise import cosine_similarity

# — Load artifacts —
vectorizer = joblib.load(os.path.join(os.getcwd(), "tfidf_vectorizer.pkl"))
kmeans = joblib.load(os.path.join(os.getcwd(), "kmeans_model.pkl"))
df = pd.read_csv(os.path.join(os.getcwd(), "karkidi_jobs.csv"), encoding="utf-8")

# — Preprocess for matching —
df["Skills_proc"] = df["Skills"].fillna("").str.lower().str.replace(",", " ")

# — Main Title —
st.title("🎯 Skill-Based Job Matching")

# --- 📩 Subscription Section ---
st.subheader("📩 Subscribe for Daily Job Alerts")
user_email = st.text_input("📧 Your Email")
user_skills = st.text_input("🔧 Your Skills (comma-separated)", placeholder="e.g., python, sql")

if st.button("✅ Subscribe"):
    if user_email and user_skills:
        subs_path = os.path.join(os.getcwd(), "subscribers.csv")
        file_exists = os.path.isfile(subs_path)
        with open(subs_path, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["email", "skills"])
            writer.writerow([user_email, user_skills])
        st.success("You're subscribed for daily job alerts!")
    else:
        st.warning("Please enter both email and skills.")

# --- 🔍 Filter Controls (replaces sidebar) ---
st.subheader("🎛️ Filters")
col1, col2 = st.columns(2)
with col1:
    location_filter = st.selectbox("📍 Filter by Location", ["All"] + sorted(df["Location"].dropna().unique()))
with col2:
    company_filter = st.selectbox("🏢 Filter by Company", ["All"] + sorted(df["Company"].dropna().unique()))

col3, col4 = st.columns(2)
with col3:
    cluster_filter = st.selectbox("🧠 Filter by Cluster", ["All"] + sorted(df["Cluster"].unique().astype(str)))
with col4:
    top_n = st.slider("🔢 Number of Results", 1, 20, 5)

# --- 🧠 Job Recommender Section ---
st.subheader("🔎 Find Matching Jobs Now")
user_input = st.text_input("🛠️ Your Skills (comma-separated)", placeholder="e.g., python, machine learning, sql")

if user_input:
    ui = user_input.lower().replace(",", " ").strip()
    uvec = vectorizer.transform([ui])
    jvecs = vectorizer.transform(df["Skills_proc"])

    sims = cosine_similarity(uvec, jvecs).flatten()
    df["Similarity"] = sims
    res = df.sort_values("Similarity", ascending=False)

    if location_filter != "All":
        res = res[res["Location"] == location_filter]
    if company_filter != "All":
        res = res[res["Company"] == company_filter]
    if cluster_filter != "All":
        res = res[res["Cluster"] == int(cluster_filter)]

    st.subheader(f"📋 Top {top_n} Matching Jobs")
    if res.empty:
        st.warning("No matches found—try broadening your skills!")
    else:
        for _, row in res.head(top_n).iterrows():
            st.markdown(f"### 🧾 {row['Title']}")
            st.write(f"**Company:** {row['Company']}")
            st.write(f"**Location:** {row['Location']}")
            st.write(f"**Experience:** {row['Experience'] or 'N/A'}")
            st.write(f"**Skills:** {row['Skills']}")
            st.write(f"**Summary:** {row['Summary'] or 'N/A'}")
            st.write(f"**Cluster:** {row['Cluster']}")
            st.write(f"**Similarity:** {row['Similarity']:.2f}")
            if pd.notna(row["JobLink"]) and row["JobLink"].startswith("http"):
                st.markdown(f"[🔗 View Job Posting]({row['JobLink']})")
            else:
                st.write("_No direct link available_")
            st.write("---")
