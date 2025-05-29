import streamlit as st
import pandas as pd
import joblib
import os
import csv
from sklearn.metrics.pairwise import cosine_similarity

# — Page configuration —
st.set_page_config(
    page_title="Auto Job Matcher",
    page_icon="🎯",
    layout="wide"
)

# — Load artifacts —
cur_dir = os.getcwd()
vectorizer = joblib.load(os.path.join(cur_dir, "tfidf_vectorizer.pkl"))
kmeans = joblib.load(os.path.join(cur_dir, "kmeans_model.pkl"))
df = pd.read_csv(os.path.join(cur_dir, "karkidi_jobs.csv"), encoding="utf-8")

# — Preprocess for matching —
df["Skills_proc"] = df["Skills"].fillna("").str.lower().str.replace(",", " ")

# — Sidebar controls —
st.sidebar.title("🔍 Filters & Subscription")
# Subscription in sidebar
with st.sidebar.expander("📩 Subscribe for Daily Alerts", expanded=False):
    email = st.text_input("Your Email", key="sub_email")
    skills_sub = st.text_input("Your Skills (comma-separated)", key="sub_skills")
    if st.button("Subscribe", key="subscribe_btn"):
        if email and skills_sub:
            subs_file = os.path.join(cur_dir, "subscribers.csv")
            exists = os.path.isfile(subs_file)
            with open(subs_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not exists:
                    writer.writerow(["email", "skills"])
                writer.writerow([email, skills_sub])
            st.sidebar.success("Subscribed for daily alerts!")
        else:
            st.sidebar.error("Please provide both email and skills.")

# main filters
location_filter = st.sidebar.selectbox(
    "Location", ["All"] + sorted(df["Location"].dropna().unique())
)
company_filter = st.sidebar.selectbox(
    "Company", ["All"] + sorted(df["Company"].dropna().unique())
)
cluster_filter = st.sidebar.selectbox(
    "Cluster", ["All"] + sorted(df["Cluster"].unique().astype(str))
)

# — Main layout —
st.title("🎯 Auto Job Clustering & Matching")
st.markdown(
    "Enter your skills below to find top matching jobs instantly, or subscribe for daily email alerts!"
)

# two-column input
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input(
        "🛠️ Your Skills (comma-separated)",
        placeholder="e.g., python, machine learning, sql"
    )
with col2:
    top_n = st.number_input(
        "Results to Show", min_value=1, max_value=20, value=5, step=1
    )

if user_input:
    ui = user_input.lower().replace(",", " ").strip()
    uvec = vectorizer.transform([ui])
    jvecs = vectorizer.transform(df["Skills_proc"])
    sims = cosine_similarity(uvec, jvecs).flatten()
    df["Similarity"] = sims
    res = df.sort_values("Similarity", ascending=False)

    # apply filters
    if location_filter != "All":
        res = res[res["Location"] == location_filter]
    if company_filter != "All":
        res = res[res["Company"] == company_filter]
    if cluster_filter != "All":
        res = res[res["Cluster"] == int(cluster_filter)]

    st.subheader(f"📋 Top {top_n} Matching Jobs")
    if res.empty:
        st.warning("No matches found—try different skills or filters.")
    else:
        for _, row in res.head(top_n).iterrows():
            st.write("---")
            job_col1, job_col2 = st.columns([3, 1])
            with job_col1:
                st.markdown(f"### {row['Title']}")
                st.write(f"**Company:** {row['Company']}")
                st.write(f"**Location:** {row['Location']}")
                st.write(f"**Experience:** {row['Experience'] or 'N/A'}")
                st.write(f"**Skills:** {row['Skills']}")
                st.write(f"**Summary:** {row['Summary'] or 'N/A'}")
            with job_col2:
                st.metric(label="Similarity", value=f"{row['Similarity']:.2f}")
                st.metric(label="Cluster", value=row['Cluster'])
                if pd.notna(row['JobLink']) and row['JobLink'].startswith('http'):
                    st.markdown(f"[🔗 View Job]({row['JobLink']})")
            
