import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# — Load artifacts —
vectorizer = joblib.load("tfidf_vectorizer.pkl")
kmeans = joblib.load("kmeans_model.pkl")
df = pd.read_csv("karkidi_jobs.csv", encoding="utf-8")

# — Preprocess for matching —
df["Skills_proc"] = df["Skills"].fillna("").str.lower().str.replace(",", " ")

# — Sidebar filters & sliders —
st.sidebar.header("🔍 Filters & Display")
location_filter = st.sidebar.selectbox("📍 Location", ["All"] + sorted(df["Location"].dropna().unique()))
company_filter = st.sidebar.selectbox("🏢 Company", ["All"] + sorted(df["Company"].dropna().unique()))
cluster_filter = st.sidebar.selectbox("🧠 Cluster", ["All"] + sorted(df["Cluster"].unique().astype(str)))
top_n = st.sidebar.slider("🔢 Number of Results", 1, 20, 5)

# — Main UI —
st.title("🎯 Skill-Based Job Matching")
user_input = st.text_input(
    "🛠️ Your Skills (comma-separated)",
    placeholder="e.g., python, machine learning, sql"
)

if user_input:
    # 1) Clean input: commas → spaces, lowercase
    ui = user_input.lower().replace(",", " ").strip()

    # 2) Vectorize
    uvec = vectorizer.transform([ui])
    jvecs = vectorizer.transform(df["Skills_proc"])

    # 3) Compute similarity & sort
    sims = cosine_similarity(uvec, jvecs).flatten()
    df["Similarity"] = sims
    res = df.sort_values("Similarity", ascending=False)

    # 4) Apply filters
    if location_filter != "All":
        res = res[res["Location"] == location_filter]
    if company_filter != "All":
        res = res[res["Company"] == company_filter]
    if cluster_filter != "All":
        res = res[res["Cluster"] == int(cluster_filter)]

    # 5) Display top_n
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
            st.write(f"**Cluster:** `{row['Cluster']}`")
            st.write(f"**Similarity:** `{row['Similarity']:.2f}`")
            if pd.notna(row["JobLink"]) and row["JobLink"].startswith("http"):
                st.markdown(f"[🔗 View Job Posting]({row['JobLink']})")
            else:
                st.write("_No direct link available_")
            st.write("---")
