import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import joblib

def scrape_karkidi_jobs(keyword="data science", pages=2):
    headers = {"User-Agent":"Mozilla/5.0"}
    base = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    L = []

    for page in range(1, pages+1):
        url = base.format(page=page, query=keyword.replace(" ", "%20"))
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.content, "html.parser")
        for job in soup.select("div.ads-details"):
            title = job.select_one("h4")
            title = title.get_text(strip=True) if title else None

            link_tag = job.find("a", href=lambda u: u and "Find-Jobs-Details" in u)
            job_link = "https://www.karkidi.com" + link_tag["href"] if link_tag else None

            company = (job.find("a", href=lambda u: u and "Employer-Profile" in u)
                        .get_text(strip=True) if job.find("a", href=lambda u: u and "Employer-Profile" in u) else None)

            location = job.find("p", class_=None).get_text(strip=True) if job.find("p", class_=None) else None
            experience = job.find("p", class_="emp-exp").get_text(strip=True) if job.find("p", class_="emp-exp") else None

            ks = job.find("span", string="Key Skills")
            skills = ks.find_next("p").get_text(strip=True) if ks else None

            sm = job.find("span", string="Summary")
            summary = sm.find_next("p").get_text(strip=True) if sm else None

            if title and skills:
                L.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Summary": summary,
                    "Skills": skills,
                    "JobLink": job_link
                })

    df = pd.DataFrame(L).dropna(subset=["Title", "Skills"])
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["Skills"])
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")

    # KMeans Clustering
    NUM_CLUSTERS = 5  # you can increase this for more granularity
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(tfidf_matrix)
    joblib.dump(kmeans, "kmeans_model.pkl")

    # Save final job list with clusters
    df.to_csv("karkidi_jobs.csv", index=False, encoding="utf-8")
    print(f"Scraped and clustered {len(df)} jobs into {NUM_CLUSTERS} clusters.")
    return df

if __name__ == "__main__":
    scrape_karkidi_jobs()
