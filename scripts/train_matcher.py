"""
Train a simple model that predicts match score between job and candidate.
Features:
 - cosine similarity between embeddings
 - skill overlap ratio
Labeling: if your dataset has a 'label' column (matched=1), train supervised; otherwise create proxy label using cosine>threshold.
"""
import os, joblib, numpy as np, pandas as pd
from config import Config
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

data_dir = Config.DATA_DIR

def load_embeddings(file):
    d = joblib.load(file)
    return d['rows'], d['embeddings']

if __name__ == "__main__":
    jobs_df, job_embs = load_embeddings(os.path.join(data_dir,"job_embeddings.pkl"))
    cands_df, cand_embs = load_embeddings(os.path.join(data_dir,"candidate_embeddings.pkl"))

    X = []
    y = []
    # create sample dataset by pairing every job with some candidates (for demo, pair first 200 x first 200)
    max_j = min(200, len(job_embs))
    max_c = min(500, len(cand_embs))
    for ji in range(max_j):
        for ci in range(max_c):
            emb_sim = float(cosine_similarity([job_embs[ji]],[cand_embs[ci]])[0,0])
            # skill overlap proxy:
            # attempt to extract skills from text using simple token matching
            job_text = str((jobs_df.iloc[ji].get('title') or "") + " " + (jobs_df.iloc[ji].get('description') or ""))
            cand_text = str((cands_df.iloc[ci].get('resume_text') or "") if 'resume_text' in cands_df.columns else cands_df.iloc[ci].astype(str).agg(" ".join))
            # naive skill score: fraction of shared words in top 200 tokens
            job_tokens = set(job_text.lower().split()[:200])
            cand_tokens = set(cand_text.lower().split()[:200])
            overlap = len(job_tokens.intersection(cand_tokens))
            skill_ratio = overlap / max(1, len(job_tokens))
            X.append([emb_sim, skill_ratio])
            # proxy label: higher if emb_sim high and skill_ratio high
            label = 0.7*emb_sim + 0.3*skill_ratio
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    print("RMSE:", mean_squared_error(y_test, preds, squared=False))
    print("R2:", r2_score(y_test, preds))
    joblib.dump(rf, os.path.join(data_dir, "matcher_rf.pkl"))
    print("Saved matcher model.")



