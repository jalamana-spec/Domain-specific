import os, re, json
from config import Config

# -------------------- PDF Text Extraction --------------------
def extract_text_from_pdf(path):
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text + "\n"
        return re.sub(r'\s+', ' ', text or '').strip()
    except Exception:
        try:
            with open(path, "r", encoding="utf8", errors="ignore") as f:
                return re.sub(r'\s+', ' ', f.read()).strip()
        except Exception:
            return ""

# -------------------- Load Skill Aliases --------------------
def load_skill_aliases(csv_path):
    mapping = {}
    try:
        import pandas as pd
        if not os.path.exists(csv_path):
            return mapping
        df = pd.read_csv(csv_path, engine="python")
        for _, r in df.iterrows():
            skill = None
            for c in ["skill","Skill","SkillName","skill_name","Skill_Name","Skillname"]:
                if c in df.columns:
                    skill = r.get(c)
                    break
            if not skill:
                skill = r.iloc[0]
            skill = str(skill).strip()
            aliases = ""
            for c in ["aliases","Aliases","alias","Alias"]:
                if c in df.columns:
                    aliases = r.get(c)
                    break
            alias_list = []
            if aliases and not pd.isna(aliases):
                alias_list = [a.strip().lower() for a in str(aliases).replace("|",";").split(";") if a.strip()]
            alias_list.append(skill.lower())
            mapping[skill.lower()] = list(dict.fromkeys(alias_list))
    except Exception:
        mapping = {}
    return mapping

# -------------------- Skill Extraction --------------------
_skill_map_cache = None
def get_skill_map():
    global _skill_map_cache
    if _skill_map_cache is None:
        csv_path = os.path.join(Config.DATA_DIR, "skills_with_aliases_100.csv")
        _skill_map_cache = load_skill_aliases(csv_path)
    return _skill_map_cache

def extract_skills_from_text(text, skill_map=None):
    if not skill_map:
        skill_map = get_skill_map()
    found = set()
    text_l = (text or "").lower()
    for canonical, aliases in skill_map.items():
        for a in aliases:
            if re.search(r'\b' + re.escape(a) + r'\b', text_l):
                found.add(canonical)
                break
    return sorted(found)

# -------------------- Attribute Extraction --------------------
def extract_experience(text):
    text = text.lower()
    
    # 1️⃣ Try explicit "years" pattern first
    match = re.search(r'(\d+)\+?\s*(years?|yrs?)', text)
    if match:
        return int(match.group(1))
    
    # 2️⃣ Try date range detection
    date_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{4}'
    dates = re.findall(date_pattern, text)
    
    # Extract all years from text
    years = re.findall(r'(20\d{2}|19\d{2})', text)
    years = [int(y) for y in years]
    
    if len(years) >= 2:
        start_year = min(years)
        end_year = max(years)
        duration = end_year - start_year
        if duration == 0:
            duration = 1  # assume at least 1 year if same year
        return duration

    # 3️⃣ Default if nothing found
    return 0

import re

import re
import re

def extract_qualification(text):
    """
    Extract and normalize educational qualification from text.
    Handles cases like 'B.Tech / M.Tech', 'Masters in CS', etc.
    Returns: 'bachelor', 'master', 'phd', or 'unknown'
    """
    if not text:
        return "unknown"

    text_l = text.lower().replace(".", "").replace(" ", "")

    # 🔹 Check for highest qualification first
    if re.search(r"(phd|doctorate|doctorofphilosophy|dr)", text_l):
        return "phd"
    elif re.search(r"(mtech|me|msc|mca|ms|master|postgraduate|pg)", text_l):
        return "master"
    elif re.search(r"(btech|be|bsc|bca|bachelor|undergraduate|graduate|degree)", text_l):
        return "bachelor"
    else:
        return "unknown"



# -------------------- SBERT Embedding --------------------
_EMB_MODEL = None
def _ensure_embedding_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("🔹 Loading SBERT model:", Config.EMBEDDING_MODEL)
            _EMB_MODEL = SentenceTransformer(Config.EMBEDDING_MODEL)
            print("✅ SBERT model loaded successfully.")
        except Exception as e:
            print("❌ ERROR loading SBERT model:", e)
            raise RuntimeError(
                "Embedding model not available. Install sentence-transformers and dependencies."
            )

def embed_texts(texts):
    """Return list of embeddings for given texts."""
    _ensure_embedding_model()
    return _EMB_MODEL.encode(texts, convert_to_tensor=False)

# -------------------- Cosine Similarity --------------------
def cosine_sim(a, b):
    try:
        import numpy as np
        a = np.array(a); b = np.array(b)
        na = (a*a).sum()**0.5; nb = (b*b).sum()**0.5
        if na == 0 or nb == 0:
            return 0.0
        return float((a*b).sum() / (na*nb))
    except Exception:
        return 0.0

# -------------------- Weighted Match Score --------------------
def compute_match_score(cand_emb, job_emb, skill_ratio, cand_text=None, job_text=None, job_title=""):
    
        emb_sim = 0.0
        if cand_emb is not None and job_emb is not None:
            emb_sim = cosine_sim(cand_emb, job_emb)

        # --- Skills ---
        skill_score = skill_ratio

        # --- Qualification match ---
        q_cand = extract_qualification(cand_text or "")
        q_job = extract_qualification(job_text or "")
        if q_cand == q_job and q_job != "unknown":
            qual_score = 1.0
        elif (q_cand, q_job) in [("master", "bachelor"), ("phd", "master"), ("bachelor", "master")]:
            qual_score = 0.8
        else:
            qual_score = 0.0

        # --- Experience match ---
        exp_cand = extract_experience(cand_text or "")
        exp_job = extract_experience(job_text or "")
        if exp_job > 0:
            exp_score = min(exp_cand, exp_job) / exp_job
        else:
            exp_score = 0.5  # partial if no job exp mentioned

        # --- Weights (rebalanced) ---
        weights = {
            "embedding": 0.60,
            "skills": 0.25,
            "qualification": 0.10,
            "experience": 0.05
        }

        final_score = (
            weights["embedding"] * emb_sim +
            weights["skills"] * skill_score +
            weights["qualification"] * qual_score +
            weights["experience"] * exp_score
        )

        # --- Breakdown for dashboard ---
        contributions = {
            "embedding": round(weights["embedding"] * emb_sim * 100, 2),
            "skills": round(weights["skills"] * skill_score * 100, 2),
            "qualification": round(weights["qualification"] * qual_score * 100, 2),
            "experience": round(weights["experience"] * exp_score * 100, 2),
        }

        return round(final_score, 4), contributions



# -------------------- Name & Email Extraction --------------------
def extract_name_email(text):
    
    email = None
    name = None

    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if email_match:
        email = email_match.group(0)

    lines = [l.strip() for l in text.split('\n') if l.strip()]
    possible_names = []
    for i, line in enumerate(lines):
        if email and email in line:
            if i > 0:
                possible_names.append(lines[i-1])
        elif len(line.split()) in [2, 3] and line.istitle():
            possible_names.append(line)

    if possible_names:
        name = possible_names[0].title()

    if not name and lines:
        tokens = lines[0].split()
        if len(tokens) >= 2:
            name = f"{tokens[0].title()} {tokens[1].title()}"

    return name or "Unknown Candidate", email or "Not found"

