import os, json, pandas as pd, re, requests, numpy as np
from bs4 import BeautifulSoup
from flask import render_template, request, redirect, url_for, flash, Response
from werkzeug.utils import secure_filename
from . import db
from .models import Candidate, Job
from .utils import (
    load_skill_aliases, extract_skills_from_text, embed_texts,
    compute_match_score, extract_text_from_pdf, extract_name_email
)
from .crawler import crawl_jobs


# ---------------------------------------------------------------
# Helper function to clean text
# ---------------------------------------------------------------
def clean_text(s):
    if s is None:
        return ""
    if isinstance(s, bytes):
        s = s.decode("utf-8", errors="ignore")
    return s.replace("\x00", "").strip()


# ---------------------------------------------------------------
# Initialize routes
# ---------------------------------------------------------------
def init_app(app):
    SKILL_MAP = load_skill_aliases(os.path.join(app.config["DATA_DIR"], "skills_with_aliases_100.csv"))
    app.config["UPLOAD_DIR"] = os.path.join(app.config["DATA_DIR"], "uploads")
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)

    # ---------------- Home ----------------
    @app.route("/")
    def index():
        return render_template("index.html")

    # ---------------- Candidate Login ----------------
    @app.route("/candidate_login")
    def candidate_login():
        return render_template("candidate_login.html")

    # ---------------- Candidate Job Search ----------------
    @app.route("/search_jobs", methods=["POST"])
    def search_jobs():
        location = request.form.get("location", "").strip()
        qualification = request.form.get("qualification", "").strip()
        skills = request.form.get("skills", "").strip()

        if not skills:
            flash("Please enter skills to search", "warning")
            return redirect(url_for("candidate_login"))

        jobs = crawl_jobs(skills, location, qualification)
        if not jobs:
            flash("No live jobs found, showing fallback data.", "info")
            jobs_path = os.path.join(app.config["DATA_DIR"], "job_title_des.csv")
            try:
                df = pd.read_csv(jobs_path)
                jobs = [
                    {
                        "title": r.get("job_title") or r.get("title") or "",
                        "company": r.get("company") or "",
                        "summary": r.get("description") or "",
                        "url": r.get("url") or "#"
                    }
                    for _, r in df.head(10).iterrows()
                ]
            except Exception:
                jobs = []

        return render_template("jobs.html", jobs=jobs, skills=skills, location=location)

    # ---------------- Candidate Resume Upload ----------------
    @app.route("/upload_cv", methods=["POST"])
    def upload_cv():
        file = request.files.get("resume")
        if not file or not file.filename:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return {"status": "error", "message": "Please upload a valid resume file."}, 400
            flash("Please upload a valid resume file.", "danger")
            return redirect(request.referrer)

        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_DIR"], filename)
        file.save(save_path)

        # Extract text and candidate info
        resume_text = clean_text(extract_text_from_pdf(save_path))
        extracted_name, extracted_email = extract_name_email(resume_text)
        skills = extract_skills_from_text(resume_text, SKILL_MAP)

        # ✅ Generate embedding for resume
        cand_emb = embed_texts([resume_text])[0] if resume_text else None

        candidate = Candidate(
            name=clean_text(extracted_name),
            email=clean_text(extracted_email),
            resume_text=resume_text,
            skills_json=json.dumps(skills),
            embedding=json.dumps(cand_emb.tolist()) if cand_emb is not None else None
        )
        db.session.add(candidate)
        db.session.commit()

        # ✅ Handle AJAX vs normal form
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return {"status": "success", "message": f"Resume uploaded successfully for {extracted_name}."}, 200
        else:
            flash(f"Resume uploaded successfully for {extracted_name}.", "success")
            return redirect(request.referrer)

    # ---------------- HR Login ----------------
    @app.route("/hr_login", methods=["GET", "POST"])
    def hr_login():
        if request.method == "POST":
            username = request.form.get("username")
            password = request.form.get("password")
            if username == "admin" and password == "admin123":
                return redirect(url_for("upload_job_description"))
            flash("Invalid credentials", "danger")
            return redirect(url_for("hr_login"))
        return render_template("login.html")

    # ---------------- HR Upload Job Description ----------------
    @app.route("/upload_job", methods=["GET", "POST"])
    def upload_job_description():
        if request.method == "POST":
            job_title = request.form.get("job_title", "").strip()
            company = request.form.get("company", "").strip()
            job_desc = request.form.get("job_desc", "").strip()

            if not (job_title and job_desc):
                flash("Please provide both job title and job description.", "danger")
                return redirect(url_for("upload_job_description"))

            # ✅ Generate embedding for job description
            job_emb = embed_texts([job_desc])[0] if job_desc else None

            job = Job(
                title=clean_text(job_title),
                company=clean_text(company),
                description=clean_text(job_desc),
                skills_required=json.dumps(extract_skills_from_text(job_desc, SKILL_MAP)),
                embedding=json.dumps(job_emb.tolist()) if job_emb is not None else None
            )
            db.session.add(job)
            db.session.commit()

            # Automatically match with all stored resumes
            candidates = Candidate.query.all()
            if not candidates:
                flash("No candidate resumes found in database.", "warning")
                return redirect(url_for("upload_job_description"))

            results = []
            for cand in candidates:
                if not cand.resume_text:
                    continue

                # Load embeddings back from JSON
                cand_emb = np.array(json.loads(cand.embedding)) if cand.embedding else embed_texts([cand.resume_text])[0]
                j_emb = np.array(json.loads(job.embedding)) if job.embedding else embed_texts([job.description])[0]

                cand_skills = json.loads(cand.skills_json or "[]")
                job_skills = json.loads(job.skills_required or "[]")

                overlap = len(set(job_skills).intersection(set(cand_skills)))
                skill_ratio = overlap / max(1, len(job_skills)) if job_skills else 0.0

                # ✅ Compute full ATS-style score
                score, breakdown = compute_match_score(
                    cand_emb, j_emb, skill_ratio,
                    cand_text=cand.resume_text,
                    job_text=job.description
                )

                score_percent = round(score * 100, 2)
                results.append({
                    "candidate": cand,
                    "email": cand.email,
                    "score": score_percent,
                    "skills_match": breakdown["skills"],
                    "qualification_match": breakdown["qualification"],
                    "experience_match": breakdown["experience"],
                    "embedding_match": breakdown["embedding"],
                    "status": "Shortlisted" if score_percent >= 50 else "Rejected"
                })

            return render_template("shortlist.html", results=sorted(results, key=lambda x: x["score"], reverse=True))

        return render_template("upload.html")

    # ---------------- Download Results ----------------
    @app.route("/download_results")
    def download_results():
        from io import StringIO
        import csv

        results = []
        jobs = Job.query.all()
        candidates = Candidate.query.all()

        for job in jobs:
            job_skills = json.loads(job.skills_required or "[]")
            job_emb = np.array(json.loads(job.embedding)) if job.embedding else None

            for cand in candidates:
                cand_skills = json.loads(cand.skills_json or "[]")
                cand_emb = np.array(json.loads(cand.embedding)) if cand.embedding else None

                overlap = len(set(job_skills).intersection(set(cand_skills)))
                skill_ratio = overlap / max(1, len(job_skills)) if job_skills else 0.0

                score, breakdown = compute_match_score(
                    cand_emb, job_emb, skill_ratio,
                    cand_text=cand.resume_text, job_text=job.description
                )

                results.append({
                    "name": cand.name,
                    "email": cand.email,
                    "score": round(score * 100, 2),
                    "skills": breakdown["skills"],
                    "qualification": breakdown["qualification"],
                    "experience": breakdown["experience"],
                    "embedding": breakdown["embedding"],
                    "status": "Shortlisted" if score >= 0.5 else "Rejected"
                })

        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(["Candidate Name", "Email", "Total Score (%)", "Skills (%)", "Qualification (%)", "Experience (%)", "Embedding (%)", "Status"])

        for r in results:
            writer.writerow([r["name"], r["email"], r["score"], r["skills"], r["qualification"], r["experience"], r["embedding"], r["status"]])

        output.seek(0)
        return Response(output, mimetype="text/csv",
                        headers={"Content-Disposition": "attachment;filename=HR_dashboard_results.csv"})
