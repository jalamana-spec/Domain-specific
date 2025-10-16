from . import db
import json

class Candidate(db.Model):
    __tablename__ = "candidates"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    email = db.Column(db.String(255))
    qualification = db.Column(db.String(255))
    skills_json = db.Column(db.Text)       # JSON string of skills
    resume_path = db.Column(db.String(1024))
    resume_text = db.Column(db.Text)
    embedding = db.Column(db.PickleType)  # optional embedding stored as pickled array

    def skill_list(self):
        try:
            return json.loads(self.skills_json or "[]")
        except Exception:
            return [s.strip() for s in (self.skills_json or "").split(",") if s.strip()]

class Job(db.Model):
    __tablename__ = "jobs"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255))
    company = db.Column(db.String(255))
    description = db.Column(db.Text)
    skills_required = db.Column(db.Text)   # JSON string of skills
    embedding = db.Column(db.PickleType)

    def skill_list(self):
        try:
            return json.loads(self.skills_required or "[]")
        except Exception:
            return [s.strip() for s in (self.skills_required or "").split(",") if s.strip()]

class Application(db.Model):
    __tablename__ = "applications"
    id = db.Column(db.Integer, primary_key=True)
    candidate_id = db.Column(db.Integer, db.ForeignKey("candidates.id"))
    job_id = db.Column(db.Integer, db.ForeignKey("jobs.id"))
    match_score = db.Column(db.Float)

    candidate = db.relationship("Candidate", backref=db.backref("applications", lazy="dynamic"))
    job = db.relationship("Job", backref=db.backref("applications", lazy="dynamic"))
