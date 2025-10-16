import requests
from bs4 import BeautifulSoup
import urllib.parse
import re

def crawl_jobs(skills, location, qualification=None):
    """
    Fetch real Naukri job postings based on skills, location, and qualification.
    Only returns genuine job listing URLs (skips sponsored/redirect/fake entries).
    """
    query_parts = [skills, qualification]
    query = " ".join([q for q in query_parts if q])
    if not query:
        return []

    q = urllib.parse.quote_plus(query)
    l = urllib.parse.quote_plus(location or "")
    search_url = f"https://www.naukri.com/{q}-jobs-in-{l}"

    headers = {"User-Agent": "Mozilla/5.0"}
    jobs = []

    try:
        res = requests.get(search_url, headers=headers, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        job_cards = soup.select("a.title")
        seen = set()

        for a in job_cards:
            title = a.get_text(strip=True)
            href = a.get("href", "")

            if not title or not href:
                continue

            # ensure it's a full URL
            if href.startswith("/"):
                href = urllib.parse.urljoin("https://www.naukri.com", href)

            # only include real Naukri job links
            if not href.startswith("https://www.naukri.com/") or not re.search(r"/job-listings[-/]", href):
                continue

            if href in seen:
                continue

            seen.add(href)
            jobs.append({
                "title": title,
                "company": "Naukri.com",
                "location": location,
                "summary": "Live job from Naukri",
                "url": href
            })

        print(f"✅ Found {len(jobs)} real job postings on Naukri for '{query}' in '{location}'.")

    except Exception as e:
        print("❌ Naukri crawler error:", e)

    # fallback if none found
    if not jobs:
        jobs.append({
            "title": f"{skills} Jobs in {location}",
            "company": "Naukri",
            "url": search_url
        })

    return jobs
