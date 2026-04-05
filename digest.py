import os
import re
import time
import html
import json
import requests
from bs4 import BeautifulSoup
import sendgrid
from sendgrid.helpers.mail import Mail

# =========================
# Configuration
# =========================
CATEGORIES = ["nucl-ex", "nucl-th"]
THRESHOLD = 4
DEEP_DIVE_LIMIT = 5
USER_AGENT = "ArXivBot/1.0"
ARXIV_HEADERS = {"User-Agent": USER_AGENT}

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

def heuristic_score(title: str, abstract: str):
    text = clean_text(f"{title} {abstract}").lower()
    weights = {
        "star": 4, "rhic": 4, "sphenix": 4, "eic": 4, "heavy ion": 3,
        "heavy-ion": 3, "hyperon": 2, "polarization": 2, "lambda": 1,
        "xi": 1, "omega": 1, "bes-ii": 3, "beam energy scan": 3,
        "cme": 2, "anisotropic flow": 3, "elliptic flow": 2, "flow": 1,
        "qcd": 2, "critical point": 2, "cgc": 2, "glasma": 2, "jets": 2,
        "jet quenching": 3, "heavy flavor": 2, "j/psi": 2, "photoproduction": 2,
        "upc": 2, "pomeron": 2, "femtoscopy": 2, "hbt": 2, "gdr": 2,
        "nuclear deformation": 2, "machine learning": 1, "ml": 1,
    }

    raw = 0
    hits = []
    for kw, w in weights.items():
        if kw in text:
            raw += w
            hits.append(kw)

    score = max(1, min(10, 1 + raw))
    summary = f"Heuristic match via: {', '.join(hits[:4])}." if hits else clean_text(abstract)[:160]

    if any(x in text for x in ["star", "rhic", "sphenix"]):
        star = "Direct RHIC/STAR/sPHENIX relevance."
    elif any(x in text for x in ["hyperon", "polarization", "lambda", "xi", "omega"]):
        star = "Potential comparison to STAR hyperon polarization measurements."
    elif any(x in text for x in ["flow", "anisotropic flow", "elliptic flow"]):
        star = "Potential comparison to STAR flow observables."
    elif any(x in text for x in ["upc", "photoproduction", "j/psi", "pomeron"]):
        star = "Possible UPC / J/psi photoproduction relevance for STAR/EIC."
    else:
        star = "N/A"

    return score, summary, star

def parse_json_text(raw: str):
    start = raw.find('{')
    end = raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end+1]

    try:
        obj = json.loads(raw)
        score = int(obj.get("score", 1))
        return max(1, min(10, score)), clean_text(obj.get("summary", "")), clean_text(obj.get("star_angle", "N/A"))
    except:
        return 1, "JSON Parsing Error", "N/A"

def extract_main_text_from_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return clean_text(main.get_text(separator=" ", strip=True))[:15000]

def get_html_text(arxiv_url: str) -> str:
    html_url = arxiv_url.replace("/abs/", "/html/")
    try:
        r = requests.get(html_url, headers=ARXIV_HEADERS, timeout=20)
        if r.status_code != 200:
            return ""
        return extract_main_text_from_html(r.text)
    except:
        return ""

def call_gemini(prompt: str, system_instr: str):
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable")

    active_model = "gemini-2.0-flash-lite"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:generateContent?key={api_key}"

    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instr}]
        },
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json"
        }
    }

    try:
        r = requests.post(endpoint, json=payload, timeout=45)
        r.raise_for_status()
        data = r.json()

        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("No candidates in response")

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts or "text" not in parts[0]:
            raise RuntimeError("No text in response")

        return parts[0]["text"]
    except Exception as e:
        raise RuntimeError(f"API request failed: {str(e)[:200]}")

def score_paper(title: str, abstract: str, url: str):
    system_instr = (
        "You are a STAR Experiment physicist at BNL. "
        "Rank arXiv papers from 1 to 10 for relevance to STAR/RHIC/EIC heavy-ion physics. "
        "Prefer higher scores for Lambda/hyperon polarization, BES-II, FXT, CME, flow, "
        "critical-point searches, jets, heavy flavor, UPC, J/psi photoproduction, "
        "femtoscopy, nuclear deformation, and observables testable with STAR or sPHENIX."
    )

    base_prompt = f"""
Return ONLY valid JSON with exactly these keys:
{{
  "score": 7,
  "summary": "one-sentence plain-English summary",
  "star_angle": "specific STAR/sPHENIX/EIC testable angle or N/A"
}}

Title: {title}
Abstract: {abstract}
""".strip()

    raw = call_gemini(base_prompt, system_instr)
    score, summary, star_angle = parse_json_text(raw)

    if score >= DEEP_DIVE_LIMIT:
        full_text = get_html_text(url)
        if full_text:
            deep_prompt = f"""
Return ONLY valid JSON with exactly these keys:
{{
  "score": {score},
  "summary": "improved one-sentence summary",
  "star_angle": "specific STAR/sPHENIX/EIC testable angle or N/A"
}}

Use the paper text below to refine the previous ranking and especially the STAR angle.

Title: {title}
Abstract: {abstract}
Paper HTML text:
{full_text}
""".strip()
            deep_raw = call_gemini(deep_prompt, system_instr)
            score, summary, star_angle = parse_json_text(deep_raw)

    return score, summary, star_angle

def fetch_papers_for_category(session: requests.Session, cat: str):
    url = f"https://arxiv.org/list/{cat}/recent"
    r = session.get(url, headers=ARXIV_HEADERS, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    papers = []

    for dt, dd in zip(soup.find_all("dt"), soup.find_all("dd")):
        link = dt.find("a", title="Abstract")
        title_tag = dd.find("div", class_="list-title")
        abstract_tag = dd.find("p", class_="mathjax")
        authors_tag = dd.find("div", class_="list-authors")

        if not link or not title_tag:
            continue

        papers.append({
            "id": "https://arxiv.org" + link["href"],
            "title": clean_text(title_tag.get_text(" ", strip=True).replace("Title:", "")),
            "abstract": clean_text(abstract_tag.get_text(" ", strip=True) if abstract_tag else ""),
            "authors": clean_text(authors_tag.get_text(" ", strip=True).replace("Authors:", "") if authors_tag else "")[:180],
            "cat": cat,
        })
    return papers

def build_email_html(hits, total_papers):
    body = [f"<h2>ArXiv Digest -- STAR Physics ({len(hits)} papers)</h2>"]
    body.append(f"<p>Scanned {total_papers} papers across {html.escape(', '.join(CATEGORIES))}.</p><hr>")

    if not hits:
        body.append(f"<p>No papers scored at or above {THRESHOLD}.</p>")
        return "".join(body)

    for h in hits:
        body.append(
            f"""
            <p>
                <b>[{h['score']}/10]</b>
                <a href="{html.escape(h['id'])}">{html.escape(h['title'])}</a><br>
                <i>{html.escape(h['authors'])} | {html.escape(h['cat'])}</i><br>
                {html.escape(h['summary'])}<br>
                <b>STAR angle:</b> {html.escape(h['star'])}
            </p>
            <hr>
            """
        )
    return "".join(body)

def main():
    required_env = ["GEMINI_API_KEY", "SENDGRID_API_KEY", "FROM_EMAIL", "TO_EMAIL"]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

    session = requests.Session()
    all_papers = []

    for cat in CATEGORIES:
        try:
            papers = fetch_papers_for_category(session, cat)
            print(f"Fetched {len(papers)} papers from {cat}")
            all_papers.extend(papers)
        except Exception as e:
            print(f"Failed fetching {cat}: {e}")

    unique_papers = list({p["id"]: p for p in all_papers}.values())
    print(f"Fetched {len(unique_papers)} unique papers total")
    print(f"Gemini key active: {bool(os.environ.get('GEMINI_API_KEY'))}")

    hits = []
    failures = 0

    for i, p in enumerate(unique_papers, start=1):
        try:
            score, summary, star = score_paper(p["title"], p["abstract"], p["id"])
        except Exception as e:
            failures += 1
            print(f"Gemini failed on {p['title'][:80]}: {e}")
            score, summary, star = heuristic_score(p["title"], p["abstract"])

        print(f"{i:03d}/{len(unique_papers)} | {score}/10 | {p['title'][:90]}")

        if score >= THRESHOLD:
            hits.append({
                **p,
                "score": score,
                "summary": summary,
                "star": star,
            })
        time.sleep(0.5)

    hits.sort(key=lambda x: (-x["score"], x["title"].lower()))
    print(f"Found {len(hits)} relevant papers")
    print(f"Model failures: {failures}")

    html_body = build_email_html(hits, len(unique_papers))

    sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
    mail = Mail(
        from_email=os.environ["FROM_EMAIL"],
        to_emails=os.environ["TO_EMAIL"],
        subject=f"ArXiv Digest: {len(hits)} STAR-relevant papers",
        html_content=html_body,
    )
    sg.send(mail)
    print("Email sent.")

if __name__ == "__main__":
    main()
