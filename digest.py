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
HEURISTIC_THRESHOLD = 4
EMAIL_THRESHOLD = 4
USER_AGENT = "ArXivBot/1.0"
ARXIV_HEADERS = {"User-Agent": USER_AGENT}

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()

# =========================
# Heuristic Scorer
# =========================
def heuristic_score(title: str, abstract: str):
    text = clean_text(f"{title} {abstract}").lower()
    weights = {
        "star detector": 5, "star experiment": 5, "rhic": 4, "sphenix": 4,
        "eic": 4, "electron-ion collider": 4,
        "chiral magnetic effect": 5, "cme": 4,
        "photonuclear": 5, "photoproduction": 4, "upc": 4,
        "ultra-peripheral": 4, "j/psi": 3, "pomeron": 3,
        "hyperon polarization": 5, "lambda polarization": 5,
        "global polarization": 4,
        "bes-ii": 4, "beam energy scan": 4, "fxt": 3,
        "critical point": 3, "qcd critical": 3,
        "baryon stopping": 3, "net-proton": 3,
        "anisotropic flow": 3, "elliptic flow": 3, "v2": 2,
        "triangular flow": 2,
        "femtoscopy": 3, "hbt": 3, "source size": 2,
        "heavy flavor": 3, "charm quark": 2, "bottom quark": 2,
        "upsilon": 3, "charmonium": 3, "quarkonia": 3,
        "jet quenching": 3, "jets": 2, "di-jet": 3,
        "parton energy loss": 3,
        "heavy ion": 2, "heavy-ion": 2, "quark-gluon plasma": 3,
        "qgp": 3, "glasma": 2, "cgc": 2,
        "nuclear deformation": 2, "polarization": 1,
        "hyperon": 2, "lambda": 1, "xi": 1, "omega": 1,
        "vorticity": 2, "spin alignment": 3,
        "machine learning": 1,
    }

    raw, hits = 0, []
    for kw, w in weights.items():
        if kw in text:
            raw += w
            hits.append(kw)

    score = max(1, min(10, 1 + raw))
    summary = f"Keyword matches: {', '.join(hits[:6])}." if hits else clean_text(abstract)[:200]

    if any(x in text for x in ["star detector", "star experiment", "rhic", "sphenix"]):
        star = "Direct RHIC/STAR/sPHENIX relevance."
    elif any(x in text for x in ["hyperon polarization", "lambda polarization", "global polarization"]):
        star = "Direct connection to STAR hyperon polarization program."
    elif any(x in text for x in ["chiral magnetic effect", "cme"]):
        star = "Direct connection to STAR CME search program."
    elif any(x in text for x in ["photonuclear", "photoproduction", "upc", "ultra-peripheral"]):
        star = "Relevant to STAR/EIC UPC and photonuclear program."
    elif any(x in text for x in ["bes-ii", "beam energy scan", "fxt", "critical point"]):
        star = "Relevant to STAR BES-II / critical point search."
    elif any(x in text for x in ["flow", "anisotropic flow", "elliptic flow"]):
        star = "Comparison to STAR flow measurements."
    elif any(x in text for x in ["femtoscopy", "hbt"]):
        star = "Relevant to STAR femtoscopy program."
    else:
        star = "N/A"

    return score, summary, star

# =========================
# Gemini
# =========================
def call_gemini(prompt: str, system_instr: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        "gemini-2.0-flash-lite:generateContent?key=" + api_key
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_instr}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    r = requests.post(endpoint, json=payload, timeout=45)
    if r.status_code == 429:
        raise RuntimeError("Rate limited (429)")
    r.raise_for_status()
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    if not parts or "text" not in parts[0]:
        raise RuntimeError("No text in response")
    return parts[0]["text"]

def parse_deep_json(raw: str):
    start, end = raw.find('{'), raw.rfind('}')
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    try:
        obj = json.loads(raw)
        score = max(1, min(10, int(obj.get("score", 1))))
        return (
            score,
            clean_text(obj.get("summary", "")),
            clean_text(obj.get("star_angle", "N/A")),
            clean_text(obj.get("key_results", "")),
            clean_text(obj.get("what_you_learn", "")),
            clean_text(obj.get("followup_ideas", "")),
        )
    except:
        return None

def extract_main_text_from_html(html_text: str) -> str:
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    main = soup.find("article") or soup.find("main") or soup.body or soup
    return clean_text(main.get_text(separator=" ", strip=True))[:20000]

def get_html_text(arxiv_url: str) -> str:
    html_url = arxiv_url.replace("/abs/", "/html/")
    try:
        r = requests.get(html_url, headers=ARXIV_HEADERS, timeout=20)
        if r.status_code != 200:
            return ""
        return extract_main_text_from_html(r.text)
    except:
        return ""

def gemini_deep_dive(title: str, abstract: str, url: str):
    full_text = get_html_text(url)
    system_instr = (
        "You are a STAR Experiment physicist at BNL. "
        "Evaluate arXiv papers for relevance to STAR/RHIC/EIC heavy-ion physics. "
        "Topics of highest interest: CME, hyperon/Lambda polarization, BES-II, FXT, "
        "photonuclear/UPC/J/psi photoproduction, femtoscopy, flow, jet quenching, "
        "heavy flavor, quarkonia, spin alignment, vorticity, critical point search."
    )
    prompt = f"""
Read this paper carefully and return ONLY valid JSON with exactly these keys:
{{
  "score": <1-10 integer>,
  "summary": "<2-3 sentence summary of what the paper does and its main findings>",
  "star_angle": "<specific STAR/sPHENIX/EIC observable or measurement this connects to, or N/A>",
  "key_results": "<the 2-3 most important quantitative or qualitative results>",
  "what_you_learn": "<what a STAR physicist would take away from reading this>",
  "followup_ideas": "<2-3 concrete research directions or measurements this paper motivates>"
}}

Title: {title}
Abstract: {abstract}
Full paper text:
{full_text if full_text else "(HTML not available — use abstract only)"}
""".strip()

    raw = call_gemini(prompt, system_instr)
    return parse_deep_json(raw)

# =========================
# Paper Fetching
# =========================
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
            "authors": clean_text(authors_tag.get_text(" ", strip=True).replace("Authors:", "") if authors_tag else "")[:200],
            "cat": cat,
        })
    return papers

# =========================
# Email Builder
# =========================
def build_email_html(hits, total_papers):
    body = [
        "<h2>ArXiv Digest — STAR Physics</h2>",
        f"<p><b>{len(hits)} relevant papers</b> out of {total_papers} scanned "
        f"across {html.escape(', '.join(CATEGORIES))}.</p><hr>",
    ]
    if not hits:
        body.append(f"<p>No papers scored at or above {EMAIL_THRESHOLD} today.</p>")
        return "".join(body)

    for h in hits:
        tag = "Gemini" if h.get("gemini_scored") else "Heuristic"
        body.append(
            f"<p>"
            f"<b>[{h['score']}/10] [{tag}]</b> "
            f"<a href=\"{html.escape(h['id'])}\">{html.escape(h['title'])}</a><br>"
            f"<i>{html.escape(h['authors'])} | {html.escape(h['cat'])}</i><br><br>"
            f"<b>Summary:</b> {html.escape(h['summary'])}<br>"
        )
        if h.get("key_results"):
            body.append(f"<b>Key results:</b> {html.escape(h['key_results'])}<br>")
        if h.get("what_you_learn"):
            body.append(f"<b>What you learn:</b> {html.escape(h['what_you_learn'])}<br>")
        body.append(f"<b>STAR angle:</b> {html.escape(h['star'])}<br>")
        if h.get("followup_ideas"):
            body.append(f"<b>Follow-up ideas:</b> {html.escape(h['followup_ideas'])}<br>")
        body.append("</p><hr>")

    return "".join(body)

# =========================
# Main
# =========================
def main():
    required_env = ["GEMINI_API_KEY", "SENDGRID_API_KEY", "FROM_EMAIL", "TO_EMAIL"]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")

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
    print(f"Total unique papers: {len(unique_papers)}")

    hits = []
    gemini_success = 0
    gemini_fail = 0

    for i, p in enumerate(unique_papers, start=1):
        h_score, h_summary, h_star = heuristic_score(p["title"], p["abstract"])
        print(f"{i:03d}/{len(unique_papers)} | heuristic={h_score}/10 | {p['title'][:80]}")

        if h_score < HEURISTIC_THRESHOLD:
            continue

        # Gemini deep dive on papers that pass heuristic
        gemini_result = None
        try:
            gemini_result = gemini_deep_dive(p["title"], p["abstract"], p["id"])
            if gemini_result:
                gemini_success += 1
                print(f"    Gemini={gemini_result[0]}/10")
        except Exception as e:
            gemini_fail += 1
            print(f"    Gemini failed: {e}")

        if gemini_result:
            score, summary, star, key_results, what_you_learn, followup_ideas = gemini_result
            gemini_scored = True
        else:
            score, summary, star = h_score, h_summary, h_star
            key_results = what_you_learn = followup_ideas = ""
            gemini_scored = False

        if score >= EMAIL_THRESHOLD:
            hits.append({
                **p,
                "score": score,
                "summary": summary,
                "star": star,
                "key_results": key_results,
                "what_you_learn": what_you_learn,
                "followup_ideas": followup_ideas,
                "gemini_scored": gemini_scored,
            })

        time.sleep(1)  # light throttle between Gemini calls

    hits.sort(key=lambda x: (-x["score"], x["title"].lower()))
    print(f"Found {len(hits)} relevant papers | Gemini OK={gemini_success} fail={gemini_fail}")

    html_body = build_email_html(hits, len(unique_papers))

    sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
    mail = Mail(
        from_email=os.environ["FROM_EMAIL"],
        to_emails=os
