import os
import re
import time
import html
import json
import hashlib
import requests
from bs4 import BeautifulSoup
import sendgrid
from sendgrid.helpers.mail import Mail

# =========================
# Configuration
# =========================
CATEGORIES = ["hep-ex", "nucl-ex", "hep-ph"]
THRESHOLD = 5          # Gemini score gate (1-10)
DEEP_DIVE_LIMIT = 8    # Max papers sent to Gemini for deep scoring
USER_AGENT = "ArXivDigestBot/2.0"
ARXIV_HEADERS = {"User-Agent": USER_AGENT}

# Seen-papers cache file (persists across runs via artifact or local file)
SEEN_CACHE = os.environ.get("SEEN_CACHE_PATH", "/tmp/seen_papers.json")


# =========================
# Helpers
# =========================

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def load_seen() -> set:
    try:
        with open(SEEN_CACHE) as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_seen(seen: set):
    try:
        with open(SEEN_CACHE, "w") as f:
            json.dump(list(seen), f)
    except Exception:
        pass


# =========================
# Heuristic scoring
# =========================

def heuristic_score(title: str, abstract: str):
    text = clean_text(f"{title} {abstract}").lower()
    weights = {
        # Core STAR/RHIC
        "star": 5, "rhic": 5, "sphenix": 4, "phenix": 3, "brahms": 2, "phobos": 2,
        "eic": 3, "bes-ii": 4, "beam energy scan": 4, "fxt": 3,
        # UPC / photoproduction
        "upc": 5, "ultra-peripheral": 5, "photoproduction": 5, "photonuclear": 5,
        "coherent": 3, "incoherent": 3, "j/psi": 4, "jpsi": 4, "upsilon": 3,
        "pomeron": 3, "odderon": 3, "nuclear shadowing": 3, "saturation": 2,
        "cgc": 2, "bk equation": 2, "ipsat": 2,
        # Event-shape engineering / flow
        "event-shape engineering": 6, "ese": 4, "event shape": 3,
        "flow fluctuation": 5, "flow fluctuations": 5, "eccentricity fluctuation": 4,
        "anisotropic flow": 4, "elliptic flow": 3, "triangular flow": 3,
        "v2": 2, "v3": 2, "v4": 2, "cumulant": 3, "two-particle correlation": 2,
        "non-linear": 2, "mode coupling": 3, "linear response": 2,
        "initial state": 2, "initial geometry": 3, "ip-glasma": 3, "trento": 2,
        "viscous hydrodynamics": 2, "hydro": 1,
        # Other STAR topics (lower weight)
        "hyperon": 2, "lambda polarization": 3, "global polarization": 4,
        "vorticity": 2, "spin alignment": 3, "chiral magnetic": 4, "cme": 3,
        "heavy flavor": 2, "jet quenching": 2, "femtoscopy": 2, "hbt": 2,
        "nuclear deformation": 2, "qcd critical point": 3,
        # General
        "heavy ion": 2, "heavy-ion": 2, "gold": 1, "au+au": 3, "pbpb": 2,
        "pb+pb": 2, "d+au": 2, "p+pb": 1,
    }
    raw = 0
    hits = []
    for kw, w in weights.items():
        if kw in text:
            raw += w
            hits.append(kw)
    score = max(1, min(10, 1 + raw // 2))
    summary = f"Heuristic match: {', '.join(hits[:6])}." if hits else clean_text(abstract)[:200]

    if any(x in text for x in ["upc", "ultra-peripheral", "photoproduction", "photonuclear", "j/psi", "jpsi"]):
        star = "UPC / J/psi photoproduction relevance for STAR."
    elif any(x in text for x in ["event-shape engineering", "ese", "flow fluctuation", "flow fluctuations", "eccentricity fluctuation"]):
        star = "Event-shape engineering or flow fluctuations — direct STAR ESE/flow program."
    elif any(x in text for x in ["star", "rhic", "sphenix"]):
        star = "Direct STAR/RHIC/sPHENIX relevance."
    elif any(x in text for x in ["anisotropic flow", "elliptic flow", "cumulant"]):
        star = "Flow observables — potential comparison to STAR measurements."
    elif any(x in text for x in ["cme", "chiral magnetic"]):
        star = "CME search — direct STAR program connection."
    else:
        star = "N/A"

    return score, summary, star, hits


# =========================
# INSPIRE-HEP search
# =========================

def fetch_inspire_papers(topics: list, days: int = 1) -> list:
    """Query INSPIRE-HEP API for recent papers on given topics."""
    papers = []
    base = "https://inspirehep.net/api/literature"
    # Build query: papers from last `days` days matching any topic keyword
    q_terms = " OR ".join(f'a "{t}"' if " " in t else t for t in topics)
    params = {
        "sort": "mostrecent",
        "size": 25,
        "page": 1,
        "fields": "arxiv_eprints,titles,abstracts,authors,publication_info,citation_count",
        "q": f"({q_terms}) AND date>{days}d",
    }
    try:
        r = requests.get(base, params=params, headers=ARXIV_HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        for hit in data.get("hits", {}).get("hits", []):
            meta = hit.get("metadata", {})
            eprints = meta.get("arxiv_eprints", [])
            if not eprints:
                continue
            arxiv_id = eprints[0].get("value", "")
            if not arxiv_id:
                continue
            titles = meta.get("titles", [])
            title = titles[0].get("title", "") if titles else ""
            abstracts = meta.get("abstracts", [])
            abstract = abstracts[0].get("value", "") if abstracts else ""
            authors_list = meta.get("authors", [])
            authors = ", ".join(
                a.get("full_name", "") for a in authors_list[:5]
            ) + (" et al." if len(authors_list) > 5 else "")
            cite_count = meta.get("citation_count", 0)
            papers.append({
                "id": f"https://arxiv.org/abs/{arxiv_id}",
                "arxiv_id": arxiv_id,
                "title": clean_text(title),
                "abstract": clean_text(abstract),
                "authors": clean_text(authors)[:200],
                "cat": "inspire",
                "cite_count": cite_count,
                "source": "INSPIRE-HEP",
            })
    except Exception as e:
        print(f"INSPIRE fetch failed: {e}")
    return papers


# =========================
# CERN CDS search
# =========================

def fetch_cds_papers(topics: list) -> list:
    """Search CERN CDS for recent records matching topics."""
    papers = []
    base = "https://cds.cern.ch/search"
    query = " OR ".join(f'"{t}"' for t in topics)
    params = {
        "p": query,
        "f": "",
        "action_search": "Search",
        "c": "",
        "sf": "year",
        "so": "d",
        "rm": "",
        "rg": 20,
        "sc": 0,
        "of": "hx",  # MARCXML output
        "ln": "en",
    }
    try:
        r = requests.get(base, params=params, headers=ARXIV_HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        for record in soup.find_all("record")[:15]:
            # Extract arXiv ID if present (field 037 with source arXiv)
            arxiv_id = None
            for f037 in record.find_all("datafield", tag="037"):
                src = f037.find("subfield", code="9")
                val = f037.find("subfield", code="a")
                if src and "arxiv" in src.get_text().lower() and val:
                    arxiv_id = val.get_text().replace("arXiv:", "").strip()
                    break
            # Title (245 $a)
            t245 = record.find("datafield", tag="245")
            title = t245.find("subfield", code="a").get_text() if t245 else ""
            # Abstract (520 $a)
            t520 = record.find("datafield", tag="520")
            abstract = t520.find("subfield", code="a").get_text() if t520 else ""
            if not title:
                continue
            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
            papers.append({
                "id": url or f"https://cds.cern.ch/search?p={clean_text(title)[:60]}",
                "arxiv_id": arxiv_id or "",
                "title": clean_text(title),
                "abstract": clean_text(abstract),
                "authors": "",
                "cat": "cds",
                "cite_count": 0,
                "source": "CERN CDS",
            })
    except Exception as e:
        print(f"CDS fetch failed: {e}")
    return papers


# =========================
# arXiv fetching
# =========================

def fetch_papers_for_category(session: requests.Session, cat: str) -> list:
    url = f"https://arxiv.org/list/{cat}/new"
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
            "arxiv_id": link["href"].replace("/abs/", ""),
            "title": clean_text(title_tag.get_text(" ", strip=True).replace("Title:", "")),
            "abstract": clean_text(abstract_tag.get_text(" ", strip=True) if abstract_tag else ""),
            "authors": clean_text(authors_tag.get_text(" ", strip=True).replace("Authors:", "") if authors_tag else "")[:200],
            "cat": cat,
            "cite_count": 0,
            "source": "arXiv",
        })
    return papers


# =========================
# Full-text and plot extraction
# =========================

def get_html_text(arxiv_url: str) -> str:
    html_url = arxiv_url.replace("/abs/", "/html/")
    try:
        r = requests.get(html_url, headers=ARXIV_HEADERS, timeout=25)
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        main = soup.find("article") or soup.find("main") or soup.body or soup
        return clean_text(main.get_text(separator=" ", strip=True))[:18000]
    except Exception:
        return ""


def extract_figures_tables(arxiv_url: str) -> list:
    """Extract figure/table captions from the HTML version of the paper."""
    html_url = arxiv_url.replace("/abs/", "/html/")
    captions = []
    try:
        r = requests.get(html_url, headers=ARXIV_HEADERS, timeout=25)
        if r.status_code != 200:
            return captions
        soup = BeautifulSoup(r.text, "html.parser")
        for fig in soup.find_all("figure")[:20]:
            cap = fig.find("figcaption")
            if cap:
                text = clean_text(cap.get_text(" ", strip=True))
                if len(text) > 20:
                    captions.append(("Figure", text[:300]))
        for caption_tag in soup.find_all(["caption"])[:10]:
            text = clean_text(caption_tag.get_text(" ", strip=True))
            if len(text) > 20:
                captions.append(("Table", text[:300]))
    except Exception:
        pass
    return captions[:12]


# =========================
# Gemini LLM scoring
# =========================

def call_gemini(prompt: str, system_instr: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    model = "gemini-2.0-flash-lite"
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_instr}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    r = requests.post(endpoint, json=payload, timeout=50)
    if r.status_code == 429:
        raise RuntimeError("Rate limited (429)")
    r.raise_for_status()
    data = r.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise RuntimeError("No candidates")
    parts = candidates[0].get("content", {}).get("parts", [])
    if not parts or "text" not in parts[0]:
        raise RuntimeError("No text in response")
    return parts[0]["text"]


def parse_json_text(raw: str):
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
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
            clean_text(obj.get("cite_worthy_plots", "")),
            obj.get("is_genuinely_new", False),
        )
    except Exception:
        return 1, "JSON parse error", "N/A", "", "", "", "", False


def score_paper(title: str, abstract: str, url: str, captions: list):
    system_instr = (
        "You are a STAR Experiment physicist at BNL specialising in UPC, J/psi "
        "photoproduction, event-shape engineering, and flow fluctuations at RHIC energies. "
        "Score arXiv papers 1-10 for relevance to STAR-era analyses. "
        "Prefer HIGH scores for: direct STAR/RHIC UPC or photoproduction data, "
        "new ESE methodology, flow-fluctuation theory/measurement at RHIC energies, "
        "Pomeron/CGC models testable with STAR. "
        "is_genuinely_new = true only if the paper presents a result not previously published "
        "(not a review, not a proceedings summary of known results)."
    )
    full_text = get_html_text(url)
    captions_text = "\n".join(f"[{kind} caption] {c}" for kind, c in captions) if captions else "(not available)"
    prompt = (
        "Return ONLY valid JSON with exactly these keys:\n"
        "{\n"
        '  "score": <1-10 integer>,\n'
        '  "summary": "<2-3 sentence summary of what the paper does and its main finding>",\n'
        '  "star_angle": "<specific STAR/RHIC/EIC observable this connects to, or N/A>",\n'
        '  "key_results": "<2-3 most important quantitative or qualitative results>",\n'
        '  "what_you_learn": "<what a STAR UPC/ESE/flow physicist takes away>",\n'
        '  "followup_ideas": "<2-3 concrete measurements or directions this motivates>",\n'
        '  "cite_worthy_plots": "<figure/table numbers and one-sentence description of the 1-2 plots or tables most worth citing — e.g. Fig 3 (J/psi |t| spectrum) and Table 1 (cross-section values)>",\n'
        '  "is_genuinely_new": <true|false>\n'
        "}\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
        f"Figure/Table captions:\n{captions_text}\n"
        f"Full paper text:\n{full_text if full_text else '(HTML not available)'}"
    )
    raw = call_gemini(prompt, system_instr)
    return parse_json_text(raw)


# =========================
# Novelty / influence gate
# =========================

def passes_notification_gate(hits: list) -> bool:
    """Return True only if at least one paper is genuinely new, unusually
    influential (high cite_count from INSPIRE), or scores very highly."""
    for h in hits:
        if h.get("is_genuinely_new"):
            return True
        if h.get("cite_count", 0) >= 10:   # already picking up citations fast
            return True
        if h.get("score", 0) >= 8:
            return True
    return False


# =========================
# Email builder
# =========================

def build_email_html(hits: list, total_papers: int, sources_used: list) -> str:
    body = [
        f"<h2>arXiv Digest — STAR UPC/ESE/Flow ({len(hits)} papers)</h2>",
        f"<p>Scanned {total_papers} unique papers from "
        f"{html.escape(', '.join(CATEGORIES))} + {html.escape(', '.join(sources_used))}.</p><hr>",
    ]
    if not hits:
        body.append(f"<p>No papers scored ≥ {THRESHOLD}/10 today.</p>")
        return "".join(body)

    for h in hits:
        scored_by = "Gemini" if h.get("gemini_scored") else "Heuristic"
        new_flag = " 🆕" if h.get("is_genuinely_new") else ""
        cite_str = f" | {h['cite_count']} citations" if h.get("cite_count", 0) > 0 else ""
        body.append(
            f"<p><b>[{h['score']}/10] [{scored_by}]{new_flag}</b> "
            f"<a href=\"{html.escape(h['id'])}\">{html.escape(h['title'])}</a><br>"
            f"<i>{html.escape(h['authors'])} | {html.escape(h['cat'])}"
            f"{html.escape(cite_str)}</i><br><br>"
        )
        body.append(f"<b>Summary:</b> {html.escape(h['summary'])}<br>")
        if h.get("key_results"):
            body.append(f"<b>Key results:</b> {html.escape(h['key_results'])}<br>")
        if h.get("what_you_learn"):
            body.append(f"<b>What you learn:</b> {html.escape(h['what_you_learn'])}<br>")
        if h.get("cite_worthy_plots"):
            body.append(f"<b>Plots/tables to cite:</b> {html.escape(h['cite_worthy_plots'])}<br>")
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
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}")

    topics = [
        "UPC", "ultra-peripheral", "J/psi photoproduction", "photonuclear",
        "event-shape engineering", "flow fluctuations", "eccentricity fluctuations",
        "anisotropic flow RHIC", "Pomeron", "STAR RHIC",
    ]

    # --- Fetch from all three sources ---
    session = requests.Session()
    all_papers = []
    sources_used = []

    # arXiv
    for cat in CATEGORIES:
        try:
            papers = fetch_papers_for_category(session, cat)
            print(f"arXiv {cat}: {len(papers)} papers")
            all_papers.extend(papers)
        except Exception as e:
            print(f"arXiv {cat} failed: {e}")
    sources_used.append("arXiv")

    # INSPIRE-HEP
    inspire_papers = fetch_inspire_papers(topics, days=2)
    print(f"INSPIRE-HEP: {len(inspire_papers)} papers")
    all_papers.extend(inspire_papers)
    if inspire_papers:
        sources_used.append("INSPIRE-HEP")

    # CERN CDS
    cds_papers = fetch_cds_papers(topics[:5])
    print(f"CERN CDS: {len(cds_papers)} papers")
    all_papers.extend(cds_papers)
    if cds_papers:
        sources_used.append("CERN CDS")

    # Deduplicate by arxiv_id, then by title hash
    seen_ids = {}
    for p in all_papers:
        key = p.get("arxiv_id") or hashlib.md5(p["title"].encode()).hexdigest()
        if key not in seen_ids:
            seen_ids[key] = p
        else:
            # Keep whichever has more metadata
            existing = seen_ids[key]
            if not existing.get("abstract") and p.get("abstract"):
                seen_ids[key] = p
    unique_papers = list(seen_ids.values())
    print(f"Total unique papers: {len(unique_papers)}")

    # Load seen-papers cache to avoid re-notifying
    seen_cache = load_seen()

    # --- Heuristic pass ---
    candidates = []
    for p in unique_papers:
        if p["id"] in seen_cache:
            continue
        h_score, h_summary, h_star, h_hits = heuristic_score(p["title"], p["abstract"])
        p["h_score"] = h_score
        p["h_summary"] = h_summary
        p["h_star"] = h_star
        if h_score >= THRESHOLD - 1:   # slightly lower bar to not miss anything
            candidates.append(p)

    # Sort candidates: STAR/UPC/ESE papers first, then by heuristic score
    candidates.sort(key=lambda x: -x["h_score"])
    print(f"Candidates for deep scoring: {len(candidates)}")

    # --- Gemini deep scoring (top N candidates) ---
    hits = []
    failures = 0
    deep_scored = 0

    for p in candidates:
        if deep_scored >= DEEP_DIVE_LIMIT:
            # Use heuristic for remaining candidates
            if p["h_score"] >= THRESHOLD:
                hits.append({
                    **p,
                    "score": p["h_score"],
                    "summary": p["h_summary"],
                    "star": p["h_star"],
                    "key_results": "",
                    "what_you_learn": "",
                    "followup_ideas": "",
                    "cite_worthy_plots": "",
                    "is_genuinely_new": False,
                    "gemini_scored": False,
                })
            continue

        captions = extract_figures_tables(p["id"]) if "/abs/" in p["id"] else []
        try:
            score, summary, star, key_results, what_you_learn, followup_ideas, cite_worthy_plots, is_new = score_paper(
                p["title"], p["abstract"], p["id"], captions
            )
            gemini_scored = True
            deep_scored += 1
            print(f"  Gemini={score}/10 | new={is_new} | {p['title'][:80]}")
        except Exception as e:
            failures += 1
            print(f"  Gemini failed ({e}) — heuristic fallback")
            score, summary, star = p["h_score"], p["h_summary"], p["h_star"]
            key_results = what_you_learn = followup_ideas = cite_worthy_plots = ""
            is_new = False
            gemini_scored = False

        if score >= THRESHOLD:
            hits.append({
                **p,
                "score": score,
                "summary": summary,
                "star": star,
                "key_results": key_results,
                "what_you_learn": what_you_learn,
                "followup_ideas": followup_ideas,
                "cite_worthy_plots": cite_worthy_plots,
                "is_genuinely_new": is_new,
                "gemini_scored": gemini_scored,
            })

        time.sleep(1.2)

    hits.sort(key=lambda x: (-x["score"], x["title"].lower()))
    print(f"Relevant papers: {len(hits)} | Gemini failures: {failures}")

    # --- Novelty / influence gate: only email if something is worth it ---
    if not hits:
        print("No relevant papers — skipping email.")
        return

    if not passes_notification_gate(hits):
        print("Papers found but none pass novelty/influence gate — skipping email.")
        return

    # --- Build and send email ---
    html_body = build_email_html(hits, len(unique_papers), sources_used)

    sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
    mail = Mail(
        from_email=os.environ["FROM_EMAIL"],
        to_emails=os.environ["TO_EMAIL"],
        subject=f"arXiv Digest: {len(hits)} STAR UPC/ESE/flow papers",
        html_content=html_body,
    )
    sg.send(mail)
    print("Email sent.")

    # --- Update seen-papers cache ---
    new_seen = seen_cache | {p["id"] for p in hits}
    save_seen(new_seen)


if __name__ == "__main__":
    main()
