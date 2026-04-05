"""
arXiv Digest — Prithwish Tribedy / STAR RHIC
Scans arXiv (hep-ex, nucl-ex, hep-ph, nucl-th), INSPIRE-HEP, and CERN CDS.
Elevated scoring for papers by close collaborators regardless of topic.
"""

import os
import re
import time
import html
import json
import yaml
import hashlib
import requests
from bs4 import BeautifulSoup
import sendgrid
from sendgrid.helpers.mail import Mail

# =========================
# Configuration
# =========================
with open(os.path.join(os.path.dirname(__file__), "config.yaml")) as f:
    CONFIG = yaml.safe_load(f)

CATEGORIES    = CONFIG.get("subjects", ["hep-ex", "nucl-ex", "hep-ph", "nucl-th"])
THRESHOLD     = CONFIG.get("threshold", 5)
DEEP_LIMIT    = 10          # max Gemini calls per run
USER_AGENT    = "ArXivDigestBot/3.0 (Tribedy-STAR)"
ARXIV_HEADERS = {"User-Agent": USER_AGENT}
SEEN_CACHE    = os.environ.get("SEEN_CACHE_PATH", "/tmp/seen_papers.json")

# Build collaborator name set for fast lookup
COLLABORATORS = CONFIG.get("collaborators", [])
COLLAB_NAMES  = set()
for c in COLLABORATORS:
    # store last-name and full-name variants, lowercase
    name = c["name"].lower()
    COLLAB_NAMES.add(name)
    parts = name.split()
    if parts:
        COLLAB_NAMES.add(parts[-1])          # last name only
        if len(parts) > 1:
            COLLAB_NAMES.add(parts[0][0] + ". " + parts[-1])  # "B. Schenke"

# INSPIRE identifiers for collaborator queries
COLLAB_INSPIRE_IDS = [c["inspire_id"] for c in COLLABORATORS if c.get("inspire_id")]


# =========================
# Utilities
# =========================

def clean(text: str) -> str:
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

def paper_key(p: dict) -> str:
    return p.get("arxiv_id") or hashlib.md5(p["title"].encode()).hexdigest()

def author_is_collaborator(author_str: str) -> bool:
    a = author_str.lower()
    return any(cname in a for cname in COLLAB_NAMES)

def authors_contain_collaborator(authors: list) -> str | None:
    """Return collaborator name if any author matches, else None."""
    for auth in authors:
        if author_is_collaborator(auth):
            # find which collaborator
            for c in COLLABORATORS:
                lname = c["name"].split()[-1].lower()
                if lname in auth.lower():
                    return c["name"]
    return None


# =========================
# Heuristic scoring
# =========================

WEIGHTS = {
    # UPC / photoproduction — Tribedy's heaviest focus
    "upc": 6, "ultra-peripheral": 6, "photoproduction": 6, "photonuclear": 6,
    "coherent j/psi": 5, "incoherent j/psi": 5, "j/psi": 4, "jpsi": 4,
    "upsilon": 3, "breit-wheeler": 5, "linear polarization": 4,
    "polarized photon": 4, "pomeron": 3, "odderon": 3, "nuclear shadowing": 3,
    "saturation": 3, "cgc": 3, "glasma": 3, "ip-glasma": 4, "ip-sat": 3,
    "b-cgc": 3, "bk equation": 2, "jimwlk": 2, "small-x": 2,
    # Baryon transport / junction
    "baryon junction": 6, "gluon junction": 6, "baryon stopping": 5,
    "baryon transport": 5, "net-baryon": 4, "net baryon": 4,
    "baryon number": 3, "charge stopping": 4,
    # CME / topology
    "chiral magnetic": 5, "cme": 4, "isobar": 4,
    "charge separation": 3, "three-particle correlator": 3,
    "reaction plane correlator": 3, "anomalous transport": 3,
    # ESE / flow fluctuations — another core interest
    "event-shape engineering": 7, "ese": 5, "q-vector": 4,
    "flow fluctuation": 6, "flow fluctuations": 6,
    "eccentricity fluctuation": 5, "longitudinal decorrelation": 4,
    "flow-plane decorrelation": 4, "multiplane cumulant": 4,
    "non-linear mode coupling": 5, "linear response coefficient": 4,
    "mode coupling": 3, "anisotropic flow": 4, "elliptic flow": 3,
    "triangular flow": 3, "v2": 2, "v3": 2, "v4": 2,
    "two-particle cumulant": 3, "four-particle cumulant": 3,
    "initial eccentricity": 3, "initial geometry": 3,
    # BES-II / critical point
    "bes-ii": 5, "beam energy scan": 4, "net-proton": 4, "net-kaon": 3,
    "net-charge fluctuation": 3, "higher-order cumulant": 3,
    "qcd critical point": 4, "fxt": 3,
    # Nuclear structure
    "nuclear deformation": 4, "nuclear shape": 3, "neutron skin": 3,
    "isobar collision": 4, "ru+ru": 4, "zr+zr": 4, "u+u": 3,
    # Directed flow / EM fields
    "directed flow": 3, "v1": 2, "electromagnetic field": 3,
    "magnetic field": 2,
    # Heavy flavor / quarkonium
    "heavy flavor": 2, "open charm": 2, "d meson": 2,
    # Femtoscopy
    "femtoscopy": 2, "hbt": 2, "source size": 1,
    # Hyperon / polarization
    "hyperon polarization": 3, "global polarization": 4, "spin alignment": 3,
    "vorticity": 2, "lambda polarization": 3,
    # Small systems / collectivity
    "small system": 2, "proton shape": 3, "proton fluctuation": 3,
    "ridge": 2, "long-range correlation": 2,
    # EIC
    "electron-ion collider": 3, "eic": 3,
    # STAR / RHIC / sPHENIX
    "star": 5, "rhic": 5, "sphenix": 4, "phenix": 2,
    "au+au": 3, "au + au": 3, "pb+pb": 2, "d+au": 2,
    "heavy ion": 2, "heavy-ion": 2, "quark-gluon plasma": 2, "qgp": 2,
}

def heuristic_score(title: str, abstract: str):
    text = clean(f"{title} {abstract}").lower()
    raw, hits = 0, []
    for kw, w in WEIGHTS.items():
        if kw in text:
            raw += w
            hits.append(kw)
    score = max(1, min(10, 1 + raw // 3))

    if any(x in text for x in ["upc", "ultra-peripheral", "photoproduction", "photonuclear", "j/psi", "jpsi", "breit-wheeler"]):
        star = "UPC / J/psi photoproduction — core Tribedy/STAR program."
    elif any(x in text for x in ["baryon junction", "gluon junction", "baryon stopping", "baryon transport"]):
        star = "Baryon junction / stopping — direct overlap with Tribedy group papers."
    elif any(x in text for x in ["event-shape engineering", "ese", "flow fluctuation", "flow fluctuations", "eccentricity fluctuation", "longitudinal decorrelation", "multiplane cumulant", "non-linear mode coupling"]):
        star = "ESE / flow fluctuations — core Tribedy/STAR analysis program."
    elif any(x in text for x in ["chiral magnetic", "cme", "isobar"]):
        star = "CME / isobar — Tribedy led STAR blind analysis."
    elif any(x in text for x in ["star", "rhic", "sphenix"]):
        star = "Direct STAR/RHIC/sPHENIX relevance."
    elif any(x in text for x in ["bes-ii", "beam energy scan", "net-proton", "qcd critical point"]):
        star = "BES-II / critical point — STAR BES program."
    elif any(x in text for x in ["anisotropic flow", "elliptic flow", "cumulant"]):
        star = "Flow observables — testable with STAR."
    elif any(x in text for x in ["eic", "electron-ion collider"]):
        star = "EIC physics — Tribedy is active in EIC program."
    else:
        star = "N/A"

    return score, hits


# =========================
# Full-text fetching
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
        return clean(main.get_text(separator=" ", strip=True))[:18000]
    except Exception:
        return ""

def extract_figure_captions(arxiv_url: str) -> list:
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
                text = clean(cap.get_text(" ", strip=True))
                if len(text) > 20:
                    captions.append(("Figure", text[:350]))
        for cap_tag in soup.find_all("caption")[:8]:
            text = clean(cap_tag.get_text(" ", strip=True))
            if len(text) > 20:
                captions.append(("Table", text[:350]))
    except Exception:
        pass
    return captions[:12]


# =========================
# Paper fetching — arXiv
# =========================

def fetch_arxiv(session: requests.Session, cat: str) -> list:
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
            "title": clean(title_tag.get_text(" ", strip=True).replace("Title:", "")),
            "abstract": clean(abstract_tag.get_text(" ", strip=True) if abstract_tag else ""),
            "authors": clean(authors_tag.get_text(" ", strip=True).replace("Authors:", "") if authors_tag else "")[:250],
            "cat": cat, "cite_count": 0, "source": "arXiv",
        })
    return papers


# =========================
# Paper fetching — INSPIRE-HEP (topic search)
# =========================

def fetch_inspire_topics(topics: list, days: int = 1) -> list:
    papers = []
    q_parts = " OR ".join(f'"{t}"' for t in topics)
    q = f"({q_parts}) AND date>{days}d"
    params = {
        "sort": "mostrecent", "size": 30, "page": 1,
        "fields": "arxiv_eprints,titles,abstracts,authors,citation_count",
        "q": q,
    }
    try:
        r = requests.get("https://inspirehep.net/api/literature",
                         params=params, headers=ARXIV_HEADERS, timeout=30)
        r.raise_for_status()
        for hit in r.json().get("hits", {}).get("hits", []):
            p = _inspire_hit_to_paper(hit, "inspire-topic")
            if p:
                papers.append(p)
    except Exception as e:
        print(f"INSPIRE topic search failed: {e}")
    return papers


# =========================
# Paper fetching — INSPIRE-HEP (collaborator author search)
# =========================

def fetch_inspire_collaborators(days: int = 2) -> list:
    """Fetch recent papers from any of Tribedy's close collaborators."""
    papers = []
    for inspire_id in COLLAB_INSPIRE_IDS[:10]:   # cap to avoid hammering API
        q = f"a {inspire_id} AND date>{days}d"
        params = {
            "sort": "mostrecent", "size": 8, "page": 1,
            "fields": "arxiv_eprints,titles,abstracts,authors,citation_count",
            "q": q,
        }
        try:
            r = requests.get("https://inspirehep.net/api/literature",
                             params=params, headers=ARXIV_HEADERS, timeout=25)
            r.raise_for_status()
            for hit in r.json().get("hits", {}).get("hits", []):
                p = _inspire_hit_to_paper(hit, f"collab:{inspire_id}")
                if p:
                    p["is_collaborator_paper"] = True
                    papers.append(p)
            time.sleep(0.4)
        except Exception as e:
            print(f"INSPIRE collab fetch {inspire_id} failed: {e}")
    return papers


def _inspire_hit_to_paper(hit: dict, source_tag: str) -> dict | None:
    m = hit.get("metadata", {})
    eprints = m.get("arxiv_eprints", [])
    if not eprints:
        return None
    arxiv_id = eprints[0].get("value", "")
    if not arxiv_id:
        return None
    titles = m.get("titles", [])
    title = titles[0].get("title", "") if titles else ""
    abstracts = m.get("abstracts", [])
    abstract = abstracts[0].get("value", "") if abstracts else ""
    authors_list = m.get("authors", [])
    authors_str = ", ".join(a.get("full_name", "") for a in authors_list[:6])
    if len(authors_list) > 6:
        authors_str += " et al."
    return {
        "id": f"https://arxiv.org/abs/{arxiv_id}",
        "arxiv_id": arxiv_id,
        "title": clean(title),
        "abstract": clean(abstract),
        "authors": clean(authors_str)[:250],
        "cat": source_tag,
        "cite_count": m.get("citation_count", 0),
        "source": "INSPIRE-HEP",
        "is_collaborator_paper": False,
    }


# =========================
# Paper fetching — CERN CDS
# =========================

def fetch_cds(topics: list) -> list:
    query = " OR ".join(f'"{t}"' for t in topics[:6])
    params = {
        "p": query, "action_search": "Search",
        "sf": "year", "so": "d", "rg": 15, "of": "hx", "ln": "en",
    }
    papers = []
    try:
        r = requests.get("https://cds.cern.ch/search",
                         params=params, headers=ARXIV_HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "xml")
        for record in soup.find_all("record")[:12]:
            arxiv_id = None
            for f037 in record.find_all("datafield", tag="037"):
                src = f037.find("subfield", code="9")
                val = f037.find("subfield", code="a")
                if src and "arxiv" in src.get_text().lower() and val:
                    arxiv_id = val.get_text().replace("arXiv:", "").strip()
                    break
            t245 = record.find("datafield", tag="245")
            title = t245.find("subfield", code="a").get_text() if t245 else ""
            t520 = record.find("datafield", tag="520")
            abstract = t520.find("subfield", code="a").get_text() if t520 else ""
            if not title:
                continue
            papers.append({
                "id": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
                "arxiv_id": arxiv_id or "",
                "title": clean(title),
                "abstract": clean(abstract),
                "authors": "",
                "cat": "cds",
                "cite_count": 0,
                "source": "CERN CDS",
                "is_collaborator_paper": False,
            })
    except Exception as e:
        print(f"CDS fetch failed: {e}")
    return papers


# =========================
# Gemini LLM scoring
# =========================

def call_gemini(prompt: str, system_instr: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")
    model = "gemini-2.0-flash-lite"
    ep = (f"https://generativelanguage.googleapis.com/v1beta/models/"
          f"{model}:generateContent?key={api_key}")
    payload = {
        "systemInstruction": {"parts": [{"text": system_instr}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    r = requests.post(ep, json=payload, timeout=55)
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


def parse_llm_json(raw: str) -> dict:
    s, e = raw.find("{"), raw.rfind("}")
    if s != -1 and e != -1:
        raw = raw[s: e + 1]
    try:
        return json.loads(raw)
    except Exception:
        return {}


SYSTEM_INSTR = (
    "You are Prithwish Tribedy, an Associate Physicist at BNL and STAR collaborator. "
    "Your primary research is: UPC / J/psi photoproduction, baryon junctions and stopping, "
    "the Chiral Magnetic Effect and isobar blind analysis, event-shape engineering (ESE), "
    "flow fluctuations (v_n cumulants, eccentricity, non-linear mode coupling), "
    "IP-Glasma/CGC initial-state physics, BES-II critical point, "
    "and EIC detector/physics. "
    "Score 1-10 for relevance to YOUR current analyses. "
    "is_genuinely_new=true only for first-time results (not proceedings of known results). "
    "is_collaborator_paper=true if any author is one of your close collaborators: "
    + ", ".join(c["name"] for c in COLLABORATORS) + "."
)

def score_paper_llm(p: dict, captions: list) -> dict:
    captions_text = "\n".join(f"[{k} caption] {c}" for k, c in captions) or "(not available)"
    full_text = get_html_text(p["id"]) if "/abs/" in p["id"] else ""
    prompt = (
        'Return ONLY valid JSON with exactly these keys:\n'
        '{\n'
        '  "score": <1-10 integer>,\n'
        '  "summary": "<2-3 sentence summary of main finding>",\n'
        '  "star_angle": "<specific STAR/RHIC/EIC observable this connects to, or N/A>",\n'
        '  "key_results": "<2-3 most important quantitative or qualitative results>",\n'
        '  "what_you_learn": "<what you, as Tribedy, take away from this paper>",\n'
        '  "followup_ideas": "<2-3 concrete STAR measurements or directions this motivates>",\n'
        '  "cite_worthy_plots": "<figure/table numbers + 1-sentence description of the 1-2 most cite-worthy items>",\n'
        '  "is_genuinely_new": <true|false>,\n'
        '  "is_collaborator_paper": <true|false>\n'
        '}\n\n'
        f"Title: {p['title']}\n"
        f"Authors: {p['authors']}\n"
        f"Abstract: {p['abstract']}\n"
        f"Figure/Table captions:\n{captions_text}\n"
        f"Full paper text:\n{full_text or '(HTML not available)'}"
    )
    raw = call_gemini(prompt, SYSTEM_INSTR)
    obj = parse_llm_json(raw)
    score = max(1, min(10, int(obj.get("score", 1))))
    return {
        "score": score,
        "summary": clean(obj.get("summary", "")),
        "star_angle": clean(obj.get("star_angle", "N/A")),
        "key_results": clean(obj.get("key_results", "")),
        "what_you_learn": clean(obj.get("what_you_learn", "")),
        "followup_ideas": clean(obj.get("followup_ideas", "")),
        "cite_worthy_plots": clean(obj.get("cite_worthy_plots", "")),
        "is_genuinely_new": bool(obj.get("is_genuinely_new", False)),
        "is_collaborator_paper": bool(obj.get("is_collaborator_paper", False)),
        "gemini_scored": True,
    }


# =========================
# Notification gate
# =========================

def passes_gate(hits: list) -> bool:
    """Only email if at least one paper is worth it."""
    for h in hits:
        if h.get("is_genuinely_new"):
            return True
        if h.get("is_collaborator_paper"):
            return True
        if h.get("cite_count", 0) >= 8:
            return True
        if h.get("score", 0) >= 8:
            return True
    return False


# =========================
# Email builder
# =========================

def build_html(hits: list, total: int, sources: list) -> str:
    body = [
        f"<h2>arXiv Digest — Tribedy/STAR ({len(hits)} papers)</h2>",
        f"<p>Scanned {total} unique papers from "
        f"{html.escape(', '.join(CATEGORIES))} + "
        f"{html.escape(', '.join(sources))}.</p><hr>",
    ]
    if not hits:
        body.append(f"<p>No papers scored ≥ {THRESHOLD}/10 today.</p>")
        return "".join(body)

    for h in hits:
        scored_by = "Gemini" if h.get("gemini_scored") else "Heuristic"
        badges = []
        if h.get("is_genuinely_new"):
            badges.append("🆕 New result")
        if h.get("is_collaborator_paper"):
            collab = authors_contain_collaborator(h["authors"].split(", "))
            badges.append(f"👥 Collaborator: {collab or 'known'}")
        if h.get("cite_count", 0) >= 5:
            badges.append(f"📈 {h['cite_count']} citations")
        badge_str = " &nbsp;|&nbsp; ".join(badges)
        body.append(
            f"<p>"
            f"<b>[{h['score']}/10] [{scored_by}]</b>"
            + (f"  <span style='color:#c00'>{badge_str}</span>" if badge_str else "")
            + f"<br>"
            f"<a href='{html.escape(h['id'])}'><b>{html.escape(h['title'])}</b></a><br>"
            f"<i>{html.escape(h['authors'][:180])} | {html.escape(h['cat'])}</i><br><br>"
        )
        if h.get("summary"):
            body.append(f"<b>Summary:</b> {html.escape(h['summary'])}<br>")
        if h.get("key_results"):
            body.append(f"<b>Key results:</b> {html.escape(h['key_results'])}<br>")
        if h.get("what_you_learn"):
            body.append(f"<b>What you learn:</b> {html.escape(h['what_you_learn'])}<br>")
        if h.get("cite_worthy_plots"):
            body.append(f"<b>Cite-worthy plots/tables:</b> {html.escape(h['cite_worthy_plots'])}<br>")
        if h.get("star_angle") and h["star_angle"] != "N/A":
            body.append(f"<b>STAR angle:</b> {html.escape(h['star_angle'])}<br>")
        if h.get("followup_ideas"):
            body.append(f"<b>Follow-up ideas:</b> {html.escape(h['followup_ideas'])}<br>")
        body.append("</p><hr>")
    return "".join(body)


# =========================
# Main
# =========================

def main():
    for k in ["GEMINI_API_KEY", "SENDGRID_API_KEY", "FROM_EMAIL", "TO_EMAIL"]:
        if not os.environ.get(k):
            raise RuntimeError(f"Missing env var: {k}")

    # Topic keywords for INSPIRE + CDS searches
    topics = [
        "UPC", "ultra-peripheral", "J/psi photoproduction", "photonuclear",
        "coherent J/psi", "Breit-Wheeler", "baryon junction", "gluon junction",
        "baryon stopping", "baryon transport",
        "event-shape engineering", "flow fluctuations",
        "eccentricity fluctuation", "non-linear mode coupling",
        "longitudinal decorrelation", "multiplane cumulant",
        "chiral magnetic effect", "isobar", "IP-Glasma", "CGC saturation",
        "STAR RHIC", "BES-II", "net-proton cumulant",
    ]

    session = requests.Session()
    all_papers = []
    sources_used = []

    # 1. arXiv new listings
    for cat in CATEGORIES:
        try:
            papers = fetch_arxiv(session, cat)
            print(f"arXiv {cat}: {len(papers)} papers")
            all_papers.extend(papers)
        except Exception as e:
            print(f"arXiv {cat} failed: {e}")
    sources_used.append("arXiv")

    # 2. INSPIRE topic search
    inspire_topic = fetch_inspire_topics(topics, days=2)
    print(f"INSPIRE topic: {len(inspire_topic)} papers")
    all_papers.extend(inspire_topic)
    if inspire_topic:
        sources_used.append("INSPIRE-HEP topics")

    # 3. INSPIRE collaborator search (ALWAYS run — catch any collaborator paper)
    inspire_collab = fetch_inspire_collaborators(days=2)
    print(f"INSPIRE collaborators: {len(inspire_collab)} papers")
    all_papers.extend(inspire_collab)
    if inspire_collab:
        sources_used.append("INSPIRE-HEP collaborators")

    # 4. CERN CDS
    cds_papers = fetch_cds(topics[:8])
    print(f"CERN CDS: {len(cds_papers)} papers")
    all_papers.extend(cds_papers)
    if cds_papers:
        sources_used.append("CERN CDS")

    # Deduplicate
    seen_keys: dict = {}
    for p in all_papers:
        k = paper_key(p)
        if k not in seen_keys:
            seen_keys[k] = p
        else:
            # merge: prefer entry with more data
            existing = seen_keys[k]
            if not existing.get("abstract") and p.get("abstract"):
                seen_keys[k] = {**existing, **p}
            # Preserve is_collaborator_paper flag from either
            seen_keys[k]["is_collaborator_paper"] = (
                existing.get("is_collaborator_paper") or p.get("is_collaborator_paper", False)
            )
    unique = list(seen_keys.values())
    print(f"Unique papers: {len(unique)}")

    # Filter already-seen (only if NOT a collaborator paper)
    seen_cache = load_seen()
    new_papers = []
    for p in unique:
        k = paper_key(p)
        if k in seen_cache and not p.get("is_collaborator_paper"):
            continue
        new_papers.append(p)
    print(f"Papers after seen-cache filter: {len(new_papers)}")

    # Detect collaborator papers by author string (for arXiv papers not caught by INSPIRE)
    for p in new_papers:
        if not p.get("is_collaborator_paper"):
            collab = authors_contain_collaborator(p["authors"].split(", "))
            if collab:
                p["is_collaborator_paper"] = True

    # Heuristic pre-filter
    candidates = []
    for p in new_papers:
        score, hits = heuristic_score(p["title"], p["abstract"])
        p["h_score"] = score
        p["h_hits"] = hits
        # Always include collaborator papers; others need to pass heuristic bar
        if score >= THRESHOLD - 1 or p.get("is_collaborator_paper"):
            candidates.append(p)

    # Sort: collaborator papers first, then by heuristic score
    candidates.sort(key=lambda x: (
        0 if x.get("is_collaborator_paper") else 1,
        -x["h_score"]
    ))
    print(f"Candidates for deep scoring: {len(candidates)}")

    # Gemini deep scoring
    hits = []
    failures = 0
    deep_count = 0

    for p in candidates:
        is_collab = p.get("is_collaborator_paper", False)

        if deep_count >= DEEP_LIMIT and not is_collab:
            # Heuristic fallback for non-collaborator papers beyond limit
            if p["h_score"] >= THRESHOLD:
                hits.append({**p, "score": p["h_score"],
                              "summary": f"Heuristic match: {', '.join(p['h_hits'][:5])}",
                              "star_angle": "N/A", "key_results": "",
                              "what_you_learn": "", "followup_ideas": "",
                              "cite_worthy_plots": "", "is_genuinely_new": False,
                              "gemini_scored": False})
            continue

        captions = extract_figure_captions(p["id"]) if "/abs/" in p["id"] else []
        try:
            result = score_paper_llm(p, captions)
            deep_count += 1
            score = result["score"]
            print(f"  Gemini={score}/10 new={result['is_genuinely_new']} "
                  f"collab={result['is_collaborator_paper']} | {p['title'][:70]}")
        except Exception as e:
            failures += 1
            print(f"  Gemini failed ({e}) — heuristic fallback")
            result = {
                "score": p["h_score"],
                "summary": f"Heuristic: {', '.join(p['h_hits'][:5])}",
                "star_angle": "N/A", "key_results": "", "what_you_learn": "",
                "followup_ideas": "", "cite_worthy_plots": "",
                "is_genuinely_new": False,
                "is_collaborator_paper": is_collab,
                "gemini_scored": False,
            }
            score = p["h_score"]

        # Include if: passes threshold, OR is a collaborator paper (always include)
        if score >= THRESHOLD or is_collab or result.get("is_collaborator_paper"):
            hits.append({**p, **result})

        time.sleep(1.2)

    hits.sort(key=lambda x: (
        0 if (x.get("is_genuinely_new") or x.get("is_collaborator_paper")) else 1,
        -x["score"]
    ))
    print(f"Relevant papers: {len(hits)} | Gemini failures: {failures}")

    if not hits:
        print("Nothing found — skipping email.")
        return

    if not passes_gate(hits):
        print("Papers found but none pass novelty/collaborator/influence gate — skipping email.")
        return

    # Build and send
    html_body = build_html(hits, len(unique), sources_used)
    sg = sendgrid.SendGridAPIClient(api_key=os.environ["SENDGRID_API_KEY"])
    mail = Mail(
        from_email=os.environ["FROM_EMAIL"],
        to_emails=os.environ["TO_EMAIL"],
        subject=f"arXiv Digest: {len(hits)} papers — Tribedy/STAR",
        html_content=html_body,
    )
    sg.send(mail)
    print("Email sent.")

    # Update seen-papers cache
    new_seen = seen_cache | {paper_key(p) for p in hits}
    save_seen(new_seen)


if __name__ == "__main__":
    main()
