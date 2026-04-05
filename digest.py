import os, re, json, time, datetime, tqdm, yaml, urllib.request
from bs4 import BeautifulSoup as bs
from google import genai
from google.genai import types
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

# ── Config ───────────────────────────────────────────────────────────────────

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

with open("config.yaml") as f:
    config = yaml.safe_load(f)

SENDGRID_API_KEY = os.environ["SENDGRID_API_KEY"]
FROM_EMAIL       = os.environ["FROM_EMAIL"]
TO_EMAIL         = os.environ["TO_EMAIL"]
MODEL            = "gemini-2.0-flash"

# ── Fetch Papers ─────────────────────────────────────────────────────────────

def fetch_papers(subject):
    url  = f"https://arxiv.org/list/{subject}/new"
    page = urllib.request.urlopen(url)
    soup = bs(page, "html.parser")
    dl   = soup.body.find("div", {"id": "content"}).dl
    papers = []
    for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd")):
        num = dt.text.strip().split(" ")[2].split(":")[-1]
        papers.append({
            "main_page": f"https://arxiv.org/abs/{num}",
            "title":     dd.find("div", {"class": "list-title mathjax"}).text.replace("Title:", "").strip(),
            "authors":   dd.find("div", {"class": "list-authors"}).text.replace("Authors:", "").replace("\n", "").strip(),
            "abstract":  dd.find("p",   {"class": "mathjax"}).text.replace("\n", " ").strip(),
        })
    return papers

def get_all_papers(subjects):
    seen, all_papers = set(), []
    for subj in subjects:
        try:
            for p in fetch_papers(subj.strip()):
                if p["main_page"] not in seen:
                    seen.add(p["main_page"])
                    all_papers.append(p)
        except Exception as e:
            print(f"Failed to fetch {subj}: {e}")
    print(f"Total papers: {len(all_papers)}")
    return all_papers

# ── Relevancy Scoring ────────────────────────────────────────────────────────

PROMPT_HEADER = (
    "You will be given a list of research papers and a description of topics of interest.\n"
    "For each paper output exactly one JSON object on its own line with this format:\n"
    'N. {"Relevancy score": <1-10>, "Reason": "<brief reason>"}\n\n'
)

def build_prompt(interest, papers):
    prompt = PROMPT_HEADER + f"Topics of interest:\n{interest}\n\nPapers:\n"
    for i, p in enumerate(papers, 1):
        prompt += f"###\n{i}. Title: {p['title']}\n{i}. Authors: {p['authors']}\n{i}. Abstract: {p['abstract']}\n"
    prompt += "\nGenerate response:\n1."
    return prompt

def parse_response(papers, text, threshold):
    results = []
    scored_lines = [l for l in text.split("\n") if "relevancy score" in l.lower()]
    for i, line in enumerate(scored_lines):
        if i >= len(papers):
            break
        try:
            item  = json.loads(re.sub(r"^\d+\.\s*", "", line))
            raw   = item.get("Relevancy score", 0)
            score = int(str(raw).split("/")[0])
            if score >= threshold:
                results.append({**papers[i], **item, "Relevancy score": score})
        except Exception as e:
            print(f"Parse error line {i}: {e} — {line}")
    return sorted(results, key=lambda x: x["Relevancy score"], reverse=True)

def score_papers(papers, interest, threshold=8, batch=4):
    results = []
    for i in tqdm.tqdm(range(0, len(papers), batch)):
        chunk = papers[i:i + batch]
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=build_prompt(interest, chunk),
                config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=128 * batch),
            )
            results.extend(parse_response(chunk, resp.text, threshold))
        except Exception as e:
            print(f"Gemini error batch {i}: {e}")
        time.sleep(1)
    return results

# ── Email ────────────────────────────────────────────────────────────────────

def build_html(papers):
    if not papers:
        return "<p>No highly relevant papers found this week.</p>"
    rows = [f"<h2>ArXiv Digest — {datetime.date.today().strftime('%b %d, %Y')}</h2><ul>"]
    for p in papers:
        rows.append(
            f"<li style='margin-bottom:16px'>"
            f"<b><a href='{p['main_page']}'>{p['title']}</a></b><br>"
            f"<i>{p['authors']}</i><br>"
            f"Score: {p['Relevancy score']}/10 — {p.get('Reason', '')}"
            f"</li>"
        )
    rows.append("</ul>")
    return "\n".join(rows)

def send_email(html):
    msg = Mail(
        from_email=FROM_EMAIL,
        to_emails=TO_EMAIL,
        subject=f"ArXiv Digest — {datetime.date.today().strftime('%b %d, %Y')}",
        html_content=html,
    )
    resp = SendGridAPIClient(SENDGRID_API_KEY).send(msg)
    print(f"Email sent — status {resp.status_code}")

# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    papers   = get_all_papers(config["subjects"])
    relevant = score_papers(papers, config["interest"], threshold=config.get("threshold", 8))
    print(f"{len(relevant)} relevant papers found")
    send_email(build_html(relevant))
