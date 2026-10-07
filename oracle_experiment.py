"""
MOOSE-Chem Oracle Experiment (fixed version)
============================================

Pipeline:  RETRIEVE inspirations -> COMPOSE hypotheses -> RANK (pick the best one)

Conditions (only ONE stage is made "perfect" at a time):
    base                : normal retrieval -> normal composition -> LLM ranker
    oracle_retrieval    : gold inspirations -> normal composition -> LLM ranker
    oracle_composition  : normal retrieval -> stronger composition -> LLM ranker
    oracle_ranking      : normal retrieval -> normal composition -> oracle ranker
    all_oracles         : gold inspirations -> stronger composition -> oracle ranker

Candidate pool: every inspiration-paper title in the benchmark file (same pool for
every target paper). The gold inspirations of a target are marked by id.

How to use:
    1. Set JUDGE_CHECK = True and run once. Read the 'self / other / no inspiration' lines.
    2. Set JUDGE_CHECK = False, LIMIT = 3, run. Then raise LIMIT (10, 15, None).

Requirements:  pip install openpyxl; Ollama running; GEMINI_API_KEY or DEEPSEEK_API_KEY set (used only for scoring).
"""

import json
import os
import random
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

from openpyxl import load_workbook

# ================================================================
# CONFIGURATION
# ================================================================
MODEL = "qwen7-8k"      # the Ollama model you created from a 7B/8B model (see guide)
LIMIT = 3              # number of papers; None = all 51
JUDGE_CHECK = True   # True = only test the judge, then stop
N_BASE = 3             # drafts from the normal composer
N_STRONG = 6           # drafts from the stronger composer (each refined once)
WINDOW, PICK = 15, 3   # retrieval: 15 candidates at a time, keep the best 3
SEED = 0
random.seed(SEED)

DATA_FILE = (Path.home() / "Desktop" / "testing" / "MOOSE-Chem"
             / "Data" / "chem_research_2024.xlsx")
RESULTS_FILE = Path("oracle_results.json")

# The judge (scoring) uses an online model; everything else uses your local model.
# Choose "gemini" (free key from Google AI Studio, env var GEMINI_API_KEY)
# or "deepseek" (needs account balance, env var DEEPSEEK_API_KEY).
JUDGE_PROVIDER = "ollama"   # "ollama" (local, free), "gemini", or "deepseek"
# Check the provider's current model list; names change.
JUDGE_MODELS = {"ollama": "qwen7-8k", "gemini": "gemini-2.5-flash", "deepseek": "deepseek-chat"}
JUDGE_MODEL = JUDGE_MODELS[JUDGE_PROVIDER]


# ================================================================
# OLLAMA
# ================================================================
def ask(prompt, max_tokens=1500, model=None):
    """Send one request to the local Ollama server (http://localhost:11434)."""
    body = json.dumps({
        "model": model or MODEL,
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens},
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    request = urllib.request.Request(
        "http://localhost:11434/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.loads(response.read())
    except urllib.error.URLError as e:
        raise RuntimeError(
            "\nCould not connect to Ollama. Make sure it is running and test:\n"
            f'    ollama run {MODEL} "say hello"\nOriginal error: {e}\n')
    if "message" not in result:
        raise RuntimeError(f"Ollama returned an unexpected response:\n{result}")
    text = result["message"]["content"]
    # remove <think>...</think> blocks if a "thinking" model is used
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


# ================================================================
# UTILITY
# ================================================================
def ints(text):
    return [int(x) for x in re.findall(r"\d+", str(text))]


def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.upper() in {"", "NA", "N/A", "NONE", "NULL"} else text


# ================================================================
# DATA LOADING
# ================================================================
def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found:\n{DATA_FILE}")
    wb = load_workbook(DATA_FILE, read_only=True, data_only=True)
    rows = list(wb["Overall"].iter_rows(values_only=True))
    headers = [clean_text(x) for x in rows[0]]
    col = {n: i for i, n in enumerate(headers) if n}

    def cell(row, name):
        i = col.get(name)
        return clean_text(row[i]) if i is not None and i < len(row) else ""

    insp_cols = ["Inspiration paper 1 title",
                 "Inspiration paper 2 title",
                 "Inspiration paper 3 title"]

    # Pool = all distinct inspiration titles in the file
    pool, id_of = [], {}
    for row in rows[1:]:
        for c in insp_cols:
            t = cell(row, c)
            if t and t.lower() not in id_of:
                id_of[t.lower()] = f"P{len(pool)}"
                pool.append({"id": id_of[t.lower()], "title": t,
                             "text": f"Title: {t}"})

    targets = []
    for row in rows[1:]:
        title = cell(row, "Title")
        bg = cell(row, "Background Question")
        hyp = cell(row, "Main hypothesis")
        gold_ids = []
        for c in insp_cols:
            t = cell(row, c)
            if t and id_of[t.lower()] not in gold_ids:
                gold_ids.append(id_of[t.lower()])
        if title and bg and hyp and gold_ids:
            targets.append({"id": cell(row, "No"), "title": title,
                            "background": bg, "gold_hypothesis": hyp,
                            "pool": pool, "gold_ids": gold_ids})
    print(f"Targets: {len(targets)}   Pool size: {len(pool)}")
    return targets


# ================================================================
# RETRIEVAL
# ================================================================
def retrieve(item):
    """Normal retrieval: shuffle, then repeatedly let the model pick the best
    PICK candidates from windows of WINDOW, until PICK remain."""
    keep = list(item["pool"])
    random.shuffle(keep)
    while len(keep) > PICK:
        nxt = []
        for i in range(0, len(keep), WINDOW):
            win = keep[i:i + WINDOW]
            listing = "\n".join(f"{j + 1}. {c['text'][:1000]}"
                                for j, c in enumerate(win))
            out = ask(
                f"""
Research question:
{item['background']}

Candidate papers:

{listing}

Which {min(PICK, len(win))} candidates would best inspire
a NEW, non-obvious hypothesis for this research question?

Do not choose a candidate merely because it is lexically similar
to the question.

Prefer candidates whose scientific concept, mechanism, method,
material, reaction, or phenomenon could plausibly be transferred
to the research problem.

Reply with the numbers only, comma-separated.
""", 50)
            idx = [n - 1 for n in ints(out) if 1 <= n <= len(win)][:PICK]
            nxt += [win[j] for j in idx] or win[:1]
        if len(nxt) >= len(keep):
            break
        keep = nxt
    return keep[:PICK]


def gold_retrieve(item):
    """Oracle retrieval: the true inspirations (taken from the same pool)."""
    gold_ids = set(item["gold_ids"])
    return [c for c in item["pool"] if c["id"] in gold_ids]


# ================================================================
# COMPOSITION
# ================================================================
def compose(item, insps, n, refine=False):
    if not insps:
        return []
    text = "\n".join(f"- {c['text'][:1200]}" for c in insps)
    out = ask(
        f"""
Research question:
{item['background']}

Inspirations:
{text}

Write {n} different, specific, mechanistic and testable hypotheses.

Each hypothesis should:
- directly address the research question;
- use a meaningful concept from the inspirations;
- propose a plausible mechanism;
- make a concrete prediction;
- be experimentally testable.

Avoid simply restating the research question.

Separate hypotheses with a line containing only:
###

Do not include numbering before the hypotheses.
""", 3000)

    hyps = [h.strip() for h in out.split("###") if h.strip()][:n]
    if len(hyps) < n and n > 1:  # fallback if the model ignored the ### separator
        parts = [p.strip() for p in re.split(r"\n\s*\n", out) if len(p.strip()) > 40]
        if len(parts) > len(hyps):
            hyps = parts[:n]

    if refine:
        refined = []
        for h in hyps:
            improved = ask(
                f"""
Research question:
{item['background']}

Draft hypothesis:
{h}

Improve this hypothesis.

Make it:
- more specific;
- more mechanistic;
- more scientifically plausible;
- more novel;
- more directly connected to the research question;
- experimentally testable.

Reply with the improved hypothesis only.
""", 500).strip()
            refined.append(improved if improved else h)
        hyps = refined
    return hyps


# ================================================================
# RANKING (normal LLM ranker)
# ================================================================
def rank(item, hyps):
    if not hyps:
        return ""
    if len(hyps) == 1:
        return hyps[0]
    listing = "\n\n".join(f"{i + 1}. {h}" for i, h in enumerate(hyps))
    out = ask(
        f"""
Research question:
{item['background']}

Candidate hypotheses:

{listing}

Choose the SINGLE hypothesis that is best overall.

Consider:
1. novelty;
2. scientific validity;
3. mechanistic quality;
4. usefulness;
5. testability;
6. how directly it addresses the research question.

Reply with its number only.
""", 20)
    n = ints(out)
    if n and 1 <= n[0] <= len(hyps):
        return hyps[n[0] - 1]
    return hyps[0]


# ================================================================
# JUDGE
# ================================================================
_cache = {}


def _post(url, payload, headers):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError("Judge API error: " + e.read().decode())


def judge_ask(prompt):
    if JUDGE_PROVIDER == "ollama":
        return ask(prompt, 10, model=JUDGE_MODEL)
    if JUDGE_PROVIDER == "gemini":
        time.sleep(13)   # free tier allows about 5 requests per minute
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        res = _post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent",
            {"contents": [{"parts": [{"text": prompt}]}],
             "generationConfig": {"temperature": 0, "maxOutputTokens": 1000}},
            {"x-goog-api-key": key})
        try:
            return res["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return ""   # blocked or empty answer -> counted as 0
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set.")
    res = _post("https://api.deepseek.com/chat/completions",
                {"model": JUDGE_MODEL, "temperature": 0, "max_tokens": 10, "stream": False,
                 "messages": [{"role": "user", "content": prompt}]},
                {"Authorization": "Bearer " + key})
    return res["choices"][0]["message"]["content"]


def score(item, hyp):
    """Strict 0-5 judge (DeepSeek) vs the true hypothesis."""
    if not hyp:
        return 0
    key = (item["id"], hyp)
    if key not in _cache:
        out = judge_ask(f"""
Ground-truth hypothesis:
{item['gold_hypothesis']}

Generated hypothesis:
{hyp}

Compare the SPECIFIC mechanism, material, or method, not just the general topic.

Scale:
0 = unrelated
1 = same general topic only
2 = same topic, but a different mechanism or method
3 = one specific key element of the ground truth is present
4 = most key elements present, but a major element is missing or different
5 = all key elements present (identical hypotheses score 5)

Reply with ONE integer from 0 to 5 only.
""")
        n = ints(out)
        _cache[key] = min(max(n[0], 0), 5) if n else 0
    return _cache[key]


def judge_check(data, n=5):
    """Sanity test of the judge. Expect: self=5, other paper=0-1, no-inspiration low."""
    dummy = [{"id": "x", "title": "none", "text": "(no inspiration given)"}]
    print("\nJUDGE CHECK  (self should be 5, other should be 0-1, "
          "no-inspiration should be low)\n")
    for i in range(min(n, len(data))):
        a, b = data[i], data[-(i + 1)]
        h = (compose(a, dummy, 1) or [""])[0]
        print(f"paper {a['id']:>3} | self: {score(a, a['gold_hypothesis'])} | "
              f"other paper: {score(a, b['gold_hypothesis'])} | "
              f"no inspiration: {score(a, h)}", flush=True)


# ================================================================
# ONE PAPER
# ================================================================
def run_item(item):
    print(f"  Target: {item['title'][:100]}", flush=True)

    print("  [1/7] Normal retrieval...", flush=True)
    base_insp = retrieve(item)

    print("  [2/7] Gold retrieval...", flush=True)
    gold_insp = gold_retrieve(item)
    hit = len({c["id"] for c in base_insp} & set(item["gold_ids"])) / len(item["gold_ids"])
    print(f"  Retrieval hit ratio: {hit:.2f}", flush=True)

    print("  [3/7] Normal composition...", flush=True)
    base_h = compose(item, base_insp, N_BASE, refine=False)

    print("  [4/7] Oracle-retrieval composition...", flush=True)
    gold_h = compose(item, gold_insp, N_BASE, refine=False)

    print("  [5/7] Strong composition...", flush=True)
    strong_h = compose(item, base_insp, N_STRONG, refine=True)

    print("  [6/7] All-oracle composition...", flush=True)
    gold_strong_h = compose(item, gold_insp, N_STRONG, refine=True)

    print("  [7/7] Ranking and judging...", flush=True)
    base_draft_scores = [score(item, h) for h in base_h]
    gold_strong_scores = [score(item, h) for h in gold_strong_h]

    base_score = score(item, rank(item, base_h))
    oracle_retrieval_score = score(item, rank(item, gold_h))
    oracle_composition_score = score(item, rank(item, strong_h))
    oracle_ranking_score = max(base_draft_scores) if base_draft_scores else 0
    all_oracles_score = max(gold_strong_scores) if gold_strong_scores else 0

    result = {
        "id": item["id"],
        "title": item["title"],
        "retrieval_hit_ratio": hit,
        "base": base_score,
        "oracle_retrieval": oracle_retrieval_score,
        "oracle_composition": oracle_composition_score,
        "oracle_ranking": oracle_ranking_score,
        "all_oracles": all_oracles_score,
        "base_draft_scores": base_draft_scores,   # check these: do the drafts differ?
        "num_base_hypotheses": len(base_h),
        "num_gold_hypotheses": len(gold_h),
        "num_strong_hypotheses": len(strong_h),
        "num_gold_strong_hypotheses": len(gold_strong_h),
        "retrieved_titles": [c["title"] for c in base_insp],
        "gold_titles": [c["title"] for c in gold_insp],
    }
    print(f"  Scores: base={base_score}, retrieval={oracle_retrieval_score}, "
          f"composition={oracle_composition_score}, ranking={oracle_ranking_score}, "
          f"all={all_oracles_score}", flush=True)
    return result


# ================================================================
# SUMMARY
# ================================================================
STAGES = ["oracle_retrieval", "oracle_composition", "oracle_ranking"]
KEYS = ["base", *STAGES, "all_oracles"]


def shares(rows):
    m = {k: statistics.mean(r[k] for r in rows) for k in KEYS}
    gain = {k: m[k] - m["base"] for k in STAGES + ["all_oracles"]}
    pos = {k: max(gain[k], 0) for k in STAGES}
    total = sum(pos.values())
    share = {k: (100 * pos[k] / total if total else 0.0) for k in STAGES}
    return m, gain, share


def summarize(rows):
    if not rows:
        print("\nNo results were produced.")
        return
    m, gain, share = shares(rows)

    boots = {k: [] for k in STAGES}
    if len(rows) >= 2:
        for _ in range(1000):
            _, _, s = shares([random.choice(rows) for _ in rows])
            for k in STAGES:
                boots[k].append(s[k])

    print("\n" + "=" * 90)
    print("ORACLE EXPERIMENT SUMMARY")
    print("=" * 90)
    print(f"Papers: {len(rows)}")
    print(f"Mean base score (0-5): {m['base']:.2f}")
    print("Mean retrieval hit ratio: "
          f"{statistics.mean(r['retrieval_hit_ratio'] for r in rows):.2f}\n")
    print(f"{'stage':<20}{'gain (points)':>16}{'share of total %':>20}"
          f"{'% of headroom':>18}{'95% interval':>20}")
    for k in STAGES:
        if len(rows) >= 2:
            v = sorted(boots[k])
            lo = v[max(0, int(0.025 * len(v)))]
            hi = v[min(len(v) - 1, int(0.975 * len(v)) - 1)]
        else:
            lo = hi = share[k]
        head = 100 * gain[k] / (5 - m["base"]) if m["base"] < 5 else 0
        print(f"{k:<20}{gain[k]:>16.2f}{share[k]:>20.1f}{head:>18.1f}"
              f"{f'[{lo:.0f}, {hi:.0f}]':>20}")
    print(f"\nAll three oracles together: gain {gain['all_oracles']:.2f}")
    print(f"Sum of single-stage gains:  {sum(gain[k] for k in STAGES):.2f}\n")
    if len(rows) < 10:
        print("WARNING: small pilot. Do not treat these results as definitive.")
    print("=" * 90)


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MOOSE-CHEM ORACLE EXPERIMENT")
    print("=" * 70)
    print(f"Model: {MODEL}\nDataset: {DATA_FILE}\nLIMIT: {LIMIT}")
    print("=" * 70)

    data = load_data()

    if JUDGE_CHECK:
        judge_check(data)
        raise SystemExit

    if LIMIT is not None:
        data = data[:LIMIT]
    if not data:
        raise RuntimeError("No usable papers were loaded.")
    print(f"Running experiment on {len(data)} paper(s).")

    results = []

    def save():
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    for i, item in enumerate(data):
        print(f"\npaper {i + 1}/{len(data)}\n" + "-" * 70)
        try:
            results.append(run_item(item))
            save()
            print(f"  Saved {len(results)} result(s) to {RESULTS_FILE}")
        except BaseException as e:   # includes Ctrl+C
            print(f"\nStopped on paper {i + 1}: {e!r}")
            save()
            print(f"Saved {len(results)} completed result(s).")
            raise

    summarize(results)