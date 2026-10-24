#!/usr/bin/env python3

"""MOOSE-Chem Oracle Experiment: BM25 retrieval + DeepSeek-R1:7b composition.



Pipeline:

    1. BM25 searches the fixed 3000-paper master pool using each paper's TITLE + ABSTRACT.

    2. Qwen2.5:7b reranks the BM25 top 30 and selects 3 inspirations.

    3. DeepSeek-R1:7b composes/refines hypotheses; Qwen2.5:7b ranks and judges them.

    4. Oracle conditions estimate the effect of gold inspirations, stronger composition,

       and oracle ranking.



Expected input files (matching the original script's Windows setup):

    ~/Desktop/testing/MOOSE-Chem/Data/chem_research_2024.xlsx

    ~/Desktop/testing/MOOSE-Chem/Data/Inspiration_Corpus_3000.json



Expected corpus format:

    [["paper title", "paper abstract"], ...]



The corpus loader also accepts dictionaries containing title/abstract fields.



Install dependencies:

    pip install openpyxl rank-bm25



Download the local model once:

    ollama pull deepseek-r1:7b



Run a small smoke test first (LIMIT = 1 below). Restore desired settings after validating outputs.



Research caveat:

    DeepSeek-R1:7b is used for hypothesis composition/refinement. Qwen2.5:7b is used for

    short structured selection tasks (reranking/ranking) and checklist scoring, because

    DeepSeek-R1 can spend the answer token budget on reasoning before returning a selection.

    This remains an automatic model-based proxy, not a human evaluation; report both models

    explicitly and validate results with blinded review.

"""



from __future__ import annotations



import json

import os

import random

import re

import statistics

import time

import urllib.error

import urllib.request

from pathlib import Path

from typing import Any



from openpyxl import load_workbook

from rank_bm25 import BM25Okapi



# =============================================================================

# CONFIGURATION

# =============================================================================

MODEL = "deepseek-r1:7b"  # hypothesis generation and refinement

SELECTION_MODEL = "qwen2.5:7b"  # short structured reranking and hypothesis ranking

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

NUM_CTX = 8192

# Ask Ollama to suppress thinking where the model/template supports it; some DeepSeek tags may still emit a thinking field.

# Set True for a separate experiment explicitly evaluating DeepSeek-R1's thinking mode.

OLLAMA_THINK = False



# Start with 3 targets to validate the full pipeline. Set to None to run all eligible rows.

LIMIT = None

JUDGE_CHECK = False



# BM25 retrieves 30 candidates; Qwen reranks those 30 and selects 3.

BM25_TOP_K = 30

FINAL_INSPIRATIONS = 3

MAX_ABSTRACT_CHARS_RERANK = 180  # keep 30 candidates manageable within the context window

MAX_TEXT_CHARS_COMPOSE = 1400



N_BASE = 2

N_STRONG = 2

SEED = 0

random.seed(SEED)



PROJECT_DIR = Path.home() / "Desktop" / "testing" / "MOOSE-Chem"

DATA_FILE = PROJECT_DIR / "Data" / "chem_research_2024.xlsx"

CORPUS_FILE = PROJECT_DIR / "Data" / "Inspiration_Corpus_3000.json"

RESULTS_FILE = Path("oracle_results_bm25_top30_deepseek_qwen_selection.json")



# Default scoring is local and free. An online judge is optional.

# Valid: "ollama", "gemini", "deepseek".

JUDGE_PROVIDER = "ollama"

JUDGE_MODELS = {

    # Use a non-reasoning model for short yes/no scoring so the judge does not spend

    # its tiny answer budget on a long DeepSeek-R1 reasoning trace.

    "ollama": "qwen2.5:7b",

    "gemini": "gemini-2.5-flash",

    "deepseek": "deepseek-chat",

}

JUDGE_MODEL = JUDGE_MODELS[JUDGE_PROVIDER]



# =============================================================================

# OLLAMA CLIENT

# =============================================================================

def ask(prompt: str, max_tokens: int = 1500, model: str | None = None) -> str:

    """Send a single chat request to the local Ollama API."""

    body = json.dumps({

        "model": model or MODEL,

        "stream": False,

        "think": OLLAMA_THINK,

        "options": {

            "temperature": 0,

            "num_predict": max_tokens,

            "num_ctx": NUM_CTX,

        },

        "messages": [{"role": "user", "content": prompt}],

    }).encode("utf-8")

    request = urllib.request.Request(

        f"{OLLAMA_URL}/api/chat",

        data=body,

        headers={"Content-Type": "application/json"},

    )

    try:

        with urllib.request.urlopen(request, timeout=900) as response:

            result = json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:

        detail = exc.read().decode("utf-8", errors="replace")

        raise RuntimeError(f"Ollama API error ({exc.code}): {detail}") from exc

    except urllib.error.URLError as exc:

        raise RuntimeError(

            "Could not connect to Ollama. Make sure Ollama is running, then test:\n"

            f'    ollama run {model or MODEL} "say hello"\n'

            f"Ollama URL: {OLLAMA_URL}\nOriginal error: {exc}"

        ) from exc

    except TimeoutError as exc:

        raise RuntimeError(f"Ollama request timed out for model {model or MODEL}.") from exc



    message = result.get("message")

    if not isinstance(message, dict) or "content" not in message:

        raise RuntimeError(f"Ollama returned an unexpected response:\n{result}")

    text = re.sub(

        r"<think>.*?</think>", "", str(message.get("content", "")), flags=re.S

    ).strip()

    if not text:

        # Never turn an empty/truncated model response into an apparently valid zero score

        # or silently accept fallback selections. Keep diagnostics so the failure is visible.

        raise RuntimeError(

            f"Ollama returned empty final content for model={model or MODEL!r}; "

            f"done_reason={result.get('done_reason')!r}, "

            f"eval_count={result.get('eval_count')!r}, "

            f"thinking_chars={len(message.get('thinking') or '')}. "

            "Increase num_predict or use a non-reasoning model for this task."

        )

    return text





def check_ollama() -> None:

    """Fail early with a useful message if Ollama or the selected model is unavailable."""

    req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")

    try:

        with urllib.request.urlopen(req, timeout=10) as response:

            payload = json.loads(response.read().decode("utf-8"))

    except Exception as exc:

        raise RuntimeError(

            f"Ollama is not reachable at {OLLAMA_URL}. Start Ollama and retry. Error: {exc}"

        ) from exc

    available = {str(m.get("name", "")) for m in payload.get("models", [])}

    required_models = {MODEL, SELECTION_MODEL}

    if JUDGE_PROVIDER == "ollama":

        required_models.add(JUDGE_MODEL)

    missing_models = [

        name for name in sorted(required_models)

        if name not in available and not any(installed.startswith(name + ":") for installed in available)

    ]

    if missing_models:

        commands = "\n".join(f"    ollama pull {name}" for name in missing_models)

        raise RuntimeError(

            f"Required model(s) not listed by Ollama: {', '.join(missing_models)}\n"

            f"Run this in PowerShell:\n{commands}\n\n"

            f"Installed models: {', '.join(sorted(available)) or '(none)'}"

        )

    print(

        f"Ollama reachable at {OLLAMA_URL}; generation model: {MODEL}; "

        f"selection model: {SELECTION_MODEL}; "

        f"judge model: {JUDGE_MODEL if JUDGE_PROVIDER == 'ollama' else JUDGE_PROVIDER}",

        flush=True,

    )





# =============================================================================

# TEXT / JSON HELPERS

# =============================================================================

def clean_text(value: Any) -> str:

    if value is None:

        return ""

    text = str(value).strip()

    return "" if text.upper() in {"", "NA", "N/A", "NONE", "NULL", "NAN"} else text





def normalize_title(value: Any) -> str:

    """Normalize punctuation/spacing to improve Excel-to-corpus title matching."""

    text = clean_text(value).casefold()

    return re.sub(r"[^\w]+", " ", text, flags=re.UNICODE).strip()





def tokenize(text: str) -> list[str]:

    """Small deterministic tokenizer for BM25 (no external downloads required)."""

    return re.findall(r"(?u)\b\w+\b", clean_text(text).casefold())





def parse_corpus_entry(entry: Any, index: int) -> tuple[str, str]:

    """Return (title, abstract) from a list/tuple or common dictionary layouts."""

    title = ""

    abstract = ""

    if isinstance(entry, (list, tuple)):

        title = clean_text(entry[0]) if len(entry) > 0 else ""

        abstract = clean_text(entry[1]) if len(entry) > 1 else ""

    elif isinstance(entry, dict):

        for key in ("title", "Title", "paper_title", "Paper Title", "name"):

            if clean_text(entry.get(key)):

                title = clean_text(entry[key])

                break

        for key in ("abstract", "Abstract", "paper_abstract", "Paper Abstract", "summary"):

            if clean_text(entry.get(key)):

                abstract = clean_text(entry[key])

                break

    else:

        raise ValueError(

            f"Corpus entry {index} has unsupported type {type(entry).__name__}; "

            "expected [title, abstract] or a dictionary."

        )

    return title, abstract





# =============================================================================

# LOAD THE FIXED MASTER POOL AND BENCHMARK TARGETS

# =============================================================================

def load_master_pool() -> tuple[list[dict[str, Any]], dict[str, str]]:

    if not CORPUS_FILE.exists():

        raise FileNotFoundError(

            f"3000-paper corpus not found:\n{CORPUS_FILE}\n"

            "Confirm that Inspiration_Corpus_3000.json is in the project's Data folder."

        )

    with CORPUS_FILE.open("r", encoding="utf-8") as handle:

        corpus = json.load(handle)

    if not isinstance(corpus, list):

        raise ValueError("Inspiration_Corpus_3000.json must contain a JSON list.")



    pool: list[dict[str, Any]] = []

    id_of: dict[str, str] = {}

    duplicate_titles = 0

    missing_abstracts = 0



    for i, entry in enumerate(corpus):

        title, abstract = parse_corpus_entry(entry, i)

        if not title:

            continue

        if not abstract:

            missing_abstracts += 1

            # Keep the entry for transparency, but retrieval is title-only for this item.

        norm = normalize_title(title)

        if not norm:

            continue

        if norm in id_of:

            duplicate_titles += 1

            continue



        paper_id = f"P{len(pool):04d}"

        id_of[norm] = paper_id

        paper_text = f"Title: {title}\nAbstract: {abstract}" if abstract else f"Title: {title}"

        pool.append({

            "id": paper_id,

            "title": title,

            "abstract": abstract,

            "text": paper_text,

        })



    if len(pool) < BM25_TOP_K:

        raise ValueError(

            f"The master pool contains only {len(pool)} usable unique titles, "

            f"but BM25_TOP_K={BM25_TOP_K}. Check {CORPUS_FILE}."

        )



    print(f"Master corpus file: {CORPUS_FILE}")

    print(f"Master pool entries: {len(pool)} unique papers")

    print(f"Duplicate titles skipped: {duplicate_titles}")

    if missing_abstracts:

        print(f"WARNING: {missing_abstracts} corpus entry/entries have no abstract.")

    return pool, id_of





def load_data() -> list[dict[str, Any]]:

    pool, id_of = load_master_pool()

    if not DATA_FILE.exists():

        raise FileNotFoundError(f"Benchmark Excel file not found:\n{DATA_FILE}")



    workbook = load_workbook(DATA_FILE, read_only=True, data_only=True)

    try:

        if "Overall" not in workbook.sheetnames:

            raise ValueError(

                f"Worksheet 'Overall' was not found in {DATA_FILE}. "

                f"Available sheets: {workbook.sheetnames}"

            )

        rows = list(workbook["Overall"].iter_rows(values_only=True))

    finally:

        workbook.close()



    if not rows:

        raise ValueError(f"Worksheet 'Overall' is empty: {DATA_FILE}")



    headers = [clean_text(x) for x in rows[0]]

    col = {name: i for i, name in enumerate(headers) if name}

    required = ["Title", "Background Question", "Main hypothesis"]

    missing = [name for name in required if name not in col]

    if missing:

        raise ValueError(f"Missing required Excel column(s): {missing}. Found: {headers}")



    def cell(row: tuple[Any, ...], name: str) -> str:

        i = col.get(name)

        return clean_text(row[i]) if i is not None and i < len(row) else ""



    inspiration_title_columns = [

        "Inspiration paper 1 title",

        "Inspiration paper 2 title",

        "Inspiration paper 3 title",

    ]

    absent_columns = [name for name in inspiration_title_columns if name not in col]

    if absent_columns:

        raise ValueError(

            f"Missing gold-inspiration column(s): {absent_columns}. Found: {headers}"

        )



    targets: list[dict[str, Any]] = []

    skipped_no_match = 0

    total_rows = 0

    incomplete_rows = 0



    for row_num, row in enumerate(rows[1:], start=2):

        title = cell(row, "Title")

        background = cell(row, "Background Question")

        gold_hypothesis = cell(row, "Main hypothesis")

        if not (title and background and gold_hypothesis):

            incomplete_rows += 1

            continue



        all_gold_titles: list[str] = []

        gold_ids: list[str] = []

        missing_gold_titles: list[str] = []

        for column in inspiration_title_columns:

            gold_title = cell(row, column)

            if not gold_title:

                continue

            if normalize_title(gold_title) in {

                normalize_title(existing) for existing in all_gold_titles

            }:

                continue

            all_gold_titles.append(gold_title)

            paper_id = id_of.get(normalize_title(gold_title))

            if paper_id:

                if paper_id not in gold_ids:

                    gold_ids.append(paper_id)

            else:

                missing_gold_titles.append(gold_title)



        if not all_gold_titles:

            incomplete_rows += 1

            continue

        total_rows += 1

        if not gold_ids:

            skipped_no_match += 1

            continue



        row_id = cell(row, "No") or str(row_num - 1)

        targets.append({

            "id": row_id,

            "title": title,

            "background": background,

            "gold_hypothesis": gold_hypothesis,

            "pool": pool,

            "gold_ids": gold_ids,

            "all_gold_titles": all_gold_titles,

            "missing_gold_titles": missing_gold_titles,

            "corpus_gold_coverage": len(gold_ids) / len(all_gold_titles),

            "excel_row": row_num,

        })



    print(f"Benchmark Excel: {DATA_FILE}")

    print(f"Eligible targets with at least one gold inspiration in corpus: {len(targets)}")

    print(f"Rows with gold inspirations but zero corpus matches (excluded): {skipped_no_match}")

    if incomplete_rows:

        print(f"Rows skipped for missing title/background/hypothesis/inspirations: {incomplete_rows}")



    partial = [item for item in targets if item["missing_gold_titles"]]

    if partial:

        print(

            f"WARNING: {len(partial)} target(s) have only partial gold-inspiration coverage "

            "in the 3000-paper corpus. See missing_gold_titles in the results JSON."

        )

    if not targets:

        raise RuntimeError(

            "No targets have a gold inspiration represented in the 3000-paper corpus. "

            "The benchmark's inspiration titles and the corpus titles may not match."

        )

    return targets





# =============================================================================

# BM25 RETRIEVAL AND LLM RERANKING

# =============================================================================

def build_bm25_index(pool: list[dict[str, Any]]) -> BM25Okapi:

    tokenized_docs = [tokenize(paper["text"]) for paper in pool]

    if any(not tokens for tokens in tokenized_docs):

        raise ValueError("At least one paper has no searchable tokens; check the corpus.")

    return BM25Okapi(tokenized_docs)





def bm25_retrieve(item: dict[str, Any], bm25: BM25Okapi) -> list[dict[str, Any]]:

    """Rank the complete fixed pool by BM25 using the target title + background question."""

    query = f"{item['title']}\n{item['background']}"

    query_tokens = tokenize(query)

    scores = bm25.get_scores(query_tokens)

    pool = item["pool"]

    # Stable tie-breaking by original corpus order for reproducibility.

    ranked_indices = sorted(range(len(pool)), key=lambda i: (-float(scores[i]), i))

    candidates = []

    for idx in ranked_indices[: min(BM25_TOP_K, len(pool))]:

        paper = pool[idx]

        candidates.append({**paper, "bm25_score": float(scores[idx])})

    return candidates





def parse_selected_indices(output: str, candidate_count: int, limit: int) -> list[int]:

    """Parse model-selected 1-based indices, tolerating JSON or a plain comma list."""

    text = clean_text(output)

    selected: list[int] = []



    # Preferred format: {"selected": [2, 5, 11]}

    for match in re.finditer(r"\{[^{}]*\}", text, flags=re.S):

        try:

            obj = json.loads(match.group(0))

        except json.JSONDecodeError:

            continue

        values = obj.get("selected") if isinstance(obj, dict) else None

        if isinstance(values, list):

            for value in values:

                try:

                    n = int(value)

                except (TypeError, ValueError):

                    continue

                if 1 <= n <= candidate_count and n not in selected:

                    selected.append(n)

        if selected:

            break



    # Fallback to the last simple line of indices, e.g. "2, 5, 11".

    if not selected:

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line in reversed(lines):

            line = re.sub(r"^```(?:json)?|```$", "", line, flags=re.I).strip()

            if re.fullmatch(r"[\s,;**\\-\\[\\]**(){}\d]+", line):

                numbers = [int(x) for x in re.findall(r"\d+", line)]

                valid = [n for n in numbers if 1 <= n <= candidate_count]

                if valid:

                    selected = list(dict.fromkeys(valid))

                    break



    return selected[:limit]





def llm_rerank(

    item: dict[str, Any], candidates: list[dict[str, Any]],

) -> tuple[list[dict[str, Any]], str]:

    """Ask Qwen to select the final three among the BM25 top 30 candidates."""

    if not candidates:

        return [], ""

    if len(candidates) <= FINAL_INSPIRATIONS:

        return candidates[:FINAL_INSPIRATIONS], "(fewer candidates than rerank threshold)"



    listing_lines = []

    for i, paper in enumerate(candidates, start=1):

        abstract = paper.get("abstract", "")[:MAX_ABSTRACT_CHARS_RERANK]

        listing_lines.append(

            f"{i}. Title: {paper['title']}\n"

            f"Abstract: {abstract if abstract else '(No abstract available)'}"

        )

    listing = "\n\n".join(listing_lines)

    prompt = f"""You are selecting scientific inspiration papers for hypothesis generation.



Research question:

{item['background']}



Target paper title:

{item['title']}



Candidate papers (each includes title and abstract):

{listing}



Select exactly {min(FINAL_INSPIRATIONS, len(candidates))} papers that are most likely to inspire a

NEW, non-obvious, scientifically plausible and testable hypothesis for the research question.



Selection criteria:

\- Prefer transferable mechanisms, methods, materials, reactions, or phenomena.

\- Prefer scientifically useful connections, not only keyword overlap.

\- Avoid selecting papers merely because they discuss the same topic.

\- Choose a diverse set when several candidates offer different useful mechanisms.



Return only valid JSON in this exact format, with candidate numbers (not paper IDs):

{{"selected": [2, 5, 11]}}

No explanations or extra text."""

    # Qwen is used for this short structured choice to avoid DeepSeek-R1 spending

    # the output budget on a long reasoning trace before returning the JSON selection.

    output = ask(prompt, max_tokens=512, model=SELECTION_MODEL)

    indices = parse_selected_indices(output, len(candidates), FINAL_INSPIRATIONS)



    expected = min(FINAL_INSPIRATIONS, len(candidates))

    if len(indices) < expected:

        print(

            f"  WARNING: {SELECTION_MODEL} returned only {len(indices)} parseable selection(s); "

            "filling remaining slots from BM25 order. Inspect rerank_output before interpreting recall.",

            flush=True,

        )



    selected = [candidates[n - 1] for n in indices]

    # If the model returns too few or malformed choices, fill remaining slots with the

    # highest-ranked BM25 candidates not already selected, and record the raw response.

    selected_ids = {paper["id"] for paper in selected}

    for paper in candidates:

        if len(selected) >= min(FINAL_INSPIRATIONS, len(candidates)):

            break

        if paper["id"] not in selected_ids:

            selected.append(paper)

            selected_ids.add(paper["id"])

    return selected[:FINAL_INSPIRATIONS], output





def gold_retrieve(item: dict[str, Any]) -> list[dict[str, Any]]:

    """Oracle retrieval: gold inspirations represented in the fixed master pool."""

    gold_ids = set(item["gold_ids"])

    return [paper for paper in item["pool"] if paper["id"] in gold_ids]





# =============================================================================

# COMPOSITION

# =============================================================================

def compose(

    item: dict[str, Any],

    inspirations: list[dict[str, Any]],

    n: int,

    refine: bool = False,

) -> list[str]:

    if not inspirations:

        return []



    inspiration_text = "\n\n".join(

        f"- {paper['text'][:MAX_TEXT_CHARS_COMPOSE]}" for paper in inspirations

    )

    output = ask(

        f"""Research question:

{item['background']}



Inspiration papers:

{inspiration_text}



Write {n} different, specific, mechanistic and testable hypotheses.

Each hypothesis should:

\- directly address the research question;

\- use a meaningful concept from at least one inspiration paper;

\- propose a plausible mechanism;

\- make a concrete prediction;

\- be experimentally testable;

\- avoid merely restating the research question.



Separate hypotheses with a line containing only:

###



Do not number the hypotheses. Do not include an introductory paragraph.""",

        max_tokens=3000,

    )



    hypotheses = [part.strip() for part in output.split("###") if part.strip()][:n]

    if len(hypotheses) < n:

        # Fallback for models that do not honor the ### separator.

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", output) if len(p.strip()) > 40]

        if len(paragraphs) > len(hypotheses):

            hypotheses = paragraphs[:n]



    if refine:

        refined: list[str] = []

        for hypothesis in hypotheses:

            improved = ask(

                f"""Research question:

{item['background']}



Draft hypothesis:

{hypothesis}



Improve the hypothesis to be more specific, mechanistic, scientifically plausible,

novel, directly connected to the research question, and experimentally testable.

Preserve its core scientific idea. Reply with the improved hypothesis only.""",

                max_tokens=1200,

            ).strip()

            refined.append(improved if improved else hypothesis)

        hypotheses = refined

    return hypotheses[:n]





# =============================================================================

# RANKING (QWEN SELECTOR; DEEPSEEK-R1 CAN EXHAUST SHORT TOKEN BUDGETS THINKING)

# =============================================================================

def rank(item: dict[str, Any], hypotheses: list[str]) -> str:

    if not hypotheses:

        return ""

    if len(hypotheses) == 1:

        return hypotheses[0]



    listing = "\n\n".join(f"{i + 1}. {hyp}" for i, hyp in enumerate(hypotheses))

    output = ask(

        f"""Research question:

{item['background']}



Candidate hypotheses:

{listing}



Choose the SINGLE hypothesis that is best overall, considering:

1\. novelty;

2\. scientific validity;

3\. mechanistic quality;

4\. usefulness;

5\. testability;

6\. how directly it addresses the research question.



Return only the candidate number (one integer).""",

        max_tokens=256,

        model=SELECTION_MODEL,

    )

    numbers = [n for n in re.findall(r"\d+", output)]

    if numbers:

        selected = int(numbers[-1])

        if 1 <= selected <= len(hypotheses):

            return hypotheses[selected - 1]

    return hypotheses[0]





# =============================================================================

# JUDGE / SCORING

# =============================================================================

_score_cache: dict[tuple[str, str], int] = {}

_key_elements: dict[str, list[str]] = {}





def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:

    request = urllib.request.Request(

        url,

        data=json.dumps(payload).encode("utf-8"),

        headers={"Content-Type": "application/json", **headers},

    )

    try:

        with urllib.request.urlopen(request, timeout=120) as response:

            return json.loads(response.read().decode("utf-8"))

    except urllib.error.HTTPError as exc:

        detail = exc.read().decode("utf-8", errors="replace")

        raise RuntimeError(f"Judge API error ({exc.code}): {detail}") from exc

    except urllib.error.URLError as exc:

        raise RuntimeError(f"Could not reach judge API: {exc}") from exc





def judge_ask(prompt: str, max_tokens: int = 128) -> str:

    if JUDGE_PROVIDER == "ollama":

        return ask(prompt, max_tokens=max_tokens, model=JUDGE_MODEL)



    if JUDGE_PROVIDER == "gemini":

        time.sleep(13)  # Conservative pause for low free-tier request limits.

        key = os.environ.get("GEMINI_API_KEY")

        if not key:

            raise RuntimeError("GEMINI_API_KEY is not set.")

        response = _post_json(

            f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent",

            {

                "contents": [{"parts": [{"text": prompt}]}],

                "generationConfig": {"temperature": 0, "maxOutputTokens": max(100, max_tokens)},

            },

            {"x-goog-api-key": key},

        )

        try:

            return response["candidates"][0]["content"]["parts"][0]["text"]

        except (KeyError, IndexError, TypeError):

            return ""



    if JUDGE_PROVIDER == "deepseek":

        key = os.environ.get("DEEPSEEK_API_KEY")

        if not key:

            raise RuntimeError("DEEPSEEK_API_KEY is not set.")

        response = _post_json(

            "https://api.deepseek.com/chat/completions",

            {

                "model": JUDGE_MODEL,

                "temperature": 0,

                "max_tokens": max(10, max_tokens),

                "stream": False,

                "messages": [{"role": "user", "content": prompt}],

            },

            {"Authorization": "Bearer " + key},

        )

        return response["choices"][0]["message"]["content"]



    raise ValueError(f"Unsupported JUDGE_PROVIDER: {JUDGE_PROVIDER}")





def key_elements(item: dict[str, Any]) -> list[str]:

    item_id = str(item["id"])

    if item_id not in _key_elements:

        output = judge_ask(

            f"""Gold hypothesis:

{item['gold_hypothesis']}



List the 3 most specific key elements of this hypothesis: its particular mechanism,

material, or method. Write exactly 3 lines, one element per line, each under 20 words.

No numbering and no extra text.""",

            max_tokens=512,

        )

        lines = [line.strip(" -*\t0123456789.)") for line in output.splitlines() if line.strip()]

        _key_elements[item_id] = lines[:3] or [item["gold_hypothesis"][:200]]

    return _key_elements[item_id]





def score(item: dict[str, Any], hypothesis: str) -> int:

    """Score 0-5 based on how many gold-hypothesis key elements appear in a candidate."""

    if not hypothesis:

        return 0

    cache_key = (str(item["id"]), hypothesis)

    if cache_key in _score_cache:

        return _score_cache[cache_key]



    elements = key_elements(item)

    if not elements:

        return 0

    yes = 0

    for element in elements:

        output = judge_ask(

            f"""Candidate hypothesis:

{hypothesis}



Gold-hypothesis key element:

{element}



Does the candidate hypothesis include this element (the same specific material,

mechanism, or method), even if worded differently?

Answer Yes if it does. Answer No if it is missing or replaced by something different.

Reply with one word: Yes or No.""",

            max_tokens=128,

        )

        if output.strip().casefold().startswith("yes"):

            yes += 1

    result = round(5 * yes / len(elements))

    _score_cache[cache_key] = result

    return result





def judge_check(data: list[dict[str, Any]], n: int = 3) -> None:

    dummy = [{"id": "NONE", "title": "No inspiration", "abstract": "", "text": "(no inspiration given)"}]

    print("\nJUDGE CHECK (self should be high, other paper/no inspiration should be lower)\n")

    for item in data[: min(n, len(data))]:

        other = data[-1]

        candidate = (compose(item, dummy, 1) or [""])[0]

        print(

            f"target {item['id']:>4} | self={score(item, item['gold_hypothesis'])} | "

            f"other={score(item, other['gold_hypothesis'])} | "

            f"no-inspiration={score(item, candidate)}",

            flush=True,

        )





# =============================================================================

# METRICS / ONE TARGET

# =============================================================================

def recall_of_gold(selected: list[dict[str, Any]], gold_ids: list[str]) -> float:

    relevant = set(gold_ids)

    if not relevant:

        return 0.0

    selected_ids = {paper["id"] for paper in selected}

    return len(selected_ids & relevant) / len(relevant)





def run_item(item: dict[str, Any], bm25: BM25Okapi) -> dict[str, Any]:

    print(f"  Target: {item['title'][:100]}", flush=True)

    if item["missing_gold_titles"]:

        print(

            f"  Gold coverage: {len(item['gold_ids'])}/{len(item['all_gold_titles'])}; "

            f"missing from corpus: {item['missing_gold_titles']}",

            flush=True,

        )



    print("  [1/8] BM25 retrieves top 30 from the 3000-paper pool...", flush=True)

    bm25_candidates = bm25_retrieve(item, bm25)

    top30_recall = recall_of_gold(bm25_candidates, item["gold_ids"])

    print(f"  BM25 top-{len(bm25_candidates)} gold recall (available golds): {top30_recall:.2f}", flush=True)



    print(f"  [2/8] {SELECTION_MODEL} reranks the candidates and selects 3...", flush=True)

    base_insp, rerank_output = llm_rerank(item, bm25_candidates)

    selected_recall = recall_of_gold(base_insp, item["gold_ids"])

    print(f"  Final selected gold recall (available golds): {selected_recall:.2f}", flush=True)



    print("  [3/8] Oracle retrieval (available gold inspirations)...", flush=True)

    gold_insp = gold_retrieve(item)



    print("  [4/8] Normal composition...", flush=True)

    base_hypotheses = compose(item, base_insp, N_BASE, refine=False)



    print("  [5/8] Oracle-retrieval composition...", flush=True)

    gold_hypotheses = compose(item, gold_insp, N_BASE, refine=False)



    print("  [6/8] Strong composition...", flush=True)

    strong_hypotheses = compose(item, base_insp, N_STRONG, refine=True)



    print("  [7/8] All-oracle composition...", flush=True)

    gold_strong_hypotheses = compose(item, gold_insp, N_STRONG, refine=True)



    print("  [8/8] Ranking and judging...", flush=True)

    base_draft_scores = [score(item, hypothesis) for hypothesis in base_hypotheses]

    gold_strong_scores = [score(item, hypothesis) for hypothesis in gold_strong_hypotheses]



    base_score = score(item, rank(item, base_hypotheses))

    oracle_retrieval_score = score(item, rank(item, gold_hypotheses))

    oracle_composition_score = score(item, rank(item, strong_hypotheses))

    oracle_ranking_score = max(base_draft_scores) if base_draft_scores else 0

    all_oracles_score = max(gold_strong_scores) if gold_strong_scores else 0



    result = {

        "id": item["id"],

        "excel_row": item["excel_row"],

        "title": item["title"],

        "model": MODEL,

        "selection_model": SELECTION_MODEL,

        "judge_model": JUDGE_MODEL if JUDGE_PROVIDER == "ollama" else JUDGE_PROVIDER,

        "retrieval_method": "BM25 top 30 + Qwen2.5:7b rerank to 3",

        "corpus_file": str(CORPUS_FILE),

        "corpus_size": len(item["pool"]),

        "corpus_gold_coverage": item["corpus_gold_coverage"],

        "all_gold_titles": item["all_gold_titles"],

        "missing_gold_titles": item["missing_gold_titles"],

        "bm25_top30_gold_recall_available": top30_recall,

        "retrieval_hit_ratio": selected_recall,

        "base": base_score,

        "oracle_retrieval": oracle_retrieval_score,

        "oracle_composition": oracle_composition_score,

        "oracle_ranking": oracle_ranking_score,

        "all_oracles": all_oracles_score,

        "base_draft_scores": base_draft_scores,

        "num_base_hypotheses": len(base_hypotheses),

        "num_gold_hypotheses": len(gold_hypotheses),

        "num_strong_hypotheses": len(strong_hypotheses),

        "num_gold_strong_hypotheses": len(gold_strong_hypotheses),

        "bm25_candidates": [

            {"title": paper["title"], "score": round(paper["bm25_score"], 6)}

            for paper in bm25_candidates

        ],

        "rerank_output": rerank_output,

        "selected_titles": [paper["title"] for paper in base_insp],

        "gold_titles_available_in_corpus": [paper["title"] for paper in gold_insp],

    }

    print(

        f"  Scores: base={base_score}, oracle-retrieval={oracle_retrieval_score}, "

        f"oracle-composition={oracle_composition_score}, oracle-ranking={oracle_ranking_score}, "

        f"all-oracles={all_oracles_score}",

        flush=True,

    )

    return result





# =============================================================================

# SUMMARY

# =============================================================================

STAGES = ["oracle_retrieval", "oracle_composition", "oracle_ranking"]

KEYS = ["base", *STAGES, "all_oracles"]





def shares(rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:

    means = {key: statistics.mean(float(row[key]) for row in rows) for key in KEYS}

    gains = {key: means[key] - means["base"] for key in STAGES + ["all_oracles"]}

    positive_gains = {key: max(gains[key], 0.0) for key in STAGES}

    total = sum(positive_gains.values())

    stage_share = {

        key: (100.0 * positive_gains[key] / total if total else 0.0)

        for key in STAGES

    }

    return means, gains, stage_share





def summarize(rows: list[dict[str, Any]]) -> None:

    if not rows:

        print("\nNo results were produced.")

        return

    means, gains, stage_share = shares(rows)

    bootstraps = {key: [] for key in STAGES}

    if len(rows) >= 2:

        for _ in range(1000):

            sample = [random.choice(rows) for _ in rows]

            _, _, sampled_shares = shares(sample)

            for key in STAGES:

                bootstraps[key].append(sampled_shares[key])



    mean_top30_recall = statistics.mean(row["bm25_top30_gold_recall_available"] for row in rows)

    mean_selected_recall = statistics.mean(row["retrieval_hit_ratio"] for row in rows)

    mean_corpus_coverage = statistics.mean(row["corpus_gold_coverage"] for row in rows)



    print("\n" + "=" * 100)

    print("MOOSE-CHEM ORACLE EXPERIMENT — BM25 + DEEPSEEK-R1:7B COMPOSITION")

    print("=" * 100)

    print(f"Completed targets: {len(rows)}")

    print(f"Mean base score (0-5): {means['base']:.2f}")

    print(f"Mean corpus gold coverage: {mean_corpus_coverage:.2f}")

    print(f"Mean BM25 top-30 recall (available golds): {mean_top30_recall:.2f}")

    print(f"Mean final top-3 recall (available golds): {mean_selected_recall:.2f}\n")

    print(f"{'Stage':<22}{'Gain (points)':>16}{'Share of positive gains %':>27}{'% of headroom':>18}{'95% bootstrap interval':>25}")

    for key in STAGES:

        if len(rows) >= 2:

            values = sorted(bootstraps[key])

            lower = values[max(0, int(0.025 * len(values)))]

            upper = values[min(len(values) - 1, int(0.975 * len(values)) - 1)]

        else:

            lower = upper = stage_share[key]

        headroom = 100 * gains[key] / (5 - means["base"]) if means["base"] < 5 else 0.0

        print(

            f"{key:<22}{gains[key]:>16.2f}{stage_share[key]:>27.1f}"

            f"{headroom:>18.1f}{f'[{lower:.0f}, {upper:.0f}]':>25}"

        )

    print(f"\nAll three oracles together: gain {gains['all_oracles']:.2f}")

    print(f"Sum of single-stage gains:  {sum(gains[key] for key in STAGES):.2f}")

    if any(row["corpus_gold_coverage"] < 1.0 for row in rows):

        print(

            "\nWARNING: some gold inspirations are missing from the 3000-paper corpus. "

            "Oracle scores for those targets use only gold inspirations present in the corpus."

        )

    if len(rows) < 10:

        print("WARNING: small pilot; do not treat these results as definitive.")

    print("=" * 100)

    print(f"Results JSON: {RESULTS_FILE.resolve()}")





# =============================================================================

# MAIN

# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 76)

    print("MOOSE-CHEM ORACLE EXPERIMENT — BM25 + DEEPSEEK-R1:7B COMPOSITION")

    print("=" * 76)

    print(f"Hypothesis generation/refinement model: {MODEL}")

    print(f"Reranking/ranking model: {SELECTION_MODEL}")

    print(f"Judge model: {JUDGE_MODEL if JUDGE_PROVIDER == 'ollama' else JUDGE_PROVIDER}")

    print(f"BM25 corpus: {CORPUS_FILE}")

    print(f"Benchmark: {DATA_FILE}")

    print(f"BM25 top K: {BM25_TOP_K}; LLM final inspirations: {FINAL_INSPIRATIONS}")

    print(f"LIMIT: {LIMIT}; judge check: {JUDGE_CHECK}; judge provider: {JUDGE_PROVIDER}")

    print("=" * 76)



    check_ollama()

    data = load_data()

    if LIMIT is not None:

        data = data[:LIMIT]

    if not data:

        raise RuntimeError("No usable target papers were selected.")

    print(f"Running on {len(data)} target paper(s).", flush=True)



    # All targets share the exact same 3000-paper pool; build its BM25 index once.

    bm25_index = build_bm25_index(data[0]["pool"])



    if JUDGE_CHECK:

        judge_check(data,n=1)

        raise SystemExit(0)



    results: list[dict[str, Any]] = []



    def save_results() -> None:

        with RESULTS_FILE.open("w", encoding="utf-8") as handle:

            json.dump(results, handle, indent=2, ensure_ascii=False)



    for index, item in enumerate(data, start=1):

        print(f"\nTarget {index}/{len(data)}\n" + "-" * 76, flush=True)

        try:

            result = run_item(item, bm25_index)

            results.append(result)

            save_results()

            print(f"  Saved {len(results)} result(s) to {RESULTS_FILE}", flush=True)

        except KeyboardInterrupt:

            save_results()

            print(f"\nInterrupted. Saved {len(results)} completed result(s).", flush=True)

            raise

        except Exception as exc:

            save_results()

            print(f"\nFailed on target {index} ({item['id']}): {exc!r}", flush=True)

            print(f"Saved {len(results)} completed result(s) to {RESULTS_FILE}.", flush=True)

            raise



    summarize(results)
