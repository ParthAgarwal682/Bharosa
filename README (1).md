# Bharosa — a trusted search engine for families who can't afford care

> CSD358 IR Hackathon · 36 hours · team of 4 · built with Cursor Pro
> Primary track: **T5 (Indic / Hinglish search)** with **T1 (RAG)** and a small **T4 (live crawl + freshness)** component.
> Decide the final track label at hour 30, after you see which part has the strongest evidence (see Section 14).

*"Bharosa" means trust.*

---

## 0. How to use this document

This file is both the **project README** (Sections 1–8, 13–14 go into your repo as-is) and the **build walkthrough** (Sections 9–12 are for your team while building). Delete the walkthrough sections from the final README if you want it shorter.

Things marked **VERIFY** are facts you must check yourselves before relying on them (sources, licenses, robots.txt). Do not skip these.

---

## 1. What the project does (plain language)

When someone in a family falls sick, the family has to work out three things quickly:

1. **Can the medicine cost less?** Many brands have cheaper medicines with the same active ingredient (the "salt") and strength.
2. **Is there any scheme that can help pay?** Information on government health schemes is scattered, formal, and changes over time.
3. **Can I trust this message or website?** Panicked families are targeted by fake "registration fee" messages.

Bharosa is a search system for exactly this. The user types the way a normal person types, with spelling mistakes or in Hinglish. The system searches a continuously refreshed collection of official pages and a medicine database, and answers with the source and the "last checked" date.

### The three modules

| # | Module | User types | System returns | Uses LLM? |
|---|---|---|---|---|
| 1 | **Medicine finder** | `pantocid 40 sasta kya milega` | Same-salt, same-strength, same-form alternatives ranked by price | **No** (never invent drug facts) |
| 2 | **Scheme finder** | `papa ke heart operation ke liye yojana, UP, income 2 lakh` | Short answer on eligibility, coverage, documents, each sentence cited to an official page zone | **Yes** (RAG) |
| 3 | **Claim checker** (small) | Pasted forward: `pay Rs 500 to activate free health card` | Verdict: does this fee claim contradict the official page? Shows the official line | **Yes, lightly** (explanation only) |

### What it is not

Not a doctor. It never diagnoses, never recommends a drug for a symptom, and always tells the user to confirm substitutions with a doctor or pharmacist. It never asks for Aadhaar numbers, phone numbers or medical records.

---

## 2. Why this is an IR project

The LLM never "knows" the answer. The IR system finds the evidence first, and the LLM only explains what was found. Every sentence in a generated answer must point to a retrieved zone, and a checker verifies it. If retrieval is weak, the system refuses instead of guessing.

### Where RAG is used and where it is deliberately not

- **Module 1 has no LLM.** A wrong generated drug name is dangerous, and plain ranked retrieval is enough for this need.
- **Module 2 uses RAG** because the answer is spread across several pages and zones (eligibility on one, coverage on another, documents on a third), and the user cannot read five government pages.
- **Module 3 uses RAG lightly**, only to explain a verdict from a retrieved official passage.

This choice is itself a design decision to defend in your report: *generation only where retrieval alone cannot answer, and never for drug facts.*

---

## 3. System flow

```text
                         ┌──────────────────────────────┐
                         │  Official sources (seeds)    │
                         └──────────────┬───────────────┘
                                        │ polite crawl (robots.txt, delays,
                                        │ ETag, content hash)
                         ┌──────────────▼───────────────┐
                         │  CRAWLER + FRESHNESS         │  Person A
                         │  frontier, scheduler, change │
                         │  log in SQLite               │
                         └──────────────┬───────────────┘
                                        │ raw pages
                         ┌──────────────▼───────────────┐
                         │  CLEAN + ZONE SPLIT          │  Person C
                         │  eligibility / benefits /    │
                         │  documents / how-to-apply    │
                         └──────────────┬───────────────┘
        static medicine data            │ zone docs
        (declared dataset)              │
              │                         │
   ┌──────────▼──────────┐   ┌──────────▼───────────────┐
   │ MEDICINE INDEX      │   │ SCHEME INDEX             │
   │ n-gram + parametric │   │ inverted + positional +  │
   │ (salt/strength/form)│   │ parametric (state, cond.)│
   │ Person B            │   │ Person C                 │
   └──────────┬──────────┘   └──────────┬───────────────┘
              │                         │
              │      QUERY UNDERSTANDING (Hinglish lexicon,
              │      spelling normalisation, filter extraction)
              │                         │
              │                ┌────────▼─────────────────┐
              │                │ RANKING: tf-idf/cosine,  │
              │                │ BM25 baseline, g(d),     │
              │                │ freshness, heap top-K    │
              │                └────────┬─────────────────┘
              │                         │ top-K zones
              │                ┌────────▼─────────────────┐
              │                │ RAG + CITATION CHECKER   │  Person D
              │                │ refuse if low confidence │
              │                └────────┬─────────────────┘
              │                         │
              └──────────┬──────────────┘
                         ▼
                 CLI / Streamlit demo
                 + evaluation harness (Person D)
```

---

## 4. IR principles and where they live in the code

Fill the **"What it changed"** column with real numbers from your evaluation. That column is what earns the "used correctly, with choices explained" marks.

| IR principle | Where (file) | Why we chose it | What it changed |
|---|---|---|---|
| Inverted index + postings (own implementation) | `bharosa/index/inverted.py` | Fast lookup; we implement it ourselves so it can be inspected in the demo | *(fill)* |
| Positional index, phrase queries | `bharosa/index/positional.py` | Exact scheme names like "Ayushman Bharat" | *(fill)* |
| Tokenisation, normalisation, case folding | `bharosa/text/normalize.py` | Hinglish and mixed scripts | *(fill)* |
| Hinglish lexicon + query expansion | `bharosa/text/hinglish.py` | "sasta", "yojana", "dawai" map to canonical terms | *(fill)* |
| Soundex / phonetic baseline | `bharosa/medicine/phonetic.py` | Syllabus technique, used as a baseline | *(fill)* |
| Character n-gram matching | `bharosa/medicine/ngram.py` | Misspelled brand names | *(fill)* |
| Parametric index | `bharosa/medicine/parametric.py`, `bharosa/index/params.py` | Salt, strength, form, state, condition filters | *(fill)* |
| Zone index and zone weighting | `bharosa/index/zones.py` | Users ask about one zone at a time; this is also our chunking choice for RAG | *(fill)* |
| tf-idf, cosine, lnc.ltc | `bharosa/rank/tfidf.py` | Core ranking model | *(fill)* |
| Heap-based top-K | `bharosa/rank/topk.py` | Efficient result assembly | *(fill)* |
| Static quality g(d) and net score | `bharosa/rank/netscore.py` | Official domains above blogs and agents | *(fill)* |
| Crawl loop, frontier, politeness, robots.txt | `bharosa/crawl/` | Real data; ethics rules in the assignment | *(fill)* |
| URL normalisation, content-seen check | `bharosa/crawl/dedup.py` | Avoid duplicate pages | *(fill)* |
| Shingles + Jaccard | `bharosa/crawl/shingles.py` | Near-duplicate and cloned pages | *(fill)* |
| Adaptive recrawl scheduling | `bharosa/crawl/scheduler.py` | Prices and eligibility change | *(fill)* |
| BM25 (extra points) | `bharosa/rank/bm25.py` (library baseline) | Stronger baseline | *(fill)* |

**Rule for libraries:** implement the inverted index, tf-idf, cosine and top-K yourselves. Use libraries (`rank_bm25`, `jellyfish`, `requests`, `beautifulsoup4`) for baselines and plumbing, and explain what they do in IR terms in the report.

---

## 5. Repository structure

```text
bharosa/
├── README.md
├── requirements.txt
├── .env.example
├── .cursor/
│   └── rules/
│       └── bharosa.mdc          # shared Cursor rules (Section 11)
├── CONTRACTS.md                 # data schemas + function signatures (Section 10)
├── config/
│   ├── seeds.yaml               # crawl seeds, per-host delay, g(d) scores
│   └── hinglish_lexicon.csv     # Hinglish -> canonical term
├── data/
│   ├── raw/                     # crawled HTML/PDF (gitignored if large)
│   ├── medicines/               # declared static dataset(s)
│   ├── snapshots/               # archived page versions for freshness experiment
│   └── bharosa.db               # SQLite: pages, versions, change log
├── bharosa/
│   ├── crawl/
│   │   ├── frontier.py
│   │   ├── fetcher.py           # robots.txt, delays, ETag/If-Modified-Since
│   │   ├── dedup.py
│   │   ├── shingles.py
│   │   └── scheduler.py
│   ├── text/
│   │   ├── normalize.py
│   │   ├── hinglish.py
│   │   └── zones.py             # clean HTML/PDF text and split into zones
│   ├── index/
│   │   ├── inverted.py
│   │   ├── positional.py
│   │   ├── zones.py
│   │   └── params.py
│   ├── medicine/
│   │   ├── loader.py
│   │   ├── phonetic.py
│   │   ├── ngram.py
│   │   ├── parametric.py
│   │   └── search.py
│   ├── rank/
│   │   ├── tfidf.py
│   │   ├── bm25.py
│   │   ├── topk.py
│   │   └── netscore.py
│   ├── rag/
│   │   ├── llm.py               # single wrapper; provider via env var
│   │   ├── answer.py
│   │   ├── citations.py         # sentence-level citation checker
│   │   └── claimcheck.py        # Module 3
│   └── app/
│       ├── cli.py
│       └── streamlit_app.py     # optional, not graded
├── eval/
│   ├── medicine_queries.csv
│   ├── scheme_queries.csv
│   ├── scheme_qrels.csv
│   ├── claims.csv
│   ├── run_eval.py
│   └── results/                 # tables and plots for the report
├── tests/
└── report/
    ├── report.md
    └── figures/
```

---

## 6. Setup

```bash
git clone <your-repo-url> bharosa && cd bharosa
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # add your LLM API key and model name
```

`requirements.txt` (starting point):

```text
requests
beautifulsoup4
lxml
pyyaml
pandas
numpy
scikit-learn      # only for evaluation helpers, NOT for the main index
rank_bm25         # baseline only
jellyfish         # Soundex / Metaphone baselines
rapidfuzz         # fast string similarity helper
pdfplumber        # PDF text extraction
python-dotenv
streamlit         # optional
matplotlib        # evaluation plots
pytest
```

`.env.example`:

```text
LLM_PROVIDER=             # e.g. anthropic / openai (declare in report)
LLM_MODEL=
LLM_API_KEY=
CRAWL_USER_AGENT=BharosaStudentCrawler/0.1 (contact: your-college-email)
```

How to run (fill in the exact commands once built):

```bash
python -m bharosa.crawl.run --seeds config/seeds.yaml     # start in hour 1 and leave it running
python -m bharosa.index.build                              # build indexes from the crawl
python -m bharosa.app.cli                                  # ask questions
python eval/run_eval.py                                    # reproduce all evaluation tables
```

---

## 7. Data sources (VERIFY every one)

For each source, record in `config/seeds.yaml`: URL, robots.txt result, license or terms, format (HTML/PDF/CSV), and the per-host delay you used. Credit every dataset in the report.

| Need | Candidate source | Notes |
|---|---|---|
| Medicine brand → salt → strength → form | A public brand/salt medicine dataset (search Kaggle / data.gov.in) | **VERIFY** license and quality. This is a **static** dataset: say so, do not call it live. |
| Cheaper same-salt options and prices | Jan Aushadhi (PMBJP) official product list and price list | **VERIFY** exact format and where it is published |
| Price ceilings | National drug price regulator notifications | Often PDFs. **VERIFY** whether machine-readable |
| Health schemes | National health scheme portal pages, state health department pages, ministry press releases | **VERIFY** robots.txt. Some portals block bots or serve only PDFs. |
| Ground truth for fee claims | The same official scheme pages | Used by Module 3 |

**Ethics checklist (assignment rules):** obey robots.txt, delay between requests to the same host (start at 3–5 seconds), identify your crawler in the User-Agent, never crawl scam or malicious sites, collect no personal data.

**If a source blocks you or is PDF-only:** switch seeds. Do not work around robots.txt.

---

## 8. Module specifications

### Module 0 — Shared foundations (everyone, hour 0–2)

- Create the repo, branches, `CONTRACTS.md` and `.cursor/rules/bharosa.mdc`.
- Agree on the schemas in Section 10. This is what lets four people work in parallel without blocking each other.

### Module A — Crawler and freshness (Person A)

**Does:** collects official pages politely, detects changes, keeps versions.

**Build:**
1. `fetcher.py`: fetch with robots.txt check (`urllib.robotparser`), per-host delay, custom User-Agent, conditional requests (ETag / If-Modified-Since), timeout and retries.
2. `frontier.py`: priority queue (priority = base importance + staleness). Per-host queues for politeness.
3. `dedup.py`: URL normalisation (lowercase host, drop fragments and tracking params, sort query params) plus content-hash "content seen?" check.
4. `shingles.py`: word shingles (k=5) and Jaccard for near-duplicate pages.
5. `scheduler.py`: adaptive recrawl interval. Halve the interval when a page changed, grow it by a factor (e.g. 1.5) when unchanged, with min and max bounds. Fixed-interval scheduler as the baseline.
6. SQLite tables: `pages(url, domain, g_score, ...)`, `versions(url, fetched_at, content_hash, text_path)`, `changes(url, detected_at, diff_summary)`.

**Honest freshness experiment:** real pages rarely change within 36 hours. Build the experiment on **archived or simulated snapshots** (for example, replay stored versions with synthetic change rates per page), and **label it as simulated** in the video and report. Show the real crawl log separately as proof the crawler is live.

**Done when:** crawler runs unattended for hours, respects delays, logs versions, and the scheduler comparison script outputs changes-caught vs requests-used for adaptive and fixed intervals.

### Module B — Medicine finder (Person B) — the core result

**Does:** maps a messy brand query to salt, strength and form, then ranks cheaper same-salt options.

**Build:**
1. `loader.py`: clean the dataset into records `{brand, salt, strength_value, strength_unit, form, mrp, generic_price, manufacturer}`. Normalise `650mg`, `650 mg`, `650 MG` to one form.
2. `ngram.py`: character 2–3-gram vectors of brand names, cosine similarity. This handles `pantocid`, `pantocide`, `pantosid`.
3. `phonetic.py`: Soundex (and optionally Metaphone) as a **baseline**. Report honestly where it fails on Indian brand names.
4. `parametric.py`: after finding the salt, filter by **same salt AND same strength AND same dosage form**. Do **not** treat extended-release vs immediate-release, or combination drugs, as interchangeable.
5. `search.py`: query parser extracts brand text and strength tokens, matches, then ranks alternatives by price with a net score.
6. Output always includes the disclaimer: confirm with a doctor or pharmacist.

**Baselines for evaluation:** exact match; Soundex; tf-idf over brand tokens; BM25.

**Done when:** 30+ judged queries run, with a table for exact match, Soundex, tf-idf, BM25 and your n-gram + parametric system.

### Module C — Zones, scheme index and ranking (Person C)

**Does:** turns crawled pages into zones, indexes them, and ranks zones for a query.

**Build:**
1. `text/zones.py`: clean HTML/PDF text, split into zones by headings: `eligibility`, `benefits`, `documents`, `how_to_apply`, `other`. Keep `state` and `condition` metadata where detectable.
2. `index/inverted.py`: your own dictionary + postings (with tf), document frequency, idf.
3. `index/positional.py`: positional postings for phrase queries.
4. `index/params.py`: parametric filters (state, condition).
5. `rank/tfidf.py`: lnc.ltc cosine scoring. `rank/topk.py`: heap-based top-K. `rank/netscore.py`: `net = w1*cosine + w2*g(d) + w3*freshness`, with zone weights.
6. `text/hinglish.py`: lexicon-based query expansion from `config/hinglish_lexicon.csv` (`sasta` → `cheap`, `yojana` → `scheme`, `dawai` → `medicine`, …). Start with 100–200 entries covering your test queries.
7. `rank/bm25.py`: baseline using `rank_bm25`.

**Done when:** `search_schemes(query, filters, k)` returns ranked zones; ablations (no zone weight, no g(d), no freshness, no Hinglish expansion) can be switched on and off with flags.

### Module D — RAG, claim checker and evaluation (Person D)

**Does:** writes cited answers, checks them, refuses when unsure, and runs all evaluation.

**Build:**
1. `rag/llm.py`: one wrapper for the LLM provider, reading the key from `.env`. Declare the provider and model in the report.
2. `rag/answer.py`: build a prompt from the top-K zones only, each labelled `[Z1]`, `[Z2]`, …. Instruct: answer in simple Hinglish or English, one claim per sentence, every sentence ends with a zone id, use only the provided text, say "not found in official sources" otherwise. Return a structured answer (sentences + cited ids).
3. `rag/citations.py`: for each answer sentence, compute cosine similarity (using the tf-idf code from Module C) against its cited zone. Below a threshold, flag the sentence as unsupported. Tune the threshold on a small labelled set.
4. **Refusal rule:** if the top score is below a confidence threshold, or the query has no matching state or condition, return "I couldn't find this in official sources" without calling the LLM.
5. `rag/claimcheck.py` (Module 3): extract the fee or deadline claim with simple patterns (`Rs`, `₹`, `fee`, `charge`, `free`), retrieve the closest official zone with cosine similarity (message vs. zone, not Jaccard), compare the claim with the official statement, and generate a short explanation from the retrieved line.
6. `eval/run_eval.py`: runs everything and writes tables and plots to `eval/results/`.

**Done when:** one command reproduces all tables used in the report.

---

## 9. Team split and 36-hour timeline

| Person | Owns | Video segment |
|---|---|---|
| **A** | Crawler, dedup, scheduler, freshness experiment | Live crawl log, frontier, change detection, adaptive vs fixed |
| **B** | Medicine finder | Misspelled queries live, baseline comparison |
| **C** | Zones, indexes, ranking, Hinglish | Postings, weights, scores, ablations |
| **D** | RAG, citation checker, claim checker, evaluation | Cited answer, flagged sentence, refusal, results tables |

**Everyone** writes their own report section and explains their own component in the video. Shared: README, `CONTRACTS.md`, repo hygiene (rotate who merges).

| Hour | A | B | C | D |
|---|---|---|---|---|
| 0–2 | Repo, contracts, seeds, robots.txt checks. **Start the crawler by hour 1–2.** | Find and clean medicine dataset | Define zone schema, lexicon starter | LLM wrapper, query files skeleton |
| 2–8 | Fetcher, frontier, SQLite log | Loader, normalisation, n-gram | Inverted index, tf-idf, top-K | Prompt design, citation checker v1 |
| 8–16 | Dedup, shingles, scheduler | Parametric filter, Soundex baseline, alternatives ranking | Zone splitter on real crawled pages, positional index | Wire RAG to Module C output, refusal rule |
| 16–24 | Snapshot/simulation experiment | Judged queries + eval table | Net score, BM25 baseline, ablation flags, Hinglish expansion | Claim checker, eval harness |
| 24–30 | Freshness results, plots | Fix failures found in eval | Fix failures found in eval | Run full eval, plots, tune thresholds |
| 30–34 | Record own video part | Record own video part | Record own video part | Record own video part, assemble |
| 34–36 | Final README, report assembly, submission form, **buffer** | | | |

**Cut order if time runs short** (cut from the bottom): UI → positional index → claim checker → procedure/extra sources. **Never cut:** medicine matcher eval, scheme retrieval eval, citation checker, the live crawl log.

---

## 10. Contracts (put this in `CONTRACTS.md`)

Agree on these in hour 0–2. Parallel work depends on them.

**Zone document (A → C):**

```json
{
  "doc_id": "string",
  "url": "string",
  "domain": "string",
  "title": "string",
  "zone": "eligibility | benefits | documents | how_to_apply | other",
  "text": "string",
  "state": "string or null",
  "conditions": ["string"],
  "g_score": 0.0,
  "crawled_at": "ISO datetime",
  "last_changed_at": "ISO datetime",
  "content_hash": "string"
}
```

**Medicine record (B):**

```json
{
  "brand": "string",
  "salt": "string",
  "strength_value": 0.0,
  "strength_unit": "mg | mcg | ml | ...",
  "form": "tablet | capsule | syrup | ...",
  "mrp": 0.0,
  "generic_price": 0.0,
  "manufacturer": "string"
}
```

**Function signatures:**

```python
# Module B
def search_medicine(query: str, k: int = 5) -> list[MedicineHit]: ...

# Module C
def search_schemes(query: str, filters: dict, k: int = 5, flags: dict | None = None) -> list[ZoneHit]: ...
#   flags e.g. {"zone_weights": True, "g_score": True, "freshness": True, "hinglish": True, "bm25": False}

# Module D
def answer(query: str, hits: list[ZoneHit]) -> Answer: ...          # sentences + cited zone ids, or refused=True
def check_citations(ans: Answer, hits: list[ZoneHit]) -> list[SentenceCheck]: ...
def check_claim(message: str) -> Verdict: ...
```

If someone must change a contract, they post in the team chat and update `CONTRACTS.md` in the same commit.

---

## 11. Working with Cursor Pro

### 11.1 One-time setup (Person A, then everyone pulls)

1. Create the GitHub repo and push the scaffold from Section 5.
2. Add `.cursor/rules/bharosa.mdc` with the content below, and commit it so all four of you share the same rules.
3. Each person works on their **own branch** (`feat/crawler`, `feat/medicine`, `feat/index-rank`, `feat/rag-eval`). Merge to `main` through short pull requests, at least every 6 hours.
4. Commit before every large Cursor edit so you can revert.

### 11.2 Shared rules file: `.cursor/rules/bharosa.mdc`

```markdown
---
description: Bharosa project rules
alwaysApply: true
---
# Project: Bharosa (CSD358 IR hackathon)
A search system over official health-scheme pages and a medicine dataset, with cited RAG answers.

## Non-negotiable rules
- Implement the inverted index, tf-idf (lnc.ltc), cosine and heap top-K FROM SCRATCH in bharosa/index and bharosa/rank. Use libraries only for baselines (rank_bm25, jellyfish) and plumbing.
- Never use an LLM for medicine matching or drug facts. LLM is used only in bharosa/rag/.
- Every generated answer sentence must carry a zone citation. If retrieval confidence is low, refuse.
- Crawler must obey robots.txt, wait at least 3 seconds between requests to the same host, and send the User-Agent from .env. Never crawl scam or malicious sites. Never collect personal data.
- Follow the schemas and function signatures in CONTRACTS.md exactly. Do not change them without updating CONTRACTS.md.
- Only edit files in the module you were asked to work on.
- No hard-coded or faked results. Evaluation numbers must come from eval/run_eval.py.
- Add a short docstring to every function explaining the IR concept it implements.
- Write a small pytest for each module, using tiny hand-made examples with known answers.

## Style
- Python 3.11, type hints, small functions, no unnecessary abstractions.
- Print intermediate outputs (postings, weights, scores) behind a --verbose flag so they can be shown in the demo video.
```

### 11.3 How to work in Cursor

- Use **Agent mode** for building, but start every task with **Plan**: ask Cursor to propose the plan and files first, read it, then approve.
- Reference files explicitly with `@CONTRACTS.md`, `@.cursor/rules/bharosa.mdc` and the module's files.
- Ask for **one file or one function at a time**, with a test, and run the test before moving on.
- When something breaks, paste the full error and ask Cursor to explain the cause **before** it edits code.
- **Understand every line you will have to explain in the video.** The rubric needs each member to explain their own component. After Cursor writes a function, ask it to explain the function in IR terms, then write 2–3 lines of notes in your own words.
- Keep a running `AI_LOG.md` (date, what you asked Cursor for, what you changed by hand). You will need it for the AI-use declaration.

### 11.4 Prompts by person

Paste each person's **first prompt** into a fresh Cursor Agent chat with `@CONTRACTS.md` and `@.cursor/rules/bharosa.mdc` attached. Then use the follow-ups in order.

#### Person A — Crawler and freshness

**First prompt**

```text
You are helping me build the crawler for Bharosa. Read the project rules and CONTRACTS.md first.
Task: plan, then implement bharosa/crawl/ with:
1) fetcher.py: fetch URLs with robots.txt checking (urllib.robotparser, cached per host), a per-host minimum delay of 3 seconds, the User-Agent from .env, timeouts, retries, and conditional requests using ETag and If-Modified-Since.
2) frontier.py: a priority queue with per-host politeness. Priority = base importance from config/seeds.yaml + staleness.
3) dedup.py: URL normalisation (lowercase host, strip fragments and tracking params, sort query params) and a content-hash "content seen?" check.
4) A SQLite store (data/bharosa.db) with tables pages, versions, changes.
Start with a plan listing files and function signatures. Do not write code until I approve the plan. Only edit bharosa/crawl/ and config/seeds.yaml.
```

**Follow-ups, in order**
1. `Write shingles.py: word 5-shingles and Jaccard similarity, with pytest examples of near-duplicate and different texts.`
2. `Write scheduler.py with two schedulers: FixedInterval and Adaptive (halve interval on change, multiply by 1.5 on no change, with min and max bounds). Both must expose the same interface.`
3. `Write a simulation script that replays snapshots in data/snapshots/ with per-page synthetic change rates, runs both schedulers under the same request budget, and outputs changes caught and mean staleness. Label the output clearly as SIMULATED.`
4. `Write a crawl runner with a --verbose flag that prints the frontier state, robots decisions and detected changes. I will show this in the demo video.`

#### Person B — Medicine finder

**First prompt**

```text
You are helping me build the medicine finder for Bharosa. Read the project rules and CONTRACTS.md first.
Rule: NO LLM in this module. Pure IR and string matching.
Task: plan, then implement bharosa/medicine/:
1) loader.py: load the dataset in data/medicines/ into records matching the Medicine record schema. Normalise strengths like "650mg", "650 mg", "650 MG" into strength_value and strength_unit. Report how many rows were dropped and why.
2) ngram.py: character 2- and 3-gram TF-IDF vectors for brand names and cosine similarity, implemented by us (not a vectorizer library).
3) phonetic.py: Soundex baseline (jellyfish allowed).
4) parametric.py: given a matched brand, find all records with the SAME salt AND strength AND dosage form. Do not treat extended-release vs immediate-release, or combination drugs, as equal.
5) search.py: parse a messy query ("pantocid 40 sasta kya milega") into brand text and strength, match, then rank same-salt alternatives by price. Always return the disclaimer to confirm with a doctor or pharmacist.
Start with a plan. Wait for my approval before coding. Only edit bharosa/medicine/ and tests/.
```

**Follow-ups, in order**
1. `Create eval/medicine_queries.csv with columns query, gold_brand, gold_salt. Help me generate 40 realistic misspelled and Hinglish queries from our dataset. I will manually check and fix the gold labels.`
2. `Write the baseline matchers: exact match, Soundex, tf-idf over brand tokens, BM25. All with the same interface as search.py.`
3. `Write the evaluation script for this module: top-1 accuracy, P@5 and recall for each method, saved as a table to eval/results/medicine.csv, plus a bar chart.`
4. `List 10 queries where our method fails and categorise the failures. I will use them in the limitations section.`

#### Person C — Zones, index and ranking

**First prompt**

```text
You are helping me build the scheme index and ranking for Bharosa. Read the project rules and CONTRACTS.md first.
Rule: implement the inverted index, tf-idf (lnc.ltc), cosine similarity and heap-based top-K from scratch. Use rank_bm25 only for the baseline.
Task: plan, then implement:
1) bharosa/text/zones.py: clean HTML (BeautifulSoup) and PDF (pdfplumber) text and split it into zones (eligibility, benefits, documents, how_to_apply, other) using headings and keyword rules. Output the Zone document schema from CONTRACTS.md.
2) bharosa/index/inverted.py: dictionary + postings with term frequency, document frequency and idf. Add a --verbose dump of postings for chosen terms.
3) bharosa/rank/tfidf.py: lnc.ltc cosine scoring, and rank/topk.py with a min-heap.
4) bharosa/index/params.py: parametric filters on state and condition.
Start with a plan listing files and signatures. Wait for my approval. Only edit bharosa/text, bharosa/index, bharosa/rank, tests/.
```

**Follow-ups, in order**
1. `Write positional.py and phrase-query support, with tests on tiny documents.`
2. `Write netscore.py: net = w1*cosine + w2*g(d) + w3*freshness, with zone weights. Make each component switchable via the flags dict in CONTRACTS.md.`
3. `Write hinglish.py that expands queries using config/hinglish_lexicon.csv, and help me draft the first 150 lexicon entries for health and scheme vocabulary. I will review them.`
4. `Write bm25.py as a baseline using rank_bm25, with the same interface as search_schemes.`
5. `Write eval/scheme_queries.csv and scheme_qrels.csv templates, and an ablation runner (no zone weights, no g(d), no freshness, no Hinglish expansion) that outputs P@5, P@10 and recall.`

#### Person D — RAG, claim checker and evaluation

**First prompt**

```text
You are helping me build the RAG layer and evaluation for Bharosa. Read the project rules and CONTRACTS.md first.
Rules: the LLM only sees the retrieved zones; every answer sentence carries a zone citation; refuse when retrieval confidence is low; never use the LLM for drug facts.
Task: plan, then implement bharosa/rag/:
1) llm.py: one wrapper reading LLM_PROVIDER, LLM_MODEL and LLM_API_KEY from .env.
2) answer.py: build a prompt from the top-K zones labelled [Z1], [Z2], ...; ask for simple Hinglish or English, one claim per sentence, each ending with a zone id, only from the provided text, and "not found in official sources" otherwise. Return structured sentences with cited ids.
3) citations.py: for each sentence, cosine similarity (tf-idf, reuse Module C code) against its cited zone; flag sentences below a threshold as unsupported.
4) A refusal rule: if the top retrieval score is below a threshold, return a refusal without calling the LLM.
Start with a plan. Wait for my approval. Until Module C is ready, use a small mock retriever returning hand-written zones, clearly marked as a mock. Only edit bharosa/rag/, eval/ and tests/.
```

**Follow-ups, in order**
1. `Write claimcheck.py: extract fee/free/deadline claims from a pasted message using simple patterns (Rs, ₹, fee, charge, free), retrieve the closest official zone by cosine similarity, compare the claim with the official line, and generate a one-line explanation from the retrieved text.`
2. `Create eval/claims.csv with columns message, label (consistent/contradicts), source_note. Help me draft 40 messages. I must mark in the file which ones I wrote myself, because hand-written negatives make results look better than reality.`
3. `Write eval/run_eval.py that calls the medicine, scheme and claim evaluations and the RAG evaluation: percent of answer sentences supported by their citation (with and without the checker), refusal rate on 10 out-of-scope queries, and correctness on judged scheme queries. Save tables and plots to eval/results/.`
4. `Help me choose the citation and refusal thresholds using a small labelled set, and report the precision/recall trade-off as a plot.`

### 11.5 Integration prompt (hour 16–20, whoever merges)

```text
Read CONTRACTS.md and all modules. Write bharosa/app/cli.py that routes a query: if it looks like a medicine query, call search_medicine; if it looks like a scheme query, call search_schemes then answer(); if it contains a pasted message with a fee claim, call check_claim. Add --verbose to print postings, weights, scores and retrieved zones. Do not change any module signatures. List any mismatches you find between modules instead of silently patching them.
```

---

## 12. Evaluation plan

All numbers in the report must come from `eval/run_eval.py`.

| Evaluation | Data | Systems compared | Metrics |
|---|---|---|---|
| Medicine matching | 30–40 judged misspelled/Hinglish queries | Exact match, Soundex, tf-idf, BM25, **ours** (n-gram + parametric) | Top-1 accuracy, P@5, recall |
| Scheme retrieval | 30+ judged queries with relevance labels | Plain tf-idf, BM25, **ours** (zones + g(d) + freshness + Hinglish) | P@5, P@10, recall |
| Ablations | Same scheme queries | Remove one component at a time | Change in P@5 |
| Freshness | Archived/simulated snapshots | Fixed vs adaptive recrawl, same request budget | Changes caught, mean staleness. **Label as simulated** |
| RAG | Judged scheme queries + 10 out-of-scope queries | With vs without citation checker | % sentences supported, refusal rate, answer correctness |
| Claim checker | 40–60 labelled messages | Keyword baseline vs retrieval + comparison | Precision, recall. **Note which negatives you wrote yourselves** |

Judge queries yourselves, write the labelling rules down first, and have a second person re-label 10 queries to report agreement.

---

## 13. Demo video script (5–8 minutes, no slides)

| Time | Content | Who |
|---|---|---|
| 0:00–1:00 | The problem, the family story, why T5/T1/T4 | D |
| 1:00–2:00 | Live crawl log: robots check, delays, a detected change | A |
| 2:00–3:30 | Medicine finder: misspelled queries, intermediate scores, one failure case | B |
| 3:30–5:00 | Scheme search: show postings, tf-idf weights, scores, ablation toggle | C |
| 5:00–6:30 | RAG answer with citations, a flagged unsupported sentence, a refusal | D |
| 6:30–7:30 | Evaluation tables vs baselines, simulated-freshness label | A + D |
| 7:30–8:00 | Limitations and roadmap | All |

Rules from the assignment: show the real working system, show actual intermediate output, include at least one limitation, and each member explains their own part. No faked or hard-coded demos.

---

## 14. Report outline (≤ 8 pages excluding references and appendix)

1. **Problem and track relevance** — user need, papers referred, why T5/T1/T4. Choose the track label based on your evidence: **T5** if the medicine/Hinglish results are strongest; **T4** only if the crawler and freshness work is clearly the best-evidenced part.
2. **How we used IR** — Section 4 table with real "what it changed" numbers, and the pipeline diagram.
3. **Beyond IR** — the LLM, dataset, and any library, and why generation is limited to Modules 2 and 3.
4. **Novelty** — safe same-salt/strength/form matching for messy Hinglish queries, zone-level cited answers over live official pages, refusal when evidence is weak, claims checked against current official text.
5. **Evaluation** — tables and graphs from Section 12.
6. **Limitations and next steps** — include real failures, the static medicine dataset, simulated freshness, hand-written negatives. Add the roadmap.
7. **Work division** — Section 9 table. For information only.
8. **AI-use declaration** — see below.

### AI-use declaration template

```text
AI tools used:
- Cursor Pro (model: <name>) used for: scaffolding, code drafts for <modules>, unit tests, prompt drafting.
- LLM API (<provider/model>) used inside the system for: Module 2 answers and Module 3 explanations only.
- <Chat assistant> used for: brainstorming the idea, scoping, planning, README draft.
What we wrote or changed by hand: <list>.
How we verified AI output: tests, manual label checking, evaluation from run_eval.py.
Datasets and libraries: <list with licenses and credits>.
```

Keep `AI_LOG.md` current while you build so this section is accurate.

---

## 15. Safety and ethics

- Medicine output is information only. Always show: *"Confirm any substitution with your doctor or pharmacist."*
- Never suggest a medicine for a symptom and never give dosage advice.
- Never ask for or store Aadhaar numbers, phone numbers or medical records.
- Never visit scam or malicious sites. Module 3 works only from official pages and text you wrote or collected from public reports.
- Credit every dataset and be transparent about what is static versus live.

---

## 16. Honesty checklist (these protect your marks)

- [ ] Every number in the report comes from `eval/run_eval.py`.
- [ ] Freshness comparison is labelled **simulated**; the real crawl log is shown separately.
- [ ] Medicine dataset is described as **static**, not live.
- [ ] Hand-written claim examples are marked as such.
- [ ] At least one real failure case is shown in the video.
- [ ] README commands reproduce the demo from a fresh clone.
- [ ] Each person can explain their own code without reading it.
- [ ] Project was built inside the 36-hour window, not reused from an earlier project.

---

## 17. Limitations and roadmap (course-project continuation)

**Known limitations (update with real ones):** small corpus, a handful of sources, static medicine data, lexicon-based Hinglish, simple fee extraction, threshold-based refusal.

**Roadmap:**
1. Hybrid sparse + dense retrieval, with analysis of when each wins.
2. Learning-to-rank over the net-score features.
3. More states and schemes, then a UK/NHS version.
4. Better Hinglish and transliteration handling.
5. Treatment and procedure cost information, once reliable data is found.
6. Forwarded-message checker for chat-app text, voice input in Hindi.
7. PageRank-style authority between official pages for g(d).

---

## 18. Submission checklist

- [ ] Repo link with this README (setup, run, data sources, what works, what is planned)
- [ ] Demo video (5–8 min, unlisted link)
- [ ] Report PDF (≤ 8 pages) with work division and AI-use declaration
- [ ] Form submitted before the 36-hour deadline
