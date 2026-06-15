# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

I chose dorm reviews at Duke University. This knowledge is valuable because as a freshman coming into Duke I had no visibility into what the dorms are like. Official channels don't give the real scoop like student reviews do.

## Document Sources

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Ratemydorm.com| 4 reviews for Basset dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-bassett |
| 2 | Ratemydorm.com| 4 reviews for Wilson dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-wilson |
| 3 | Ratemydorm.com| 4 reviews for Giles dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-giles |
| 4 | Ratemydorm.com| 3 reviews for Keohane quad | https://www.ratemydorm.com/reviews/duke-university/duke-university-keohane-quad |
| 5 | Ratemydorm.com| 3 reviews for Trinity dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-trinity |
| 6 | Ratemydorm.com| 3 reviews for Wannamaker quad | https://www.ratemydorm.com/reviews/duke-university/duke-university-wannamaker-quad |
| 7 | Ratemydorm.com| 2 reviews for Blackwell dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-blackwell |
| 8 | Ratemydorm.com| 2 reviews for Craven quad | https://www.ratemydorm.com/reviews/duke-university/duke-university-craven-quad |
| 9 | Ratemydorm.com| 2 reviews for Kilgo quad | https://www.ratemydorm.com/reviews/duke-university/duke-university-kilgo-quad |
| 10 |Ratemydorm.com| 2 reviews for Randolph | https://www.ratemydorm.com/reviews/duke-university/duke-university-randolph |

## Chunking Strategy

<!-- Describe your chunking approach with enough specificity that someone else could reproduce it.
     Include:
     - Chunk size (characters or tokens) and why that size fits your documents
     - Overlap size and why (or why not) you used overlap
     - Any preprocessing you did before chunking (e.g., stripping HTML, removing headers)
     - What your final chunk count was across all documents -->

**Chunk size:**

One chunk ≈ one review. Each dorm's `.txt` file holds its reviews separated by a `---` delimiter line, and [ingest.py](src/ingest.py) splits on that delimiter so every review becomes its own chunk. Reviews range from ~20 to ~265 words, so most map cleanly to a single chunk. The one guardrail: `all-MiniLM-L6-v2` silently truncates anything past its 256-token limit, so any review exceeding that is sub-split on sentence boundaries (using the model's real tokenizer to measure, with a word-count heuristic as fallback).

**Overlap:**

Zero overlap *between* different reviews — they are distinct opinions with a clean `---` delimiter, so overlap would only blur the boundary between two people's views and hurt retrieval precision. The single exception is *within* a sub-split long review: each sub-chunk begins with the last sentence of the previous one (~1 sentence of overlap) so a sentence straddling the cut isn't lost from either embedding. Sub-chunks of the same review share one `review_id`.

**Why these choices fit your documents:**

The docs are full of short, opinionated reviews, not long-form writing. One or a few ideas per chunk gives high retrieval precision; a large fixed-size chunk would mix several reviewers' opinions into one vector and blur the signal. Because only 1 of 28 reviews is long enough to need sub-splitting, fixed-character chunking would needlessly fragment the other 27 clean reviews. Each chunk also carries `dorm_name` (from the filename) and `review_id` metadata, which the retrieval and generation stages use to scope by dorm and attribute sources.

**Final chunk count:**

29 chunks from 28 reviews across 10 dorm files (the extra chunk is the one Basset review that sub-split). If this were a production product I would obviously look for many more data points aka reviews and aim for many more chunks.

---

## Embedding Model

<!-- Name the embedding model you used and explain your choice.
     Then answer: if you were deploying this system for real users and cost wasn't a constraint,
     what tradeoffs would you weigh in choosing a different model?
     Consider: context length limits, multilingual support, accuracy on domain-specific text,
     latency, and local vs. API-hosted. -->

**Model used:**

`all-MiniLM-L6-v2` via `sentence-transformers`, producing 384-dim vectors stored in a local persistent ChromaDB (cosine space). The *same* model embeds both the indexed chunks ([embed.py](src/embed.py)) and the query at search time ([retrieve.py](src/retrieve.py)). Retrieval is semantic top-k (default k=5), with an optional `where={"dorm_name": ...}` metadata filter applied *before* ranking so a dorm-scoped query only competes against that dorm's reviews. It's a small, fast, free local model, well matched for short English reviews where each chunk's meaning is already compact.

**Production tradeoff reflection:**

If this were a production system for real users with cost off the table, I'd move to a larger hosted embedding model (e.g. an OpenAI or Voyage embedding API) for better accuracy on nuanced, domain-specific phrasing. The tradeoffs I'd weigh: **accuracy on domain text** - bigger models capture subtler distinctions between similar reviews; **latency and local vs. API** - MiniLM runs locally with no network round-trip, while a hosted model adds API latency and a dependency/cost per query; **context length** - not a real concern here since reviews are short, but it would matter for longer documents. **Multilingual support** doesn't really matter in this scenario since Duke is an English-speaking university. For short dorm reviews specifically, the embedding model matters less than it would for long technical text, since each chunk's context is already small, which is why MiniLM is a reasonable choice for this course project.

---

## Grounded Generation

<!-- Explain how your system enforces grounding — how does it prevent the LLM from answering
     beyond the retrieved documents?
     Describe both your system prompt (what instruction you gave the model) and any structural
     choices (e.g., how you formatted the context, whether you filtered low-relevance chunks).
     Do not just say "I told it to use the documents" — show the actual instruction or explain
     the mechanism. -->

**System prompt grounding instruction:**

Grounding is enforced through both the prompt and structural guards in [generate.py](src/generate.py), not left as a polite suggestion. The system prompt frames the rules as absolute and tells the model it has *no* knowledge outside the provided reviews:

> Use ONLY facts stated in the CONTEXT reviews. You have NO other knowledge about Duke, its dorms, or anything else. Do not add, infer, or "fill in" anything that is not explicitly in the CONTEXT, even if you think it is true or helpful. If the CONTEXT does not contain enough information to answer the question, reply with EXACTLY this sentence and nothing else: "I don't have enough information on that."

The prompt also requires the model to cite the bracketed review tag for every claim and to calibrate confidence to the number of supporting reviewers ("one reviewer mentioned…" vs. "multiple reviewers said…"). Also **if retrieval returns zero chunks, the system short-circuits to the no-information answer and never calls the LLM at all**. With no context, the model could only hallucinate, so we don't give it the chance.

**How source attribution is surfaced in the response:**

Source attribution is guaranteed **programmatically**, not trusted to the LLM. `ask()` returns `{"answer": str, "sources": list[str]}` where the `sources` list is built directly from the retrieved chunks' metadata (`dorm_name` + `review_id`), deduplicated by `review_id`. So even if the model forgot to cite inline, the response object still names exactly which review documents were retrieved. The list is also set to empty whenever the answer is the no-information phrase, so a returned source list can never name a document the answer didn't actually use. The Gradio UI ([app.py](src/app.py)) renders this as a separate "Retrieved from" box beneath the answer.

---

## Evaluation Report

<!-- Run your 5 test questions from planning.md through your system and record the results.
     Be honest — a partially accurate or inaccurate result that you explain well is more
     valuable than a suspiciously perfect result. -->

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What are the room sizes like in Randolph? | Rooms reported as small but creating good community | "One reviewer mentioned the rooms are small (randolph-2); another stated they're small but noted this makes the community better (randolph-1)." | Relevant (both Randolph reviews retrieved, top distance 0.405) | Accurate |
| 2 | Does Basset have a mold problem? | Some users reported mold in the rooms | "One reviewer mentioned mold forced a couple of people to move out mid-semester (basset-3); another mentioned having mold in their room (basset-4). At least two reviewers said Basset has a mold problem." | Relevant (all Basset reviews retrieved, incl. both sub-chunks of basset-3) | Accurate |
| 3 | How are the bathrooms in Wilson? | Suite-style bathrooms, which users liked, but they had to clean after themselves | "Suite-style bathroom is 'so nice' and private (wilson-3, wilson-2), but requires suitemates to clean it; two reviewers had negative experiences ('very small and gross', clogged) (wilson-4, wilson-2). Multiple reviewers agreed cleaning falls to the suitemates (wilson-1, wilson-2, wilson-4)." | Relevant (all 4 Wilson reviews retrieved) | Accurate (richer than expected — also surfaced the negative experiences) |
| 4 | Is it easy to make friends in Giles dorm? | Community is better than other dorms and built quickly | "Multiple reviewers describe a great, 'tight-knit' community (giles-1, giles-2); one noted the dorm's small size helps build community quickly (giles-3), suggesting it's easy to make friends." | Relevant (all 3 Giles reviews retrieved, top distance 0.381) | Accurate |
| 5 | What are the rooms like in Keohane quad? | Medium-sized/spacious rooms with large windows letting in natural light | "One reviewer said rooms are 'large' (keohane-2), another 'medium sized' (keohane-1), a third 'pretty spacey' (keohane-3)." | Relevant (all 3 Keohane reviews retrieved; keohane-2 — which contains the natural-light detail — ranked #1 at distance 0.385) | Partially accurate (correct on size, but omitted the "large windows / natural light" detail that was present in the retrieved text) |

**Retrieval quality:** Relevant / Partially relevant / Off-target  
**Response accuracy:** Accurate / Partially accurate / Inaccurate

**Summary:** 4 of 5 accurate, 1 partially accurate. Retrieval was relevant on all 5 and every question pulled the right dorm's reviews. The one shortfall (Keohane) was a *generation* issue, not retrieval, and is analyzed below.

---

## Failure Case Analysis

<!-- Identify at least one question where retrieval or generation did not work as expected.
     Write a specific explanation of *why* it failed, tied to a part of the pipeline.

     "The answer was wrong" is not an explanation.

     "The relevant information was split across a chunk boundary, so retrieval returned
     only half the context — the model didn't have enough to answer correctly" is an explanation.

     "The embedding model treated the professor's nickname as out-of-vocabulary and returned
     results from an unrelated review" is an explanation. -->

**Question that failed:**

"What are the rooms like in Keohane quad?" (Question 5) — partially accurate.

**What the system returned:**

"One reviewer mentioned that the rooms in Keohane are 'large' (source: keohane-2), while another reviewer described them as 'medium sized' (source: keohane-1). A third reviewer said that the rooms in Keohane 4B are 'pretty spacey' (source: keohane-3)." The expected answer also included that the rooms have **large windows that let in a lot of natural light** and the system omitted this entirely.

**Root cause (tied to a specific pipeline stage):**

This is a **generation (LLM summarization) failure, not a retrieval failure**. The chunk containing the missing detail, `keohane-2`, was retrieved and ranked **#1** (cosine distance 0.385); its text reads "…has large rooms… Large windows to let in a lot of natural light and the views on the second floor and above can be quite nice." So the relevant information *was* in the context window handed to the model. The model interpreted "what are the rooms *like*" narrowly as **room size** and extracted only the explicit size adjectives ("large", "medium sized", "spacey"), dropping the windows/natural-light attribute that lives in the same sentence cluster as "large rooms." The grounding worked (everything it said is sourced and true); the model just under-extracted relevant attributes from a chunk it correctly retrieved.

**What you would change to fix it:**

Two options, in order of effort. (1) **Prompt change**: instruct the model to extract *all* attributes a chunk mentions about the subject (size, light, amenities, AC, location), not just the most direct match to the question's keyword — i.e. answer the spirit of "what are the rooms like," not just "how big are the rooms." This is the cheapest fix and targets the exact behavior. (2) If coverage still lagged, raise `top-k` slightly and/or add a brief "list every distinct point the reviews make about X" intermediate step. I'd try the prompt fix first since retrieval is already returning the right chunk and adding more chunks wouldn't help when the model is ignoring detail in the chunk it already has.

---

## Spec Reflection

<!-- Reflect on how planning.md shaped your implementation.
     Answer both questions with at least 2–3 sentences each. -->

**One way the spec helped you during implementation:**

Developing the spec in stages was what kept the AI-generated code on the right track. Because I'd planned each pipeline stage in detail before writing any code — chunking, then embedding/retrieval, then generation — I could prompt the AI one milestone at a time and actually verify that its output matched what I'd specified before moving on to the next. For example, I'd already written down that `all-MiniLM-L6-v2` *silently truncates* past 256 tokens, so when I had the AI implement chunking I already knew what "correct" looked like: measure each review with the model's tokenizer and sub-split only the one review that exceeded the limit (`basset-3`), sharing a `review_id`. That let me confirm it was right immediately by printing the chunk counts, instead of discovering a silently dropped Basset review much later as a confusing retrieval bug. Testing and verifying each stage against the spec before building the next one meant most bugs got caught right where they were introduced rather than compounding downstream. Overall, planning the spec out in detail beforehand made prompting the AI to implement the actual code a whole lot faster and easier — I was handing it concrete, verifiable requirements instead of vague descriptions, so it got things right the first time far more often.

**One way your implementation diverged from the spec, and why:**

The Architecture diagram in planning.md labels the Generation stage as using **Claude**, but the implementation uses **Groq's `llama-3.3-70b-versatile`** instead. I diverged because the Milestone 5 instructions recommended Groq specifically — it's free-tier and OpenAI-compatible, which removed the cost and billing-setup friction of a paid Claude key for a course project. The grounding design is provider-agnostic (system prompt + structural guards in `generate.py`), so swapping the LLM didn't change the architecture — only the client library and model name. A second, smaller divergence: planning.md described source attribution loosely (cite sources in the response), but I tightened it to a **programmatic guarantee** — sources are derived from retrieved chunk metadata rather than parsed out of the model's text — because relying on the LLM to remember to cite is exactly the kind of grounding gap the project warns against.

---

## AI Usage

<!-- Describe at least 2 specific instances where you used an AI tool during this project.
     For each: what did you give the AI as input, what did it produce, and what did you
     change, override, or direct differently?

     "I used Claude to help me code" is not sufficient.
     "I gave Claude my Chunking Strategy section from planning.md and asked it to implement
     chunk_text(). It returned a function using a fixed character split. I overrode the
     chunk size from 500 to 200 because my documents are short reviews, not long guides." -->

**Instance 1 — Grounded generation + Gradio UI (Milestone 5)**

- *What I gave the AI:* My existing `retrieve()` function, the Milestone 5 spec from planning.md, and a generation prompt spec — answer *only* from retrieved reviews, cite which review each claim came from, hedge when only one reviewer supports a claim, and use Groq's `llama-3.3-70b-versatile`. I explicitly asked it to make the system prompt *enforce* grounding rather than suggest it, and to guarantee source attribution programmatically rather than trusting the LLM to add it.
- *What it produced:* `generate.py` (an `ask()` function that retrieves chunks, builds a tagged context block, calls Groq, and returns `{answer, sources}`) and `app.py` (the Gradio interface). It built the sources list from chunk metadata, added a strict "you have NO other knowledge" system prompt, and included a zero-chunk short-circuit that skips the LLM entirely when retrieval is empty.
- *What I changed or overrode:* I directed the design choice that source attribution must be *programmatic* (derived from retrieved metadata and deduped by `review_id`), not parsed from the model's text — and that the source list must be force-emptied on the "I don't have enough information" answer so it never names an unused document. I also kept all RAG logic in `generate.ask()` and limited `app.py` to presentation, so the pipeline stays testable without the UI.

**Instance 2 — Resolving the dependency conflict from installing Gradio**

- *What I gave the AI:* After adding `gradio>=6.9.0` to requirements and installing it, I asked the AI to verify the whole pipeline still ran end-to-end before I trusted it.
- *What it produced:* It predicted (and then confirmed by running the code) that installing Gradio had upgraded `huggingface-hub` to 1.x, which the pinned `transformers 4.57` / `sentence-transformers 3.4.1` reject — silently breaking the embedding/retrieval import with an `ImportError`. It resolved the conflict by upgrading `transformers` → 5.12 and `sentence-transformers` → 5.5.1, then re-ran `pip check` (no broken requirements) and the 5-question evaluation to confirm retrieval still worked.
- *What I changed or overrode:* Rather than let it blindly install Gradio and move on, I directed it to *verify behavior after the install* — which is what caught the conflict. I also had it flag that `requirements.txt` still pins the old `sentence-transformers==3.4.1`, so a fresh install would reintroduce the conflict, leaving me to decide whether to bump the pins.
