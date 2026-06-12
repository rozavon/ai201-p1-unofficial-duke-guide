# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->
I chose dorm reviews at Duke University. This knowledge is valuable because as a freshman coming into Duke I had no visibility into what the dorms are like. Official channels don't give the real scoop like student reviews do.

---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | Ratemydorm.com| 4 reviews for Basset dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-bassett |
| 2 | Ratemydorm.com| 4 reviews for Wilson dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-wilson |
| 3 | Ratemydorm.com| 4 reviews for Giles dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-giles |
| 4 | Ratemydorm.com| 3 reviews for Keohane quad | https://www.ratemydorm.com/reviews/duke-university/duke-university-keohane-quad |
| 5 | Ratemydorm.com| 3 reviews for Trinity dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-trinity |
| 6 | Ratemydorm.com| 3 reviews for Wannemaker quad | https://www.ratemydorm.com/reviews/duke-university/duke-university-wannamaker-quad |
| 7 | Ratemydorm.com| 2 reviews for Blackwell dorm | https://www.ratemydorm.com/reviews/duke-university/duke-university-blackwell |
| 8 | Ratemydorm.com| 2 reviews for Craven quad | https://www.ratemydorm.com/reviews/duke-university/duke-university-craven-quad |
| 9 | Ratemydorm.com| 2 reviews for Kilgo quad | https://www.ratemydorm.com/reviews/duke-university/duke-university-kilgo-quad |
| 10 |Ratemydorm.com| 2 reviews for Randolph | https://www.ratemydorm.com/reviews/duke-university/duke-university-randolph |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size:**
Small chunks (≈1 review each) 
**Overlap:**
We want little to no overlap
**Reasoning:**
We want to use small chunks because the documents do not contain long form prose but short opinionated reviews. Therefore, we want a single/few ideas in each chunk. Having large chunks could contain too many topics a lower precision. We also want little to no overlap, since most reviews are distinct and about different things, and there is a clear delimiter between reviews, so overlap might actually hurt in this case.

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs would you weigh in choosing a different embedding model — context length, multilingual support, accuracy on domain-specific text, latency? -->
Retrieval will mainly be semantic search (good for paraphrased opinions), scoped by dorm via metadata filtering (i.e. adding dorm_name metadata to each chunk/document since documents are separated by dorm). Hybrid retrieval with a mix of lexical and semantic approach and reranking are known upgrades I would add if exact-term queries became an issue with real users, but for now I'll stick with the basics

**Embedding model:**
all-MiniLM-L6-v2
**Top-k:**
k=5 (should probably test, trial and error)
**Production tradeoff reflection:**
If this was a production project for real user without cost as a constraint, I would definitely choose a higher dim model that's paid. all-MiniLm is perfectly fine for what we are doing in this course though. Also for short reviews like the ones about Duke dorms, embedding model doesn't matter too much since the context of each chunk is already pretty small. Multilingual support also wouldn't matter much since this is an English speaking university and all students need to speak English to go to Duke. 
---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What are the room sizes like in Randolph? | Users reported the rooms as small but creating good community |
| 2 | Does Basset have a mold problem? | Some users reported mold in the rooms in Basset |
| 3 | How are the bathrooms in Wilson? | The bathrooms are suite style which users reported liking but also that they had to clean after themselves |
| 4 | Is it easy to make friends in Giles dorm? | Users report that the community in Giles in better than other dorms and that community is built quickly |
| 5 | What are the rooms like in Keohane quad? | Users report medium-sized spacious rooms with large windows letting in natural light. |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. Students will inevitably ask "Is Blackwell or Trinity quieter?" That requires retrieving and comparing across two documents. A naive top-k might return 4 Blackwell chunks and 0 Trinity ones, giving a one-sided answer. 

Mitigation: detect multi-dorm queries and retrieve top-k per dorm, then let the LLM compare.

2. Lack of data per document. I have only 3–4 reviews per dorm. If 2 of 3 Blackwell reviewers happened to hate the AC, the guide will confidently report "Blackwell has bad AC", when that's really two people, possibly from the same hot summer.

Mitigation: surface how many reviews support a claim, and/or have the LLM hedge ("one reviewer mentioned…" vs "multiple reviewers said…").

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

```mermaid
flowchart TD
    subgraph INDEX["🗂️ Indexing (run once, ahead of time)"]
        direction TB
        A["📄 Document Ingestion<br/>one .txt file per dorm<br/>(Blackwell, Trinity, ...)<br/><i>Python file reader</i>"]
        B["✂️ Chunking<br/>1 chunk = 1 review<br/>+ attach dorm_name, review_id<br/><i>custom split-by-review</i>"]
        C["🧮 Embedding<br/>text → 384-dim vector<br/><i>all-MiniLM-L6-v2<br/>(sentence-transformers)</i>"]
        D[("🗄️ Vector Store<br/>vectors + text + metadata<br/><i>Chroma</i>")]
        A --> B --> C --> D
    end

    subgraph QUERY["💬 Querying (every user question)"]
        direction TB
        Q["❓ User Question<br/>'Is Blackwell quiet?'"]
        QE["🧮 Embed Query<br/><i>same MiniLM model</i>"]
        R["🔍 Retrieval<br/>filter by dorm_name,<br/>then top-k similar chunks<br/><i>Chroma similarity search</i>"]
        G["🤖 Generation<br/>question + retrieved reviews<br/>→ grounded answer w/ citations<br/><i>Claude</i>"]
        ANS["✅ Answer"]
        Q --> QE --> R --> G --> ANS
    end

    D -.->|"searched at query time"| R

    style INDEX fill:#eef6ff,stroke:#3b82f6
    style QUERY fill:#f0fdf4,stroke:#22c55e
    style D fill:#fff7ed,stroke:#f59e0b
```

---

## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**

**Milestone 4 — Embedding and retrieval:**

**Milestone 5 — Generation and interface:**
