# Where RAG is useful

RAG fits any situation where the answer lives in a body of text that the model was not trained on, changes over time, or needs to be cited. Each section below describes a use case, what you would index, and which raglab settings are a sensible starting point.

## Customer support assistant

The sample data in `examples/data` is a small help center for a fictional app. A support assistant indexes help articles, policy pages and past resolved tickets, then answers customer questions with links to the article it used.

* Index: help center articles, refund and shipping policies, release notes.
* Retriever: hybrid, because customers paraphrase ("money back") but also type exact product names and error codes.
* Why RAG helps: when a policy changes, you edit the article. No retraining, and every answer points to the page an agent can verify.

```bash
raglab ask "Can I get a refund for an App Store purchase?" --data examples/data
```

## Internal knowledge base

Teams accumulate answers in wikis, runbooks and meeting notes that nobody can find later. A RAG assistant over those files answers "how do we rotate the database credentials?" with the steps from the runbook and a pointer to it.

* Index: exported wiki pages, runbooks, architecture decision records.
* Retriever: hybrid. Internal jargon and service names favor BM25.
* Watch out for: access control. Only retrieve documents the person asking is allowed to read.

## Policy, legal and compliance questions

Contracts, regulations and HR handbooks are long, precise and full of cross references. RAG lets someone ask "what is the notice period for contractors?" and get the exact clause back.

* Index: the documents split into small chunks so each clause stands alone.
* Retriever: BM25 or hybrid, since exact terms carry legal meaning.
* Settings: lower `max_chars` (around 300) and keep `overlap` at 1 or more so clauses are not cut.
* Why RAG helps: the extractive generator returns the original wording with a citation, which is often what a reviewer wants.

## Study and research assistant

Students and researchers can index their own notes, papers and lecture transcripts and then ask questions across all of them.

* Index: Markdown notes, text exports of papers, transcripts.
* Retriever: dense with a neural embedding model, because questions are often conceptual and worded differently from the notes.
* Why RAG helps: answers come from your own material, with the source file named, so you can go back and reread the context.

## Developer documentation

Indexing a project's docs, README files and API references gives developers an assistant that knows the current version of the library rather than whatever version the model saw in training.

* Index: docs folders, docstrings extracted to text, changelogs.
* Retriever: hybrid. Function and class names are exact tokens that BM25 handles well.

## Product catalog search

An online store can index product descriptions and spec sheets so shoppers can ask "which laptops under 1.5 kg have 32 GB of RAM?" and get grounded answers instead of a keyword results page.

* Index: one chunk per product, with structured fields written out as text.
* Note: numeric filters such as price or weight are better handled by the database before retrieval. RAG is strongest on the free text part of the question.

## When RAG is the wrong tool

* The task needs reasoning over the whole corpus at once, such as "summarize every ticket from last year". Retrieval returns a handful of chunks, not everything.
* The answer requires live data, such as an account balance. Call an API or database instead, or combine RAG with tool calling.
* The goal is a new style or behavior rather than new facts. That is a job for prompting or fine tuning.
* The documents are tiny. If everything fits comfortably in the prompt, skip retrieval and paste it all in.
