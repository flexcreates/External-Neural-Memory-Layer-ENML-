# ENML Core Architectural Process

This directory houses the foundational backbone of the **Infinite Learning v2.0** engine. The Cognitive Core is split into separate independent processing nodes:

## 1. Fact Extraction (`memory/extractor.py`)
Before the Orchestrator generates a response, it intercepts the user's input and fires a hidden, small-context LLaMA 3 inference loop. This loop acts strictly as a data miner under the following constraints:
- Extract factual data (Identities, system setups, relationships).
- Refuse to extract conversational pleasantries or temporary context.
- Format the output exclusively as a set of Semantic Triples: `[{"subject": ..., "predicate": ..., "object": ...}]`

## 2. Triple Storage Layer (`memory/triple_memory.py`)
Extracted facts are piped into the `MemoryManager`. Those exceeding the Confidence Threshold (≥ 0.75) are converted into `MemoryTriple` dataclasses. 
Instead of relying on rigid internal python dictionaries (`{"age": 35}`), properties are represented purely semantically (`subject: "user", predicate: "has_age", object: "35"`).

## 3. Qdrant Ingestion & Retrieval (`vector/qdrant_client.py`)
Every Triple object is passed into the `embedding_service` using `all-MiniLM-L6-v2`. The payload string `"{subject} {predicate} {object}."` is mapped across 384 dimensions and committed to the `knowledge_collection`.

When a user submits a query, the `QueryRouter` flags intents starting with *"My [X]"* or *"What is..."* and directly searches the `knowledge_collection`. It pulls the Top-K Triples.

## 4. Grounded Context Building (`context_builder.py`)
Finally, rather than merely appending previous conversations, the Context Builder constructs a strict foundational System Prompt block:

```markdown
Relevant Known Facts:
- user has_sister Anna.
- user has_hobby paragliding.

Only answer using the provided facts where applicable. If no relevant fact exists, say you don't know.
```
This multi-layered approach guarantees that:
a) The AI never "guesses" or hallucinates parameters regarding you.
b) The database scales infinitely without outgrowing the LLM's finite context window.
