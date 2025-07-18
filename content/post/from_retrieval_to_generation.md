---
author: "Anson Wang"
date: "2025-07-16"
description: "From Retrieval to Generation: The Technical Evolution and Reflection of Semantic Search"
title: "From Retrieval to Generation"
tags: [
    "LLM"
]
categories: ["LLM"]
math: true
---





## From Retrieval to Generation
---

### Introduction: Rethinking Human Needs Behind the Search Box

When we type into the search box, we no longer simply look for keywords. We expect systems to understand our intent, context, and goals. This shift—from "find" to "synthesize"—marks the starting point of integrating semantic search with generative models.

This article is divided into three parts:
1. Beginner Tutorial: From TF-IDF to Dense Vector Search
2. Technical Details: Internal Mechanics of BM25, Dense Retriever, and FiD
3. Philosophical Perspective: What It Means When Retrieval Merges with Generation

---

#### 1. Why Semantic Search?

- Example: When searching for "apple", do we mean the fruit or the company?
- Traditional keyword-based IR cannot handle polysemy or complex context.

#### 2. TF-IDF and BM25

- TF-IDF ranks documents based on term frequency and inverse document frequency.
- BM25 improves TF-IDF by applying term frequency saturation and document length normalization.

#### 3. Dense Retriever

- Uses pretrained language models (e.g., BERT) to convert sentences into dense vectors.
- Searches using cosine similarity or dot product in the embedding space.

---

### II. Technical Details: Internal Mechanics and Formulas of BM25, Dense Retriever, and FiD

##### **BM25**

##### 1. Full Scoring Formula

$$
\mathrm{score}(D,Q) = \sum_{q_i \in Q} \mathrm{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\mathrm{avgdl}})}
$$

- $f(q_i, D)$: frequency of query term $q_i$ in document $D$
- $|D|$: document length (token count)
- $\mathrm{avgdl}$: average document length
- $k_1, b$: hyperparameters (typically 1.2–2.0 and 0.75)

##### 2. IDF Formula

$$
\mathrm{IDF}(q_i) = \log\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5} + 1\right)
$$

- $N$: total number of documents
- $n(q_i)$: number of documents containing term $q_i$

##### 3. Meaning of $k_1$ and $b$

- $k_1$: controls term frequency saturation
- $b$: controls the degree of document length normalization

#### **Dense Retriever**

#### 1. Encoder Architecture

- Dual Encoder (bi-encoder):
  - Query encoder: $f(q) = \mathrm{BERT}(q) \rightarrow \vec{q}$
  - Document encoder: $f(d) = \mathrm{BERT}(d) \rightarrow \vec{d}$

#### 2. Similarity Function

- Cosine similarity:  
  $$
  \mathrm{sim}(q, d) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\| \|\vec{d}\|}
  $$

#### 3. Index Structure

- FAISS: Flat, IVF, HNSW
- Efficient top-k search on prebuilt vector index

#### 4. Loss Functions and Training

- Contrastive Loss / InfoNCE
- Typical setup: positive pairs $(q, d^+)$ and hard negatives $(d^-)$

### **FiD (Fusion-in-Decoder)**

#### 1. Architecture Overview

- Unlike bi-encoder RAG, FiD encodes all retrieved documents independently but **fuses them at decoding time**.
- Each document $d_i$ is passed into a shared encoder along with the query $q$: $\mathrm{Enc}(q, d_i)$
- All encoder outputs are concatenated and passed to a decoder jointly:

$$
\mathrm{Dec}([\mathrm{Enc}(q, d_1); \dots; \mathrm{Enc}(q, d_k)])
$$

#### 2. Key Characteristics

- Uses **cross-attention** from decoder to all documents simultaneously
- Allows full access to all evidence during generation, unlike RAG which conditions generation on a single document at a time

#### 3. Trade-offs

- **Better performance** when synthesis across multiple retrieved contexts is important
- **More computationally expensive** due to cross-attending to many inputs during decoding

#### 4. Use Cases

- Open-domain QA with long-form answers
- Multi-hop reasoning across distributed information

---

## III. Philosophical Perspective: Retrieval is Generation?

### 1. Why Has Search Become Generation?

- It’s no longer about matching data, but **combining relevant context with the query** to generate answers aligned with user expectations.
- Users want **conclusions**, not just **lists**. Thus, search results increasingly resemble summaries, answers, or synthesis.
- Systems like RAG and FiD turn retrieval into a semantic organizer and viewpoint composer.

### 2. Is RAG Retrieval? Generation? Or Both?

- Retriever becomes a "dynamic knowledge fetcher"
- LLM performs reasoning and synthesis over retrieved context

### 3. Prompts are No Longer Keywords, but Intent and Perspective

- Search is no longer keyword matching. It's about **interpreting prompts** and **assembling knowledge** accordingly.

---

## Conclusion: Search is Becoming a Tool for Thinking

True search is not about finding what we already know—it’s about revealing what we didn’t know we were looking for.



---

**References**:  
- [DPR Paper](https://arxiv.org/abs/2004.04906) | [FiD Paper](https://arxiv.org/abs/2007.01282)  
- Code: [DPR GitHub](https://github.com/facebookresearch/DPR), [FiD GitHub](https://github.com/facebookresearch/FiD)  

