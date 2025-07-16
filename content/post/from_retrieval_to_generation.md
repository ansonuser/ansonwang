---
author: "Anson Wang"
date: "2025-07-16"
description: "Technologies and Conflict Handling with Practical Examples"
title: "From Retrieval to Generation"
tags: [
    "LLM"
]
categories: ["LLM"]
math: true
---





## Table of Contents  
1. [Retrieval Technologies](#1-retrieval-technologies)  
   - [BM25](#11-bm25-keyword-based-retrieval)  
   - [DPR](#12-dpr-dense-retrieval)  
   - [Hybrid Search](#13-hybrid-search)  
2. [Generation & Consistency Technologies](#2-generation--consistency-technologies)  
   - [REALM](#21-realm-consistency-checking)  
   - [FiD](#22-fid-fusion-in-decoder)  
3. [Technology Comparison](#3-technology-comparison)  
4. [Advanced Scenarios](#4-advanced-scenarios)  

---

## 1. Retrieval Technologies  

### 1.1 BM25 (Keyword-Based Retrieval)  
**Formula**:  
\[
\text{BM25}(Q, D) = \sum_{q \in Q} \text{IDF}(q) \cdot \frac{f(q, D) \cdot (k_1 + 1)}{f(q, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{\text{avgdl}})}
\]  

**Example**:  
- **Query**: `"How to prevent colds?"`  
- **Document**: `"Frequent handwashing, wearing masks, and vitamin C supplementation reduce cold risks."`  
- **Calculation**:  
  - Terms: `["prevent", "colds"]`  
  - TF for "colds": 1  
  - IDF (hypothetical): `IDF("prevent")=1.2`, `IDF("colds")=0.8`  
  - Score: **≈0.67**  

---

### 1.2 DPR (Dense Passage Retriever)  
**Workflow**:  
1. **Encoding**:  
   - Query: `q = \text{BERT}("When did Apollo 11 land on the moon?")`  
   - Documents:  
     - `d_1 = \text{BERT}("Apollo 11 landed on July 20, 1969...")`  
     - `d_2 = \text{BERT}("The lunar program began in 1961...")`  
2. **Similarity**:  
   - `sim(q, d_1) = 0.92` → **Top-1 Result**  

---

### 1.3 Hybrid Search  
**Score Fusion**:  
\[
\text{HybridScore} = \alpha \cdot \text{BM25} + (1-\alpha) \cdot \text{DPR}
\]  

**Example**:  
- BM25 Scores: `[0.8, 0.6, 0.5]`  
- DPR Scores: `[0.7, 0.9, 0.3]`  
- Weight: `α = 0.4`  
- **Final Scores**:  
  - Doc1: `0.74`, Doc2: `0.78`, Doc3: `0.38`  
  - **Ranking**: Doc2 > Doc1 > Doc3  

---

## 2. Generation & Consistency Technologies  

### 2.1 REALM Consistency Checking  
**Training Task (MLM)**:  
- Masked Sentence: `"The [MASK] host of COVID-19 is bats."`  
- Retrieved Document: `"Coronaviruses likely originated in bats."`  
- Model Prediction: `"virus"`  

**Inference Example**:  
- Query: `"What is the natural host of COVID-19?"`  
- Retrieved Doc: `"Coronaviruses are found in bats."`  
- **Answer**: `"Bats"`  

---

### 2.2 FiD (Fusion-in-Decoder)  
**Multi-Document Fusion Example**:  
- **Query**: `"Why is the sky blue?"`  
- **Retrieved Docs**:  
  1. `"Rayleigh scattering disperses blue light (short wavelength)."`  
  2. `"Human eyes are more sensitive to blue light."`  
  3. `"Clouds appear white due to Mie scattering."`  
- **Generated Answer**:  
  `"Blue light scatters more in the atmosphere (Rayleigh scattering), and human eyes are sensitive to it."`  

**Conflict Handling**:  
- **Query**: `"Earth's age?"`  
- **Docs**:  
  1. `"Radiometric dating: ~4.54 billion years."`  
  2. `"Religious texts claim 6,000 years."`  
- **FiD Output**:  
  `"Scientific consensus estimates 4.54 billion years with ±1% error."` *(ignores conflicting doc)*  

---

## 3. Technology Comparison  

| Technology | Retrieval Method       | Generation           | Conflict Handling              |  
|------------|------------------------|----------------------|---------------------------------|  
| BM25       | Keyword Matching       | None                 | ❌ No                          |  
| DPR        | Dense Vector           | None                 | ❌ No                          |  
| REALM      | Dense Retrieval        | BERT Generation      | ✅ Relies on retrieval quality |  
| FiD        | Any Retriever          | Multi-Doc Fusion     | ✅ Attention-weighted fusion   |  

---

## 4. Advanced Scenarios  

### 4.1 Multimodal FiD  
- **Query**: `"What is the 2023 revenue growth rate based on this line chart?"`  
- **Inputs**:  
  - Text: `"Annual report: 12% growth in 2023."`  
  - Chart: Line graph showing 12% growth  
- **Output**: `"Integrated data shows 12% revenue growth in 2023."`  

### 4.2 Real-Time Retrieval  
- **Query**: `"What is TSMC's stock price today?"`  
- **Process**:  
  1. Fetch real-time data: `"TSMC (2330): $598 ↑1.2%"`  
  2. Generate: `"TSMC closed at $598 today, up 1.2%."`  

---

**References**:  
- [DPR Paper](https://arxiv.org/abs/2004.04906) | [FiD Paper](https://arxiv.org/abs/2007.01282)  
- Code: [DPR GitHub](https://github.com/facebookresearch/DPR), [FiD GitHub](https://github.com/facebookresearch/FiD)  

- Blog: [How to Build an Open-Domain Question Answering System?](https://lilianweng.github.io/posts/2020-10-29-odqa/)