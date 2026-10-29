# NLP Evolution Lab: From Bag-of-Words to Transformers
**Generative AI • Module 1 Academic Laboratory Project**

An interactive educational website and computational laboratory designed to demonstrate how natural language representation evolved across three paradigm shifts:
1. **Bag of Words / TF-IDF** (Sparse Lexical Counts & Frequency)
2. **Dense Word Embeddings** (Continuous Distributed Vectors & Geometry)
3. **Transformers & Self-Attention** (Contextualized Bidirectional Multi-Head Attention)

---

## ⚡ Vercel Deployment Guide

This project is pre-configured and optimized to deploy seamlessly to **[Vercel](https://vercel.com)** within the free tier serverless limits (cold start < 50ms, bundle size < 2MB).

### Option A: Deploy with Vercel CLI (Fastest)

1. **Install Vercel CLI** (if you don't have it):
   ```bash
   npm install -g vercel
   ```

2. **Deploy to Preview**:
   ```bash
   vercel
   ```
   *Follow the on-screen prompts: accept default settings (no build command needed).*

3. **Deploy to Production**:
   ```bash
   vercel --prod
   ```

---

### Option B: Deploy via GitHub (Recommended for Teams)

1. Push this folder to a GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "feat: NLP Evolution Lab Vercel Ready"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo-name>.git
   git push -u origin main
   ```

2. Go to [Vercel Dashboard](https://vercel.com/new).
3. Click **"Import Project"** and select your GitHub repository.
4. Leave all settings at their defaults:
   - **Framework Preset**: `Other`
   - **Root Directory**: `./`
   - **Build Command**: *(leave blank)*
   - **Output Directory**: *(leave blank)*
5. Click **"Deploy"**. Your site will be live on an official `https://<project-name>.vercel.app` URL in under 60 seconds!

---

## 🏛️ Project Architecture & Design Overview

- **Design Philosophy**: Modern, accessible light dashboard with slate typography and teal accents (`#0d9488`, `#0f766e`, `#f0fdfa`).
- **Engine**: 100% genuine local computations with Scikit-Learn and NumPy-accelerated Multi-Head Attention. Real forward-pass tensor operations with exact measured inference latency and attention weight matrices.
- **Strict Separation**: Dedicated, independent Training corpus vs. Held-Out Evaluation Benchmark.

---

## 🚀 Key Laboratory Features

### 1. 🧪 Comparative Text Playground
- Enter any English sentence or select from challenging presets (positive sentiment, negative sentiment, negation challenges like *"not bad at all"*, expectation flips, and contrastive clauses).
- Side-by-side comparative inspection of:
  - **TF-IDF + Logistic Regression**: Displays predicted polarity, calibrated logistic probability, microsecond latency, and active n-gram token weights.
  - **Dense Word Embeddings**: Continuous Bag of Words (CBOW) centroid classification, 50D vector preview, and in-vocabulary coverage percentage.
  - **Self-Attention Transformer**: Softmax confidence, bidirectional contextualized tokens, and genuine Multi-Head attention routing.
- **Educational Score Calibration Notice**: Prominently warns students that model scores originate from mathematically distinct objective functions (calibrated logistic probability vs. softmax confidence) and cannot be directly equated.
- **Robustness**: Handles empty input, oversized input (>600 characters), loading skeletons, and network failure states.

### 2. 🔍 Word Similarity Analyzer
- Computes exact **Cosine Similarity** between 50-dimensional semantic vectors:
  $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- **The Distributional Hypothesis & Antonym Paradox**: Explains why opposites such as *"hot"* and *"cold"* or *"good"* and *"bad"* have high cosine similarity (because they share identical grammatical frames in natural corpora: *"the coffee was too [hot/cold] to drink"*).
- **Out-of-Vocabulary (OOV) Handling**: Clear warning messages for unrecognized words, explaining the polysemy and OOV limitations of static embeddings.

### 3. 🧠 Token-to-Token Attention Explorer
- Interactive 2D self-attention heatmap visualization:
  $$A = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right)$$
- Interactive Layer and Head selector (inspect local syntactic heads vs. global [CLS] aggregation heads).
- **Dual Mode Toggle**:
  - **Live Computed Attention**: Actual softmax tensor values extracted directly from the model's forward pass.
  - **Illustrative Demo Mode**: Curated, annotated classroom matrix with pedagogical commentary.
- **Scientific Caveat**: Clearly highlights that while attention shows where intermediate representations pooled information, attention weights do not constitute an unambiguous causal explanation of the final decision (*Jain & Wallace, 2019*).

### 4. 📊 Results & Held-Out Benchmark Suite
- Evaluates all three models on 20 held-out test sentences across four linguistic categories:
  1. *Standard Positive*
  2. *Standard Negative*
  3. *Negation Challenges* (e.g. *"This film was not bad at all"*, *"Hardly the masterpiece promised"*)
  4. *Contrastive Clauses* (e.g. *"Stunning visuals, but the story is completely hollow"*)
- Real performance metrics: Accuracy (%), correct counts, error counts, and mean inference latency (ms).
- Filterable error analysis table showing where older models fail and where Transformers succeed.
- **CSV Export**: Direct one-click download of the complete benchmark dataset (`nlp_evolution_lab_benchmark.csv`).
- **Methodological Disclaimer**: Explains why a small test set cannot establish general statistical model superiority, but effectively illustrates qualitative linguistic failure modes.

### 5. 📚 Module 1 Educational Guide & Presentation Scenarios
- Beginner-friendly explanations of Bag of Words, TF-IDF, GloVe/Word2Vec, and Transformers.
- 4 Team Contribution cards for academic project grading.
- 3 Scripted Presentation Demo Scenarios for defending the project in front of faculty.

---

## 🎬 3 Scripted Presentation Demo Scenarios

### Scenario 1: The Negation Paradox
- **Sentence**: `"This movie was not bad at all, in fact I quite liked it."`
- **What to demonstrate**:
  1. Load sentence in Text Playground.
  2. **TF-IDF**: Misclassifies as Negative (or low confidence) because it sees the high-magnitude token `"bad"` without syntax.
  3. **Word Embeddings**: Vector averaging pulls the sentence centroid toward negative polarity.
  4. **Transformer**: Correctly predicts **Positive** because self-attention heads link `"not"` to `"bad"`.
  5. Open Attention Explorer to show Head 2 attending directly between `"not"` and `"bad"`.

### Scenario 2: The Distributional Hypothesis
- **Word Pair**: `"hot"` vs `"cold"`
- **What to demonstrate**:
  1. Navigate to Word Similarity panel.
  2. Observe high cosine similarity (~0.62) despite them having opposite definitions.
  3. Cite J.R. Firth (1957): *"You shall know a word by the company it keeps."* Antonyms appear in interchangeable grammatical positions, proving why static embeddings capture topical relatedness rather than true semantic polarity.

### Scenario 3: Contrastive Clauses & Discourse Markers
- **Sentence**: `"The movie has stunning visuals, but the story is completely hollow."`
- **What to demonstrate**:
  1. Load sentence in Text Playground.
  2. Explain that the clause after `"but"` dominates the overarching sentiment.
  3. Show how Bag of Words gives equal weight to `"stunning"` and `"hollow"`, while the Transformer prioritizes the adversative resolution.

---

## 🛠️ Local Development Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch Local Server
```bash
python server.py
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 3. Run Automated Tests
```bash
python -m unittest test_app.py
```

---

## 👥 Academic Team Contributions

| Team Member | Role | Core Responsibility |
| :--- | :--- | :--- |
| **Team Member 1** | Model Architecture & Training | Developed TF-IDF feature pipelines, embedding vector space, and Transformer execution engine. |
| **Team Member 2** | Backend API & Evaluation Suite | Built the Flask REST API, held-out test split, benchmark runner, and CSV data serialization pipeline. |
| **Team Member 3** | Frontend UI & Interactive Heatmap | Engineered the responsive dashboard, live sentiment comparison layout, and dynamic token-to-token attention heatmap. |
| **Team Member 4** | Benchmarking & Documentation | Curated linguistic test instances (negation, contrastive clauses), authored educational guide, and prepared presentation scripts. |

---

## 📚 Academic Citations & References

1. **J.R. Firth (1957)**: *A Synopsis of Linguistic Theory 1930-1955*. Philological Society.
2. **Pennington, J., Socher, R., & Manning, C. D. (2014)**: *GloVe: Global Vectors for Word Representation*. EMNLP.
3. **Vaswani, A., et al. (2017)**: *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS).
4. **Jain, S., & Wallace, B. C. (2019)**: *Attention is not Explanation*. North American Chapter of the Association for Computational Linguistics (NAACL).
5. **Socher, R., et al. (2013)**: *Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank (SST-2)*. EMNLP.
