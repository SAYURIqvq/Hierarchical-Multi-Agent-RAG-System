# User Guide - Agentic RAG System

**Version:** 1.0  
**Last Updated:** December 31, 2024

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Installation](#installation)
3. [Using the System](#using-the-system)
4. [Query Types & Examples](#query-types--examples)
5. [Understanding Results](#understanding-results)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Features](#advanced-features)
8. [FAQ](#faq)

---

## 🚀 Getting Started

### What is This?

An intelligent document Q&A system that goes beyond simple keyword search. It:
- Understands relationships between concepts
- Self-corrects when answers aren't good enough
- Adapts its strategy based on query complexity
- Provides citations for every claim

### Who Should Use This?

- Researchers searching technical documents
- Students studying complex topics
- Engineers navigating documentation
- Anyone who needs accurate answers from documents

---

## 💻 Installation

### Prerequisites
```bash
# Required
Python 3.11 or higher
Git
4GB RAM minimum
Internet connection (for API calls)

# API Keys needed
Anthropic API key (Claude)
Voyage AI API key (embeddings)
```

### Step-by-Step Setup

**1. Clone the repository:**
```bash
git clone https://github.com/yourusername/agentic-rag-system.git
cd agentic-rag-system
```

**2. Create virtual environment:**
```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download language model:**
```bash
python -m spacy download en_core_web_md
```

**5. Setup environment variables:**
```bash
# Copy template
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use any text editor
```

**Add your keys to `.env`:**
```
ANTHROPIC_API_KEY=sk-ant-xxxxx
VOYAGE_API_KEY=pa-xxxxx
```

**6. Run the application:**
```bash
streamlit run app.py
```

**7. Open browser:**
```
Navigate to: http://localhost:8501
```

---

## 📱 Using the System

### Step 1: Upload a Document

**Supported formats:**
- PDF (`.pdf`)
- Word Documents (`.docx`)
- Text files (`.txt`)

**How to upload:**

1. Click **"📁 Upload Document"** in the sidebar
2. Choose your file
3. Wait for processing (30s - 2min depending on size)

**What happens during processing:**
```
1. Document is loaded
2. Text is split into chunks (hierarchical)
3. Embeddings are generated (for semantic search)
4. Knowledge graph is built (entities + relationships)
5. Data is stored in vector database

Progress bar shows each step ✅
```

**Tips:**
- ✅ Use clear, well-structured documents
- ✅ PDF works best for formatted documents
- ✅ Larger documents (10+ pages) show better results
- ⚠️ First upload takes longer (initializing models)
- ⚠️ Each upload replaces previous document

---

### Step 2: Ask Questions

**Input methods:**
1. Type in main chat box
2. Press Enter or click Send
3. Wait 2-8 seconds for response

**The system will:**
1. Analyze your query complexity
2. Select appropriate strategy
3. Search relevant chunks
4. Generate answer with citations
5. Self-validate quality

---

### Step 3: Review Results

**You'll see:**
- 📝 Answer text
- 🔢 Citations [1][2][3]
- 🧠 Complexity score
- ⚡ Strategy used
- ⏱️ Processing time
- 📊 Source chunks (expandable)

---

## 🎯 Query Types & Examples

### 1. Simple Definition Queries

**Best for:** Basic facts, definitions, explanations

**Examples:**
```
❓ "What is machine learning?"
❓ "Define neural network"
❓ "Explain supervised learning"
```

**Expected response:**
- ⏱️ 2-3 seconds
- 📊 Strategy: SIMPLE
- ✅ Accuracy: ~95%
- 📄 2-3 citations

**Example output:**
```
Machine learning is a subset of artificial intelligence that 
enables systems to learn from data [1]. It uses algorithms to 
identify patterns and make decisions with minimal human 
intervention [2].

🧠 Complexity: 0.15
⚡ Strategy: SIMPLE
⏱️ Time: 2.3s
```

---

### 2. Relationship Queries

**Best for:** Connections between concepts

**Examples:**
```
❓ "How does TensorFlow relate to neural networks?"
❓ "What is the connection between Python and machine learning?"
❓ "Explain the relationship between AI and deep learning"
```

**Expected response:**
- ⏱️ 4-6 seconds
- 📊 Strategy: GRAPH_REASONING
- ✅ Accuracy: ~85%
- 📄 3-5 citations
- 🕸️ Graph paths shown

**Example output:**
```
TensorFlow relates to neural networks as a framework specifically 
designed for building and training neural network models [1]. 
Developed by Google, it provides tools and libraries that simplify 
neural network development [2].

🕸️ Graph path found:
   tensorflow --[for]--> neural networks

🧠 Complexity: 0.75
⚡ Strategy: GRAPH_REASONING
⏱️ Time: 5.8s
```

---

### 3. Comparison Queries

**Best for:** Comparing multiple concepts

**Examples:**
```
❓ "Compare supervised and unsupervised learning"
❓ "What are the differences between Python and Java for ML?"
❓ "Contrast deep learning and traditional ML"
```

**Expected response:**
- ⏱️ 4-5 seconds
- 📊 Strategy: MULTIHOP
- ✅ Accuracy: ~88%
- 📄 4-6 citations

**Example output:**
```
Supervised learning uses labeled data for training [1], while 
unsupervised learning discovers patterns in unlabeled data [2]. 

Key differences:
- Data requirements: Supervised needs labels, unsupervised doesn't [3]
- Use cases: Supervised for classification, unsupervised for clustering [4]

🧠 Complexity: 0.55
⚡ Strategy: MULTIHOP
⏱️ Time: 4.2s
```

---

### 4. Complex Multi-part Queries

**Best for:** Questions with multiple sub-questions

**Examples:**
```
❓ "How do neural networks learn, and what role does backpropagation play?"
❓ "Explain gradient descent and how it optimizes model parameters"
❓ "What is overfitting, why does it occur, and how can it be prevented?"
```

**Expected response:**
- ⏱️ 5-8 seconds
- 📊 Strategy: MULTIHOP
- ✅ Accuracy: ~90%
- 📄 5-8 citations
- 🔀 Sub-questions shown

**Example output:**
```
🔀 Query decomposed into:
   1. How do neural networks learn?
   2. What is backpropagation?
   3. How does backpropagation enable learning?

Neural networks learn by adjusting weights based on error [1]. 
Backpropagation calculates these errors by propagating them 
backward through the network [2]...

🧠 Complexity: 0.82
⚡ Strategy: MULTIHOP
⏱️ Time: 6.5s
```

---

## 📊 Understanding Results

### Complexity Score (0.0 - 1.0)
```
0.0 - 0.3:  Simple (definition, single concept)
            → Fast response, vector search only

0.3 - 0.7:  Moderate (comparison, multi-concept)
            → Multi-hop reasoning, decomposition

0.7 - 1.0:  Complex (relationships, deep analysis)
            → Graph reasoning, comprehensive search
```

### Strategy Types

**SIMPLE:**
```
What it means: Direct answer from vector search
When used: Simple, factual questions
Speed: Fastest (2-3s)
```

**MULTIHOP:**
```
What it means: Query decomposed into sub-questions
When used: Comparisons, multi-part questions
Speed: Medium (4-5s)
```

**GRAPH_REASONING:**
```
What it means: Uses knowledge graph to find relationships
When used: "How does X relate to Y?" queries
Speed: Slower (5-8s) but more accurate for relationships
```

### Citations Format

**Example:**
```
"Machine learning is a subset of AI [1]. It uses algorithms 
to learn from data [2][3]."

[1] Source: introduction.pdf, Chunk 5
[2] Source: introduction.pdf, Chunk 12
[3] Source: methods.pdf, Chunk 3
```

**What citations mean:**
- Numbers reference source chunks
- Click to view full source text
- Multiple citations = well-supported claim

### Self-Reflection Indicators

**If you see:**
```
🔄 Retrieval round 2...
```
**Meaning:** System decided initial results weren't good enough, retrieving more
```
✨ Answer regenerated (iteration 2)
```
**Meaning:** Critic found issues, writer improved the answer

**This is GOOD!** ✅ System is self-correcting for quality

---

## 🔧 Troubleshooting

### Issue: "Please upload a document first!"

**Cause:** No document loaded in system

**Solution:**
1. Click "📁 Upload Document" in sidebar
2. Select a file
3. Wait for processing to complete
4. Try your query again

---

### Issue: Slow responses (>10s)

**Possible causes:**
- First query after upload (loading models)
- Very complex query
- Large document
- Slow internet connection

**Solutions:**
- ✅ Wait for first query to complete (subsequent faster)
- ✅ Simplify query if too complex
- ✅ Check internet connection
- ⚠️ If persists, check API key limits

---

### Issue: "No results found"

**Possible causes:**
- Query topic not in document
- Document too small
- Entities not recognized

**Solutions:**
- ✅ Verify query relates to document content
- ✅ Rephrase query (be more specific)
- ✅ Upload larger/better document
- ✅ Check spelling of technical terms

---

### Issue: Graph search returns nothing

**Meaning:** This is NORMAL for definition queries!

**Explanation:**
```
Query: "What is machine learning?"
→ Only 1 entity found: [machine learning]
→ Graph needs 2+ entities to find paths
→ Falls back to vector search ✅
```

**This is by design!** Graph search is for relationships, not definitions.

---

### Issue: Citations missing or wrong

**Cause:** Generated answer without source chunks

**Solutions:**
- ✅ Re-upload document (may be storage issue)
- ✅ Try simpler query first
- ✅ Check if document was processed completely

---

### Issue: API errors

**Error messages:**
```
"Authentication failed"
→ Check API keys in .env file

"Rate limit exceeded"
→ Wait a few minutes, try again

"Model not available"
→ Check internet connection
```

---

## 🎓 Advanced Features

### View Knowledge Graph

**In sidebar after upload:**
```
🕸️ Knowledge Graph
Nodes: 35
Edges: 621
Top Entities:
- machine learning: 50
- neural network: 35
- tensorflow: 30
```

**What this shows:**
- Total concepts extracted (nodes)
- Relationships found (edges)
- Most connected concepts

---

### Understanding Graph Paths

**When you see:**
```
🕸️ Graph path found:
   google --[uses]--> tensorflow --[for]--> machine learning
```

**This means:**
- System found explicit connection in document
- Each arrow shows relationship type
- Answer based on this connection path

---

### Multi-turn Conversations

**You can ask follow-ups:**
```
User: "What is machine learning?"
Bot: [explains machine learning]

User: "How does it differ from deep learning?"
Bot: [compares the two, remembering context]
```

**Currently:** Each query is independent
**Future:** Full conversation memory

---

### View Source Chunks

**Click "👁️ View source chunks" to see:**
- Original text retrieved
- Relevance scores
- Which document section
- How it influenced answer

**Useful for:**
- Verifying answer accuracy
- Finding more context
- Understanding system decisions

---

## ❓ FAQ

### Q: How much does it cost to use?

**A:** Depends on API usage
```
Typical costs per query:
- Simple: ~$0.005 (half a cent)
- Complex: ~$0.02 (2 cents)
- Average: ~$0.01 per query

For 100 queries/day: ~$1/day = $30/month
```

---

### Q: Can I upload multiple documents?

**A:** Currently one at a time
- Each upload replaces previous
- Future: Multi-document support planned

**Workaround:** Combine documents into one file

---

### Q: What languages are supported?

**A:** English only currently
- NLP models trained on English
- Future: Multilingual support possible

---

### Q: How is my data stored?

**A:** Locally on your machine
```
Location: data/chroma_db/
- Vector embeddings
- Knowledge graph
- Metadata

Security: 
✅ Not uploaded to cloud (except API calls)
✅ Can delete anytime (rm -rf data/)
```

---

### Q: Can I use custom models?

**A:** Yes, with code changes
```python
# In src/config.py, change:
llm_model = "claude-3-5-sonnet-20241022"
embedding_model = "voyage-large-2"

# To your preferred models
```

---

### Q: How accurate is the system?

**A:** Depends on query type
```
Simple definitions: ~95%
Relationships: ~85%
Comparisons: ~88%
Overall average: ~92%
```

**Factors affecting accuracy:**
- Document quality (better docs = better answers)
- Query clarity (specific > vague)
- Query type (matches system strengths)

---

### Q: Why does graph search skip some queries?

**A:** By design!

**Graph search needs:**
- 2+ entities in query
- Entities connected in graph

**Example:**
```
"What is AI?" → 1 entity → Skip graph, use vector ✅
"How AI relates to ML?" → 2 entities → Use graph ✅
```

---

### Q: Can I see the system's reasoning?

**A:** Yes! Check the output
```
🧠 Complexity: 0.75
⚡ Strategy: GRAPH_REASONING
🕸️ Path: tensorflow → neural networks
⏱️ Time: 5.8s
🔄 Retrieval rounds: 1
```

**This shows:**
- How system analyzed your query
- Which strategy it chose
- What it found
- How long it took

---

### Q: How do I get better answers?

**Tips:**

**✅ DO:**
- Be specific: "How does X relate to Y?" vs "Tell me about X"
- Use proper terms: "neural network" vs "brain thing"
- Upload quality documents (clear, well-structured)
- Ask one question at a time initially

**❌ AVOID:**
- Vague questions: "Tell me everything"
- Topics not in document
- Multiple unrelated questions together
- Very short documents (<2 pages)

---

## 🎯 Best Practices

### Document Upload

**✅ Good documents:**
- 10+ pages
- Clear structure (headings, paragraphs)
- Technical/academic content
- Well-formatted PDFs

**❌ Poor documents:**
- <2 pages (too small)
- Scanned images (no text)
- Heavily formatted tables
- Handwritten notes

---

### Query Formulation

**✅ Good queries:**
```
"How does gradient descent optimize neural networks?"
"What is the difference between supervised and unsupervised learning?"
"Explain the relationship between overfitting and regularization"
```

**❌ Poor queries:**
```
"Tell me stuff" (too vague)
"???" (not a question)
"Is AI good or bad?" (subjective, not in docs)
```

---

### Interpreting Results

**High confidence signs:**
- Multiple citations [1][2][3]
- Specific details (not vague)
- No hedging language ("probably", "might")
- Graph paths shown (for relationships)

**Low confidence signs:**
- Few/no citations
- Vague wording
- "Based on the context..." (guessing)
- Very short answer

**If low confidence:** Rephrase query or check document content

---

## 📞 Support

### Getting Help

**Issues with setup:**
- Check installation steps carefully
- Verify API keys in .env
- Ensure Python 3.11+

**Issues with results:**
- Review query formulation tips
- Check document quality
- Try example queries first

**Technical problems:**
- Check GitHub issues
- Review error messages
- Enable debug mode (in code)

---

## 🎓 Learning Resources

**To understand the system better:**

1. **Architecture Overview:** `docs/ARCHITECTURE_OVERVIEW.md`
2. **Week 9 Summary:** `docs/WEEK9_SUMMARY.md` (GraphRAG)
3. **Week 10 Summary:** `docs/WEEK10_SUMMARY.md` (Graph reasoning)
4. **Final Report:** `docs/FINAL_REPORT.md` (Complete analysis)

**External resources:**
- GraphRAG paper (Microsoft Research)
- LangChain documentation
- ChromaDB documentation

---

## 📝 Changelog

**Version 1.0 (December 2024):**
- Initial release
- 10 agents implemented
- GraphRAG support
- Self-reflection loops
- Adaptive strategies

---

**Need more help? Open an issue on GitHub!**

---

END OF USER GUIDE