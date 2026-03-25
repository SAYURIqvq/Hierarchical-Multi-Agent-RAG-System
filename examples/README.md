# Examples - Agentic RAG System

Quick start examples for using the Agentic RAG system.

---

## 📋 Available Examples

### 1. Simple Upload & Search
**File:** `simple_upload_search.py`

Upload a single document and search interactively.

```bash
python examples/simple_upload_search.py data/test_documents/python_guide.txt
```

**What it does:**
- Loads a document
- Chunks and embeds it
- Stores in ChromaDB
- Interactive search session

---

### 2. Batch Upload
**File:** `batch_upload.py`

Upload entire folder of documents at once.

```bash
python examples/batch_upload.py data/test_documents/
```

**What it does:**
- Scans folder for documents (.txt, .pdf, .docx)
- Processes all files
- Stores everything in ChromaDB
- Shows summary statistics

---

### 3. Interactive Q&A
**File:** `interactive_qa.py`

Question-answering session over indexed documents.

```bash
python examples/interactive_qa.py
```

**Commands:**
- Type questions naturally
- `stats` - Show statistics
- `help` - Show commands
- `quit` - Exit

---

## 🚀 Quick Start

### Step 1: Upload Documents
```bash
# Upload test documents
python examples/batch_upload.py data/test_documents/
```

### Step 2: Ask Questions
```bash
# Start Q&A session
python examples/interactive_qa.py
```

**Example questions:**
- "What is Python?"
- "Explain machine learning"
- "What are the types of machine learning?"

---

## 📁 Test Documents

Sample documents are in `data/test_documents/`:
- `python_guide.txt` - Python programming guide
- `machine_learning.txt` - ML fundamentals

**Add your own:**
```bash
# Copy your documents
cp /path/to/your/document.pdf data/test_documents/

# Upload
python examples/batch_upload.py data/test_documents/
```

---

## 🔧 Requirements

Before running examples:

1. **API Keys** - Set in `.env`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   VOYAGE_API_KEY=pa-...
   ```

2. **Dependencies** - Install:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Folder** - Auto-created on first run

---

## 💡 Tips

**Faster uploads:**
- Batch mode processes multiple files efficiently
- Embeddings are generated in batches of 128

**Better search:**
- Use specific questions
- Include key terms from your documents
- Try different phrasings

**Cost management:**
- Each chunk uses ~1536 tokens for embedding
- Voyage AI charges per token
- Use test documents first

---

## 🐛 Troubleshooting

**No results found:**
```bash
# Check if documents are indexed
python -c "from src.storage import VectorStore; print(VectorStore().count())"
```

**API errors:**
```bash
# Verify API keys
python -c "from src.config import get_settings; s = get_settings(); print('Keys OK')"
```

**Slow processing:**
- Large PDFs take time to embed
- Check your internet connection (API calls)
- Use smaller documents for testing

---

## 📚 Next Steps

After trying examples:
1. Upload your own documents
2. Experiment with different queries
3. Integrate with full workflow (Planner → Coordinator → Validator)
4. Build custom applications

---

**Happy searching! 🔍**