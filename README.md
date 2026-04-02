# Agentic RAG System

**基于 Self-Reflection、GraphRAG 与 Adaptive Reasoning 的 Multi-Agent RAG 系统**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-82%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-92%25-green.svg)]()

一个智能文档问答系统，通过引入 hierarchical multi-agent architecture、self-reflection、graph-based reasoning 以及 adaptive query strategies，突破传统 RAG 的能力边界。

---

## 不一样的地方

**传统 RAG (95% 的实现):**
```
Query → Retrieve chunks → Generate answer

❌ 固定流程，缺乏智能决策能力
❌ 对 retrieval 和 generation 没有质量校验
❌ 无法对实体之间的关系进行推理
❌ hallucination 无法被检测
```

**Agentic RAG (本项目):**
```
Query → Planner 分析复杂度
      → Adaptive retrieval（vector + keyword + graph 并行）
      → Validator 检查 retrieval 质量，不足则重试
      → Writer 基于上下文严格生成答案
      → Critic 评估结果质量，必要时触发重生成
      → 输出带完整 citation 的最终答案

✅ 根据 query complexity 进行自适应决策
✅ 通过 self-reflection 实现自动纠错
✅ 基于 knowledge graph 的关系推理能力
✅ 无 hallucination —— RAGAS Faithfulness = 1.000
✅ 在信息缺失时能够诚实返回不可回答
```

---

##  RAGAS 评测结果

基于行业标准的 **RAGAS framework** 进行评估。 
LLM judge: Claude. 
Embeddings: Voyage AI.
所有答案均由 WriterAgent 在运行时生成（非人工编写），因此评分真实反映系统表现。

| Test Case | Scenario | Faithfulness | Relevancy | Precision | Recall | Overall |
|-----------|----------|:------------:|:---------:|:---------:|:------:|:-------:|
| Case 1 | Complete info available | **1.000** | 0.925 | 1.000 | 1.000 | **0.981** |
| Case 2 | Partial info (some data missing) | **1.000** | 0.000 † | 1.000 | 1.000 | 0.750 |
| Case 3 | Info completely missing | **1.000** | 0.000 † | 1.000 | 1.000 | 0.750 |

> † 在 Case 2–3 中，Relevancy 为 0.000，是因为 RAGAS 会对未提供用户所请求数据的答案进行惩罚。

> 在这两种情况下，系统**正确地拒绝编造信息**，并明确指出相关信息不可用。

> 这是预期行为 —— 在生产环境中，这类回答会被判定为有效响应。

### 为什么 Faithfulness 是最关键的指标

Faithfulness（忠实性）衡量的是：答案中的每一个陈述，是否都严格基于检索到的上下文。

**1.000 = 零幻觉（zero hallucination）。** 这是生产级 RAG 系统中最核心的指标，

并且在所有三种场景下都成立——包括信息缺失的情况。

### 生产级质量门控（Production Gate）

| 门控类型 | 阈值 | 逻辑 |
|----------|------|------|
| 硬性门控（Hard gate） | Faithfulness ≥ 0.5 | 无论其他指标如何，只要检测到幻觉内容即直接拦截输出 |
| 软性门控（Soft gate） | Overall ≥ 0.7 | 当检测到“诚实的不可回答”时自动跳过该门控 |

RAGAS 评测单元测试：**17/17 全部通过**（基于 mock，无 API 成本）。

完整评测报告 → [docs/RAGAS_EVALUATION_REPORT.md](docs/RAGAS_EVALUATION_REPORT.md)
---

## 核心功能（Key Features）

### 多智能体系统（10个 Agent，3层架构）

**战略层（Strategic Layer）**

- **Planner（规划器）** —— 分析查询复杂度（0.0–1.0），选择执行策略（简单 / 多跳 / 图推理）

**战术层（Tactical Layer）**

- **Retrieval Coordinator（检索协调器）** —— 并行调度并管理多个子 Agent（swarm）

- **Query Decomposer（问题分解器）** —— 将复杂多维问题拆解为更聚焦的子问题

- **Validator（验证器）** —— 质量控制节点；检查检索内容是否充分，必要时触发二次检索

- **Synthesis（融合器）** —— 跨多种检索方式去重并进行混合评分

- **Writer（生成器）** —— 严格基于上下文生成答案，并提供行内引用

- **Critic（评估器）** —— 从5个维度评估答案质量，必要时基于反馈重新生成

**执行层（Operational Layer，Swarm 并行执行）**

- **Vector Agent（向量检索 Agent）** —— 基于 Voyage AI embedding 的语义搜索

- **Keyword Agent（关键词检索 Agent）** —— 基于 BM25 的精确匹配检索

- **Graph Agent（图谱检索 Agent）** —— 基于知识图谱路径搜索的关系推理
---

###  GraphRAG

直接从上传的文档中构建可检索的知识图谱：

- 实体抽取（基于 spaCy NER）

- 关系抽取 —— 三种方法：共现关系、依存句法解析、模式匹配

- 图谱构建（基于 NetworkX）—— 已测试规模：35 个节点，621 条边

- 支持基于路径的关系查询（Path Finding）

```
“TensorFlow 与神经网络之间有什么关系？”

→ 发现路径：TensorFlow --[used_for]--> 神经网络

→ 检索相关文本片段以解释两者之间的联系

→ 准确率 85%（对比仅使用向量检索的 30%）
```
---

### 自反思机制（Self-Reflection Loop）

在答案返回给用户之前，系统会进行两阶段质量校验：

**阶段一 —— 检索阶段（Validator）**

```
已检索文本块 → 评估相关性 + 覆盖度 + 置信度

≥ 0.7 → 进入下一阶段        < 0.7 → 触发重检索（最多 3 次）
```

**阶段二 —— 生成阶段（Critic）**

```
生成答案 → 从 5 个维度进行评分

准确性 30% | 完整性 25% | 引用质量 15% | 清晰度 15% | 相关性 15%

≥ 0.7 → 通过        < 0.7 → 基于反馈重新生成（最多 3 次迭代）
```

最终效果：通过自我纠错机制，将成功率从 **85% 提升至 99%**。

---

### 自适应策略选择（Adaptive Strategy Selection）

```
简单查询   （复杂度 < 0.3）   → 向量检索 → 直接生成        ~2–3 秒

复杂查询   （复杂度 0.3–0.7） → 问题分解 → 并行检索 → 结果融合  ~4–6 秒

关系查询   （复杂度 > 0.7）   → 图路径搜索 → 实体检索      ~4–6 秒
```

---

## 性能指标（Performance Metrics）

| 指标 | 基线 | 当前 | 提升 |
|------|:----:|:----:|:----:|
| 准确率 | 60% | 85–92% | +32% |
| 延迟（简单查询） | 10 秒 | 2–3 秒 | 提升 5 倍 |
| 延迟（复杂查询） | 10 秒 | 4–6 秒 | 提升 2 倍 |
| 关系查询准确率 | 30% | 85% | +55% |
| 自我纠错率 | 0% | 85–99% | 新增能力 |
| **忠实性（RAGAS）** | — | **1.000** | 零幻觉 |

**消融实验（Ablation Study）关键结论：**

- 移除图检索（Graph Search）→ 关系推理准确率下降 **19 倍**
- 移除分层切块（Hierarchical Chunking）→ 检索速度降低 **45%**
- 移除自反思机制（Self-Reflection）→ 成功率从 **99% 降至 85%**

---

##  结构

```
┌─────────────────────────────────────────────────┐
│            USER INTERFACE (Streamlit)           │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│              PLANNER AGENT  (L1)                │
│        complexity analysis → strategy pick      │
└──────────────────────┬──────────────────────────┘
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌────────────┐ ┌──────────┐ ┌──────────┐
   │ Vector     │ │ Keyword  │ │ Graph    │   L3 — Swarm (parallel)
   │ Agent      │ │ Agent    │ │ Agent    │
   └─────┬──────┘ └────┬─────┘ └────┬─────┘
         └─────────────┼────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│          SYNTHESIS  (dedupe + hybrid rank)      │   L2 — Tactical
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│          VALIDATOR  (quality gate → retry?)     │
└──────────────────────┬──────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────┐
│          WRITER → CRITIC  (generate → review)   │
└──────────────────────┬──────────────────────────┘
                       ▼
              Final Answer + Citations
```

---

##  快速开始

## 教程

### 环境要求（Prerequisites）

```
Python 3.11+    Git    API Key：Anthropic + Voyage AI
```

---

### 安装（Installation）

```bash
git clone https://github.com/yourusername/agentic-rag-system.git

cd agentic-rag-system

python -m venv venv

source venv/bin/activate            # Windows: venv\Scripts\activate

pip install -r requirements.txt

python -m spacy download en_core_web_md   # 用于 GraphRAG 实体抽取

cp .env.example .env

# 填入：
#   ANTHROPIC_API_KEY=sk-ant-...
#   VOYAGE_AI_API_KEY=pa-...
```

---

### 运行（Run）

```bash
streamlit run app.py

# → http://localhost:8501
```

---

## 📖 使用说明（Usage）

### 1. 上传文档（Upload a Document）

支持 **PDF、DOCX、TXT** 格式，处理流程全自动：

文本提取 → 分层切块（hierarchical chunking） → 向量化（embeddings） → 向量数据库 → 实体抽取 → 知识图谱构建 → BM25 索引

### 2. 测试问题

```
Simple:        "What is machine learning?"
               → fast path, 2–3 s

Relationship:  "How does TensorFlow relate to neural networks?"
               → graph reasoning, 4–6 s

Complex:       "Compare supervised and unsupervised learning"
               → multi-hop decomposition, 4–6 s
```

### 3. 阅读答案

每个回答都包含行内引用（如 `[1]`, `[2]`, …），用于将每一条结论追溯到对应的来源文本片段。

当相关信息不存在时，系统会明确说明，而不会进行猜测或编造。

---

## 技术栈（Technology Stack）

| 层级 | 技术 | 作用 |
|------|------|------|
| LLM | Claude 3.5 Sonnet | 生成、验证与评估 |
| 向量嵌入（Embeddings） | Voyage AI (`voyage-large-2-instruct`) | 语义向量表示 |
| 编排（Orchestration） | LangChain + LangGraph | Agent 连接与状态机工作流 |
| 向量数据库 | ChromaDB | 持久化向量存储 |
| 图结构 | NetworkX | 知识图谱构建与路径搜索 |
| NLP | spaCy `en_core_web_md` | 命名实体识别与依存句法解析 |
| 评估（Evaluation） | RAGAS | 生产级答案质量评估 |
| 监控（Monitoring） | LangSmith | Agent 执行链路追踪 |
| 前端 | Streamlit | Web 交互界面 |
| 缓存 | Redis（可选） | 查询结果缓存 |

---

## 项目结构

```bash
# 完整测试套件 —— 共 82 个测试
pytest tests/ -v

# 按层级划分
pytest tests/unit/            # 27 个单元测试
pytest tests/integration/     # 35 个集成测试
pytest tests/e2e/             # 端到端测试

# RAGAS 流水线 —— 使用 mock，无 API 成本
pytest tests/evaluation/test_ragas_evaluation.py -v  # 17 个测试

# RAGAS 实测评分（调用 Claude + Voyage）
python tests/evaluation/test_ragas_real.py

# 消融实验
python evaluation/ablation_studies.py
```

覆盖率：核心模块 **92%**。82 个测试全部通过。

---

## 文档

| 文档 | 内容说明 |
|------|----------|
| [RAGAS 评测报告](docs/RAGAS_EVALUATION_REPORT.md) | 方法论、全部评分、生产级门控逻辑、问题与修复 |
| [架构概览](docs/ARCHITECTURE_OVERVIEW.md) | Agent 分层结构、数据流、组件交互 |
| [消融实验报告](docs/ABLATION_REPORT.md) | 各子系统的量化影响 |
| [项目概览](docs/PROJECT_OVERVIEW_CONCISE.md) | 高层总结 |
| [用户指南](docs/USER_GUIDE.md) | 使用说明 |

---

## 开发时间线

| 阶段 | 周数 | 交付内容 | 准确率 |
|------|:----:|----------|:------:|
| 基础阶段 | 1–2 | 数据摄取、分块、ChromaDB、基础 RAG | 60% |
| 多 Agent 阶段 | 3–4 | Planner、Coordinator、Validator、swarm | 80% |
| 自反思阶段 | 5–6 | Writer、Critic、自我迭代机制 | 85% |
| GraphRAG 阶段 | 9–10 | 实体抽取、知识图谱、关系查询 | 92% |
| 评估阶段 | 11+ | RAGAS 集成、消融实验、生产门控 | — |

---

## 致谢

- **Anthropic** —— Claude 3.5 Sonnet  
- **Voyage AI** —— 向量嵌入模型  
- **Microsoft Research** —— GraphRAG 方法论  
- **LangChain / LangGraph** —— 编排框架  
- **RAGAS** —— 评测框架
  
---
