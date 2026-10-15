# Multi-Backend RAG Architecture (`rag/`)

Multiple retrieval and generation paths over the same synthetic media operations corpus (`data/synthetic/`), using Ollama (`llama3.2` + `nomic-embed-text`) or OpenAI and the shared system prompt (`guardrails/prompts/system_rag.txt`).

## Pipeline overview

```
data/synthetic/  (.md, .txt)
       │
       ▼
   chunk (512 chars, 50 overlap)
       │
       ▼
   embed (nomic-embed-text / text-embedding-3-small)
       │
       ▼
   vector store (pgvector, Chroma, Pinecone, LlamaIndex, Hybrid BM25+Vector)
       │
       ▼
   retrieve (top-k=4 + optional metadata filters)
       │
       ▼
   generate (LLM + guardrails/prompts/system_rag.txt)
       │
       ▼
   cited answer + source list
```

## Retrieval & orchestration comparison

| | Direct Chroma | LangChain | LlamaIndex |
|---|---------------|-----------|------------|
| **Entry** | `ingest.py` + `query.py` | `query_langchain.py` | `query_llamaindex.py` |
| **Index** | `.chroma/` | `.chroma_langchain/` | `.llamaindex_storage/` |
| **Vector store** | ChromaDB (native) | Chroma via `langchain_community` | LlamaIndex JSON persist |
| **Chunking** | Fixed char window in `ingest.py` | `RecursiveCharacterTextSplitter` | `LangchainNodeParser` + same splitter |
| **Retrieval** | `collection.query()` | `vectorstore.as_retriever(k=4)` | `index.as_query_engine(top_k=4)` |
| **Generation** | `shared/llm.chat_completion()` | `create_retrieval_chain` + stuff docs | Query engine + `PromptTemplate` |

Each path maintains its own index directory so they can run side-by-side without collision.

## Production vector stores & hybrid retrieval

- **Adapter interface:** `stores/base.py` (`VectorStore`, `ChunkHit`)
- **Corpus loader:** `stores/corpus.py` merges `sharepoint-meta.json` (`doc_type`, `confidentiality`, derived `rights_tier`)
- **Postgres `pgvector` store:** `stores/pgvector.py` — `documents` + `document_chunks` tables, HNSW cosine index, metadata-filtered search with SQL join
- **Pinecone store:** `stores/pinecone.py` — serverless index, metadata filters, auto-create index
- **Hybrid BM25 + vector:** `stores/hybrid.py` — `BM25Okapi` + reciprocal rank fusion (RRF) with dense vector hits (`query_hybrid.py`)

## Data lake & incremental ingest

The batch pipeline (`platform/pipelines/ingest_batch.py`) lands normalized documents under `data/lake/raw/`:

```bash
python platform/pipelines/ingest_batch.py
python rag/ingest_pgvector.py --data-dir data/lake/raw
python rag/ingest_pgvector.py --data-dir data/lake/raw --lora-tag
```

- **Incremental embed worker:** `embed_worker.py` upserts individual documents via `PgVectorStore.ingest_document()` on `document_ingested` events
- **LoRA tier tags:** `shared/rights_tier_tagger.py` tags chunks at ingest; authoritative SQL warehouse tiers take precedence on `rights_clearance` documents
- **Architecture:** [docs/data-plane.md](../docs/data-plane.md)

## File map

```
rag/
├── ingest.py              # Chroma ingest
├── ingest_pgvector.py     # Postgres pgvector + SQL clearance ingest
├── ingest_pinecone.py     # Pinecone serverless ingest
├── query.py               # Chroma RAG query
├── query_pgvector.py      # pgvector query + metadata filters
├── query_pinecone.py      # Pinecone query + metadata filters
├── query_hybrid.py        # BM25 + dense vector (RRF) query
├── query_langchain.py     # LangChain RAG query
├── query_llamaindex.py    # LlamaIndex RAG query
├── ARCHITECTURE.md        # Architecture overview
└── evals/
    ├── gold_qa.json       # 23 gold Q&A benchmark pairs
    ├── run_eval.py        # Multi-backend keyword & citation evaluation
    ├── run_ragas.py       # RAGAS semantic evaluation
    └── results.json       # Latest evaluation run output
```
