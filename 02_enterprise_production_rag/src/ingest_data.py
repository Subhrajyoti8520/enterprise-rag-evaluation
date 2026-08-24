# """
# SEC 10-K Data Ingestion Pipeline for RAG

# This script builds a robust data ingestion pipeline that processes SEC 10-K financial 
# reports and prepares them for Retrieval-Augmented Generation (RAG). It performs the 
# following key steps:
# 1. Data Loading: Fetches the 'virattt/financial-qa-10K' dataset from Hugging Face.
# 2. Semantic Chunking: Splits long financial documents into smaller, overlapping chunks 
#    using LangChain's RecursiveCharacterTextSplitter.
# 3. Context Enrichment: Prepends company ticker symbols to each chunk to preserve 
#    entity context for the embedding model.
# 4. Vector Embedding: Generates dense vector representations of the text using the 
#    'BAAI/bge-large-en-v1.5' model via SentenceTransformers, utilizing GPU/MPS acceleration if available.
# 5. Vector Storage: Upserts the embeddings, raw text, and metadata into a local 
#    ChromaDB vector database using cosine similarity for efficient future retrieval.
# """
# ===========================================================================================================
import time
import logging
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

import torch
import chromadb
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

#------------Configuration & Constants---------------------
@dataclass
class IngestionConfig:
    ingest_full_dataset: bool = False # Set to True to ingest everything for prduction
    dev_sample_size: int = 500

    db_path: Path = Path(__file__).resolve().parent.parent / "data" / "chroma_sec_db"
    collection_name: str = "sec_10k_reports"
    embedding_model: str = "BAAI/bge-large-en-v1.5"

    embed_batch_size: int = 32 #Tuned for local GPU VRAM
    db_batch_size: int = 500

    # Text chunking parameters
    chunk_overlap: int = 100
    chunk_size: int = 600

# Instantiate config
config = IngestionConfig()

#-------------------------Logging Setup----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def generate_chunk_id(doc_id: str, split_pos: int, text: str) -> str:
    """Combines document ID, chunk position, and text hash for guaranteed uniqueness."""
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"sec_{doc_id}_{split_pos}_{text_hash}"

def run_ingestion():
    start_time = time.time()
    logger.info("🚀 STARTING SEC 10-K DATA INGESTION PIPELINE")
    logger.info("=" * 60)

    # Load SEC 10-K Dataset
    try:
        if config.ingest_full_dataset:
            logger.info("🌍 Full Ingestion Mode: Loading complete SEC-10k dataset...")
            dataset = load_dataset("virattt/financial-qa-10K", split="train")
        else: 
            logger.info(f"🛠️ Fast Ingestion Mode: Sampling {config.dev_sample_size} random records to prevent bias...")
            dataset = load_dataset("virattt/financial-qa-10K", split="train")
            dataset = dataset.shuffle(seed=42).select(range(config.dev_sample_size))
        
        logger.info(f"Loaded {len(dataset)} records to process.")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return

    # Semantic Chunking with Context Enrichment
    logger.info("✂️ Chunking documents & injecting ticker context...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    # ticker to identify random paragraph in document which company it refers to

    for chunk_idx, row in enumerate(dataset):
        context  = row.get("context", "")
        ticker = row.get("ticker", "UNKNOWN").strip().upper()

        # Fallback ID generation if 'id' isnt in the dataset
        doc_id = str(row.get("id", f"doc_{chunk_idx}"))

        if not context.strip():
            continue

        raw_splits = text_splitter.split_text(context)
        for split_pos, raw_chunk in enumerate(raw_splits):
            # Context Enrichment: Prepend ticker header so bi-encoders capture entity ownership
            enriched_chunk = f"[{ticker} SEC 10-K Filing]\n{raw_chunk}"  

            documents.append(enriched_chunk)
            metadatas.append({
                "ticker": ticker,
                "source_doc_id": doc_id,
                "chunk_pos": split_pos
            })
            # Idempotent chunk IDs prevent duplicates if script runs twice
            ids.append(generate_chunk_id(doc_id, split_pos, enriched_chunk))

    logger.info(f"Generated {len(documents)} contextualized chunks")

    # Vector Embedding Generation
    # Detect optimal hardware
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"🧠 Loading Embedding Model: {config.embedding_model} on [{device.upper()}]...")

    try:
        model = SentenceTransformer(config.embedding_model, device=device)
        logger.info(f"⚡ Generating embeddings in batches of {config.embed_batch_size}...")

        t_embed = time.time()
        embeddings = model.encode(
            documents,
            batch_size=config.embed_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True
        ).tolist()
        logger.info(f"Embeddings generated in {time.time() - t_embed:.2f}s")
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        return

    # ChromaDB storage(Cosine Distance HNSW Index)
    logger.info("💾 Persisting vectors into ChromaDB...")
    try:
        config.db_path.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(config.db_path))

        # Retrieve or create collection using cosine similarity
        collection = chroma_client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"Batch upserting records to ChromaDB in chunks of {config.db_batch_size}...")
        # Upsert ensures that existing IDs are updated rather than throwing errors
        for i in range(0, len(ids), config.db_batch_size):
            collection.upsert(
                ids=ids[i : i + config.db_batch_size],
                embeddings=embeddings[i : i + config.db_batch_size],
                documents=documents[i : i + config.db_batch_size],
                metadatas=metadatas[i : i + config.db_batch_size]
            )

        total_time = time.time() - start_time
        logger.info(f"✅ INGESTION COMPLETE: {collection.count()} total vectors stored.")
        logger.info(f"⏱️ Total execution time: {total_time:.2f}s")
        logger.info(f"📁 Database Path: {config.db_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Database upsert transaction failed: {e}")
    
if __name__ == "__main__":
    run_ingestion()