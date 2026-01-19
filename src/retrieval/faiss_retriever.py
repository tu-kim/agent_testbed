"""
FAISS Vector Database with Tiered Storage (RAM + SSD)

Implements dense retrieval using FAISS with support for
HNSW and IVF indexing algorithms. Supports tiered storage
where hot data resides in RAM and cold data on SSD.
"""

import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import heapq

import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu")

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexAlgorithm(str, Enum):
    """Supported FAISS index algorithms."""
    HNSW = "hnsw"
    IVF = "ivf"
    FLAT = "flat"


@dataclass
class HNSWConfig:
    """Configuration for HNSW index."""
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 128


@dataclass
class IVFConfig:
    """Configuration for IVF index."""
    nlist: int = 100
    nprobe: int = 10


@dataclass
class TieredStorageConfig:
    """Configuration for tiered storage."""
    enabled: bool = False
    ram_capacity: int = 100000  # Max vectors in RAM (hot tier)
    ssd_dir: str = "./cache/ssd_index"  # Directory for SSD-based cold tier
    promotion_threshold: int = 5  # Access count to promote from cold to hot


@dataclass
class FAISSRetrieverConfig:
    """Configuration for FAISS retriever."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: str = "./cache/faiss_index"
    algorithm: IndexAlgorithm = IndexAlgorithm.HNSW
    hnsw_config: HNSWConfig = field(default_factory=HNSWConfig)
    ivf_config: IVFConfig = field(default_factory=IVFConfig)
    tiered_config: TieredStorageConfig = field(default_factory=TieredStorageConfig)
    batch_size: int = 128
    normalize_embeddings: bool = True


class TieredFAISSRetriever:
    """
    FAISS Retriever with Tiered Storage (RAM + SSD).
    
    - Hot Tier (RAM): Frequently accessed or recently added vectors
    - Cold Tier (SSD): Less frequently accessed vectors stored on disk
    
    Search queries both tiers and merges results.
    """
    
    def __init__(self, config: FAISSRetrieverConfig):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required.")
        
        self.config = config
        self.embedding_model: Optional[SentenceTransformer] = None
        self.dimension: int = 0
        
        # Hot tier (RAM)
        self.hot_index: Optional[faiss.Index] = None
        self.hot_documents: List[Dict[str, Any]] = []
        self.hot_ids: set = set()  # Track document IDs in hot tier
        
        # Cold tier (SSD)
        self.cold_index: Optional[faiss.Index] = None
        self.cold_documents: List[Dict[str, Any]] = []
        self.cold_ids: set = set()
        
        # Access tracking for promotion
        self.access_counts: Dict[str, int] = {}
        
        self._is_trained: bool = False
        
        os.makedirs(config.cache_dir, exist_ok=True)
        if config.tiered_config.enabled:
            os.makedirs(config.tiered_config.ssd_dir, exist_ok=True)
    
    def _load_embedding_model(self):
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            
            # Use PyTorch to detect CUDA availability (more reliable than FAISS GPU check)
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                logger.info(f"CUDA version: {torch.version.cuda}")
            else:
                device = "cpu"
                logger.warning("No GPU detected, using CPU for embeddings")
            
            self.embedding_model = SentenceTransformer(self.config.embedding_model, device=device)
            
            # Verify model is on the correct device
            actual_device = str(next(self.embedding_model.parameters()).device)
            logger.info(f"Embedding model loaded on device: {actual_device}")
            
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
    
    def _create_index(self, num_vectors: int = 0) -> faiss.Index:
        """Create a FAISS index based on configuration."""
        self._load_embedding_model()
        
        if self.config.algorithm == IndexAlgorithm.HNSW:
            index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_config.M)
            index.hnsw.efConstruction = self.config.hnsw_config.ef_construction
            index.hnsw.efSearch = self.config.hnsw_config.ef_search
        elif self.config.algorithm == IndexAlgorithm.IVF:
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = int(4 * np.sqrt(num_vectors)) if num_vectors > 0 else self.config.ivf_config.nlist
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            index.nprobe = self.config.ivf_config.nprobe
        else:
            index = faiss.IndexFlatL2(self.dimension)
        
        return index
    
    def _encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_embedding_model()
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)
    
    def add_documents(self, documents: List[Dict[str, Any]], tier: str = "auto"):
        """
        Add documents to the index.
        
        Args:
            documents: List of document dictionaries
            tier: "hot" (RAM), "cold" (SSD), or "auto" (fill hot first, then cold)
        """
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents (tier={tier})...")
        self._load_embedding_model()
        
        texts = [doc.get("text", "") for doc in documents]
        embeddings = self._encode_texts(texts)
        
        if not self.config.tiered_config.enabled or tier == "hot":
            # All to hot tier
            self._add_to_hot(documents, embeddings)
        elif tier == "cold":
            # All to cold tier
            self._add_to_cold(documents, embeddings)
        else:
            # Auto: fill hot tier first, overflow to cold
            hot_capacity = self.config.tiered_config.ram_capacity
            current_hot = len(self.hot_documents)
            available_hot = max(0, hot_capacity - current_hot)
            
            if available_hot > 0:
                hot_docs = documents[:available_hot]
                hot_embs = embeddings[:available_hot]
                self._add_to_hot(hot_docs, hot_embs)
            
            if len(documents) > available_hot:
                cold_docs = documents[available_hot:]
                cold_embs = embeddings[available_hot:]
                self._add_to_cold(cold_docs, cold_embs)
    
    def _add_to_hot(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to hot tier (RAM)."""
        if self.hot_index is None:
            self.hot_index = self._create_index(len(documents))
        
        if self.config.algorithm == IndexAlgorithm.IVF and not self._is_trained:
            logger.info("Training IVF index (hot tier)...")
            self.hot_index.train(embeddings)
            self._is_trained = True
        
        start_idx = len(self.hot_documents)
        for i, doc in enumerate(documents):
            doc["_tier"] = "hot"
            doc["_tier_idx"] = start_idx + i
            doc_id = doc.get("id", f"hot_{start_idx + i}")
            self.hot_ids.add(doc_id)
        
        self.hot_index.add(embeddings)
        self.hot_documents.extend(documents)
        logger.info(f"Hot tier now contains {self.hot_index.ntotal} vectors")
    
    def _add_to_cold(self, documents: List[Dict[str, Any]], embeddings: np.ndarray):
        """Add documents to cold tier (SSD)."""
        if self.cold_index is None:
            self.cold_index = self._create_index(len(documents))
        
        if self.config.algorithm == IndexAlgorithm.IVF and self.cold_index.ntotal == 0:
            logger.info("Training IVF index (cold tier)...")
            self.cold_index.train(embeddings)
        
        start_idx = len(self.cold_documents)
        for i, doc in enumerate(documents):
            doc["_tier"] = "cold"
            doc["_tier_idx"] = start_idx + i
            doc_id = doc.get("id", f"cold_{start_idx + i}")
            self.cold_ids.add(doc_id)
        
        self.cold_index.add(embeddings)
        self.cold_documents.extend(documents)
        
        # Persist cold index to SSD immediately
        self._save_cold_tier()
        logger.info(f"Cold tier now contains {self.cold_index.ntotal} vectors (saved to SSD)")
    
    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[float], float]:
        """
        Search both hot and cold tiers, merge results.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            Tuple of (documents, scores, retrieval_time_ms)
        """
        start_time = time.perf_counter()
        
        query_embedding = self._encode_texts([query], show_progress=False)
        
        all_results = []
        
        # Search hot tier (RAM)
        if self.hot_index is not None and self.hot_index.ntotal > 0:
            distances, indices = self.hot_index.search(query_embedding, min(top_k, self.hot_index.ntotal))
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.hot_documents):
                    doc = self.hot_documents[idx]
                    all_results.append((float(distances[0][i]), doc, "hot"))
                    # Track access
                    doc_id = doc.get("id", "")
                    self.access_counts[doc_id] = self.access_counts.get(doc_id, 0) + 1
        
        # Search cold tier (SSD)
        if self.cold_index is not None and self.cold_index.ntotal > 0:
            distances, indices = self.cold_index.search(query_embedding, min(top_k, self.cold_index.ntotal))
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.cold_documents):
                    doc = self.cold_documents[idx]
                    all_results.append((float(distances[0][i]), doc, "cold"))
                    # Track access for potential promotion
                    doc_id = doc.get("id", "")
                    self.access_counts[doc_id] = self.access_counts.get(doc_id, 0) + 1
        
        # Merge and sort by distance (lower is better for L2)
        all_results.sort(key=lambda x: x[0])
        top_results = all_results[:top_k]
        
        retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        
        documents = [r[1] for r in top_results]
        scores = [r[0] for r in top_results]
        
        return documents, scores, retrieval_time_ms
    
    def promote_to_hot(self, doc_ids: List[str]):
        """
        Promote documents from cold tier to hot tier.
        
        Note: This is a simplified implementation. Full implementation would
        require rebuilding indices or using IDMap indices for removal.
        """
        logger.info(f"Promoting {len(doc_ids)} documents to hot tier...")
        # In production, you'd need to:
        # 1. Find the documents in cold tier
        # 2. Re-encode or store embeddings
        # 3. Add to hot tier
        # 4. Mark as removed in cold tier (or rebuild)
        pass
    
    def demote_to_cold(self, doc_ids: List[str]):
        """
        Demote documents from hot tier to cold tier.
        
        Note: Simplified implementation.
        """
        logger.info(f"Demoting {len(doc_ids)} documents to cold tier...")
        pass
    
    def _save_cold_tier(self):
        """Save cold tier index to SSD."""
        if self.cold_index is None:
            return
        
        ssd_dir = self.config.tiered_config.ssd_dir
        faiss.write_index(self.cold_index, os.path.join(ssd_dir, "cold_index.faiss"))
        with open(os.path.join(ssd_dir, "cold_docs.pkl"), "wb") as f:
            pickle.dump(self.cold_documents, f)
    
    def _load_cold_tier(self) -> bool:
        """Load cold tier index from SSD."""
        ssd_dir = self.config.tiered_config.ssd_dir
        index_path = os.path.join(ssd_dir, "cold_index.faiss")
        docs_path = os.path.join(ssd_dir, "cold_docs.pkl")
        
        if not os.path.exists(index_path):
            return False
        
        self.cold_index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            self.cold_documents = pickle.load(f)
        
        for doc in self.cold_documents:
            self.cold_ids.add(doc.get("id", ""))
        
        return True
    
    def save(self, name: str = "index"):
        """Save both tiers to disk."""
        # Save hot tier
        if self.hot_index is not None:
            faiss.write_index(self.hot_index, os.path.join(self.config.cache_dir, f"{name}_hot.faiss"))
            with open(os.path.join(self.config.cache_dir, f"{name}_hot_docs.pkl"), "wb") as f:
                pickle.dump(self.hot_documents, f)
        
        # Save cold tier (already persisted, but save metadata)
        self._save_cold_tier()
        
        # Save metadata
        meta = {
            "algorithm": self.config.algorithm.value,
            "dimension": self.dimension,
            "is_trained": self._is_trained,
            "tiered_enabled": self.config.tiered_config.enabled,
            "hot_count": len(self.hot_documents),
            "cold_count": len(self.cold_documents),
            "access_counts": self.access_counts
        }
        with open(os.path.join(self.config.cache_dir, f"{name}_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)
        
        logger.info(f"Saved index: {len(self.hot_documents)} hot, {len(self.cold_documents)} cold")
    
    def load(self, name: str = "index") -> bool:
        """Load both tiers from disk."""
        meta_path = os.path.join(self.config.cache_dir, f"{name}_meta.pkl")
        if not os.path.exists(meta_path):
            return False
        
        # Load metadata
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        
        self.dimension = meta["dimension"]
        self._is_trained = meta["is_trained"]
        self.access_counts = meta.get("access_counts", {})
        
        # Load hot tier
        hot_index_path = os.path.join(self.config.cache_dir, f"{name}_hot.faiss")
        if os.path.exists(hot_index_path):
            self.hot_index = faiss.read_index(hot_index_path)
            with open(os.path.join(self.config.cache_dir, f"{name}_hot_docs.pkl"), "rb") as f:
                self.hot_documents = pickle.load(f)
            for doc in self.hot_documents:
                self.hot_ids.add(doc.get("id", ""))
        
        # Load cold tier
        if meta.get("tiered_enabled", False):
            self._load_cold_tier()
        
        self._load_embedding_model()
        
        logger.info(f"Loaded index: {len(self.hot_documents)} hot, {len(self.cold_documents)} cold")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "hot_vectors": self.hot_index.ntotal if self.hot_index else 0,
            "cold_vectors": self.cold_index.ntotal if self.cold_index else 0,
            "hot_documents": len(self.hot_documents),
            "cold_documents": len(self.cold_documents),
            "total_vectors": (self.hot_index.ntotal if self.hot_index else 0) + 
                            (self.cold_index.ntotal if self.cold_index else 0),
            "tiered_enabled": self.config.tiered_config.enabled,
            "ram_capacity": self.config.tiered_config.ram_capacity
        }


# Backward-compatible wrapper
class FAISSRetriever(TieredFAISSRetriever):
    """Alias for backward compatibility."""
    pass


def create_retriever(
    algorithm: str = "hnsw",
    cache_dir: str = "./cache/faiss_index",
    tiered: bool = False,
    ram_capacity: int = 100000,
    ssd_dir: str = "./cache/ssd_index",
    **kwargs
) -> FAISSRetriever:
    """
    Factory function to create a FAISS retriever.
    
    Args:
        algorithm: Index algorithm ("hnsw", "ivf", or "flat")
        cache_dir: Directory for caching hot tier index
        tiered: Enable tiered storage (RAM + SSD)
        ram_capacity: Maximum vectors in RAM (hot tier)
        ssd_dir: Directory for SSD-based cold tier
        **kwargs: Additional configuration options
        
    Returns:
        Configured FAISSRetriever instance
    """
    algo_map = {"hnsw": IndexAlgorithm.HNSW, "ivf": IndexAlgorithm.IVF, "flat": IndexAlgorithm.FLAT}
    
    tiered_config = TieredStorageConfig(
        enabled=tiered,
        ram_capacity=ram_capacity,
        ssd_dir=ssd_dir
    )
    
    config = FAISSRetrieverConfig(
        algorithm=algo_map.get(algorithm.lower(), IndexAlgorithm.HNSW),
        cache_dir=cache_dir,
        tiered_config=tiered_config,
        batch_size=kwargs.get("batch_size", 128)
    )
    
    return FAISSRetriever(config)
