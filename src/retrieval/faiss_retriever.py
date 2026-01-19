"""
FAISS Vector Database for Dense Retrieval

Implements dense retrieval using FAISS with support for
HNSW and IVF indexing algorithms.
"""

import os
import logging
import pickle
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Install with: pip install faiss-cpu or faiss-gpu")

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
class FAISSRetrieverConfig:
    """Configuration for FAISS retriever."""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: str = "./cache/faiss_index"
    algorithm: IndexAlgorithm = IndexAlgorithm.HNSW
    hnsw_config: HNSWConfig = field(default_factory=HNSWConfig)
    ivf_config: IVFConfig = field(default_factory=IVFConfig)
    batch_size: int = 256
    normalize_embeddings: bool = True
    device: str = "auto"  # "auto", "cuda", "cpu"


class FAISSRetriever:
    """
    FAISS-based dense retriever with GPU-accelerated embeddings.
    """
    
    def __init__(self, config: FAISSRetrieverConfig):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is required. Install with: pip install faiss-cpu")
        
        self.config = config
        self.embedding_model: Optional[SentenceTransformer] = None
        self.dimension: int = 0
        self.index: Optional[faiss.Index] = None
        self.documents: List[Dict[str, Any]] = []
        self._is_trained: bool = False
        self._device: str = "cpu"
        
        os.makedirs(config.cache_dir, exist_ok=True)
    
    def _determine_device(self) -> str:
        """Determine the best available device for embeddings."""
        if self.config.device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        elif self.config.device == "cpu":
            return "cpu"
        else:  # auto
            if torch.cuda.is_available():
                return "cuda"
            return "cpu"
    
    def _load_embedding_model(self):
        """Load the embedding model onto the appropriate device."""
        if self.embedding_model is not None:
            return
        
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        
        # Determine device
        self._device = self._determine_device()
        
        if self._device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.info("Using CPU for embeddings")
        
        # Load model
        self.embedding_model = SentenceTransformer(
            self.config.embedding_model,
            device=self._device
        )
        
        # Verify device placement
        model_device = next(self.embedding_model.parameters()).device
        logger.info(f"Model loaded on device: {model_device}")
        
        # Get dimension
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self.dimension}")
    
    def _create_index(self, num_vectors: int = 0) -> faiss.Index:
        """Create a FAISS index based on configuration."""
        self._load_embedding_model()
        
        if self.config.algorithm == IndexAlgorithm.HNSW:
            index = faiss.IndexHNSWFlat(self.dimension, self.config.hnsw_config.M)
            index.hnsw.efConstruction = self.config.hnsw_config.ef_construction
            index.hnsw.efSearch = self.config.hnsw_config.ef_search
            logger.info(f"Created HNSW index (M={self.config.hnsw_config.M})")
        elif self.config.algorithm == IndexAlgorithm.IVF:
            quantizer = faiss.IndexFlatL2(self.dimension)
            nlist = min(int(4 * np.sqrt(num_vectors)), num_vectors // 10) if num_vectors > 100 else self.config.ivf_config.nlist
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            index.nprobe = self.config.ivf_config.nprobe
            logger.info(f"Created IVF index (nlist={nlist}, nprobe={index.nprobe})")
        else:
            index = faiss.IndexFlatL2(self.dimension)
            logger.info("Created Flat index")
        
        return index
    
    def _encode_texts(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """Encode texts to embeddings using GPU if available."""
        self._load_embedding_model()
        
        logger.info(f"Encoding {len(texts)} texts on {self._device}...")
        
        # Encode with explicit device usage
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            device=self._device  # Explicitly pass device
        )
        
        return embeddings.astype(np.float32)
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the index."""
        if not documents:
            return
        
        logger.info(f"Adding {len(documents)} documents to index...")
        
        # Extract texts
        texts = [doc.get("text", "") for doc in documents]
        
        # Encode
        start_time = time.time()
        embeddings = self._encode_texts(texts)
        encode_time = time.time() - start_time
        logger.info(f"Encoding completed in {encode_time:.2f}s ({len(texts)/encode_time:.0f} texts/s)")
        
        # Create index if needed
        if self.index is None:
            self.index = self._create_index(len(documents))
        
        # Train IVF if needed
        if self.config.algorithm == IndexAlgorithm.IVF and not self._is_trained:
            logger.info("Training IVF index...")
            train_start = time.time()
            self.index.train(embeddings)
            self._is_trained = True
            logger.info(f"IVF training completed in {time.time() - train_start:.2f}s")
        
        # Add to index
        logger.info("Adding vectors to FAISS index...")
        add_start = time.time()
        self.index.add(embeddings)
        logger.info(f"Added {len(embeddings)} vectors in {time.time() - add_start:.2f}s")
        
        # Store documents
        start_idx = len(self.documents)
        for i, doc in enumerate(documents):
            doc["_idx"] = start_idx + i
            if "id" not in doc:
                doc["id"] = f"doc_{start_idx + i}"
        self.documents.extend(documents)
        
        logger.info(f"Index now contains {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[float], float]:
        """
        Search for similar documents.
        
        Returns:
            Tuple of (documents, scores, retrieval_time_ms)
        """
        if self.index is None or self.index.ntotal == 0:
            return [], [], 0.0
        
        start_time = time.perf_counter()
        
        # Encode query
        query_embedding = self._encode_texts([query], show_progress=False)
        
        # Search
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(query_embedding, k)
        
        retrieval_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Gather results
        results = []
        scores = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
                scores.append(float(distances[0][i]))
        
        return results, scores, retrieval_time_ms
    
    def save(self, name: str = "index"):
        """Save index and documents to disk."""
        if self.index is None:
            logger.warning("No index to save")
            return
        
        index_path = os.path.join(self.config.cache_dir, f"{name}.faiss")
        docs_path = os.path.join(self.config.cache_dir, f"{name}_docs.pkl")
        meta_path = os.path.join(self.config.cache_dir, f"{name}_meta.pkl")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save documents
        with open(docs_path, "wb") as f:
            pickle.dump(self.documents, f)
        
        # Save metadata
        meta = {
            "algorithm": self.config.algorithm.value,
            "dimension": self.dimension,
            "is_trained": self._is_trained,
            "num_vectors": self.index.ntotal,
            "num_documents": len(self.documents)
        }
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        
        logger.info(f"Saved index ({self.index.ntotal} vectors) to {self.config.cache_dir}")
    
    def load(self, name: str = "index") -> bool:
        """Load index and documents from disk."""
        index_path = os.path.join(self.config.cache_dir, f"{name}.faiss")
        docs_path = os.path.join(self.config.cache_dir, f"{name}_docs.pkl")
        meta_path = os.path.join(self.config.cache_dir, f"{name}_meta.pkl")
        
        if not os.path.exists(index_path):
            return False
        
        # Load metadata
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.dimension = meta.get("dimension", 0)
            self._is_trained = meta.get("is_trained", False)
        
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load documents
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)
        
        # Load embedding model
        self._load_embedding_model()
        
        logger.info(f"Loaded index ({self.index.ntotal} vectors, {len(self.documents)} documents)")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "num_vectors": self.index.ntotal if self.index else 0,
            "num_documents": len(self.documents),
            "dimension": self.dimension,
            "algorithm": self.config.algorithm.value,
            "device": self._device,
            "is_trained": self._is_trained
        }


def create_retriever(
    algorithm: str = "hnsw",
    cache_dir: str = "./cache/faiss_index",
    batch_size: int = 256,
    device: str = "auto",
    **kwargs
) -> FAISSRetriever:
    """
    Factory function to create a FAISS retriever.
    
    Args:
        algorithm: Index algorithm ("hnsw", "ivf", or "flat")
        cache_dir: Directory for caching index
        batch_size: Batch size for encoding
        device: Device for embeddings ("auto", "cuda", "cpu")
        
    Returns:
        Configured FAISSRetriever instance
    """
    algo_map = {
        "hnsw": IndexAlgorithm.HNSW,
        "ivf": IndexAlgorithm.IVF,
        "flat": IndexAlgorithm.FLAT
    }
    
    config = FAISSRetrieverConfig(
        algorithm=algo_map.get(algorithm.lower(), IndexAlgorithm.HNSW),
        cache_dir=cache_dir,
        batch_size=batch_size,
        device=device
    )
    
    return FAISSRetriever(config)
