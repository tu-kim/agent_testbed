"""
Tests for FAISS Retriever
"""

import pytest
import tempfile
import os

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.retrieval.faiss_retriever import (
    FAISSRetriever,
    FAISSRetrieverConfig,
    IndexAlgorithm,
    HNSWConfig,
    IVFConfig,
    TextChunker,
    create_retriever
)


class TestTextChunker:
    """Tests for TextChunker class."""
    
    def test_chunk_text_basic(self):
        """Test basic text chunking."""
        chunker = TextChunker(chunk_size=10, chunk_overlap=2)
        text = " ".join([f"word{i}" for i in range(25)])
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) > 0
        assert all("text" in chunk for chunk in chunks)
        assert all(chunk["doc_id"] == "doc1" for chunk in chunks)
    
    def test_chunk_text_empty(self):
        """Test chunking empty text."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("", "doc1")
        
        assert len(chunks) == 0
    
    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=10)
        text = "This is a short text."
        
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) == 1
        assert chunks[0]["text"] == text


class TestFAISSRetriever:
    """Tests for FAISSRetriever class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {"id": "1", "text": "Paris is the capital of France. It is known for the Eiffel Tower."},
            {"id": "2", "text": "Machine learning is a subset of artificial intelligence."},
            {"id": "3", "text": "Python is a popular programming language for data science."},
            {"id": "4", "text": "The Eiffel Tower was built in 1889 for the World's Fair."},
            {"id": "5", "text": "Deep learning uses neural networks with many layers."},
        ]
    
    def test_create_retriever_hnsw(self, temp_cache_dir):
        """Test creating HNSW retriever."""
        retriever = create_retriever(
            algorithm="hnsw",
            cache_dir=temp_cache_dir
        )
        
        assert retriever is not None
        assert retriever.config.algorithm == IndexAlgorithm.HNSW
    
    def test_create_retriever_ivf(self, temp_cache_dir):
        """Test creating IVF retriever."""
        retriever = create_retriever(
            algorithm="ivf",
            cache_dir=temp_cache_dir
        )
        
        assert retriever is not None
        assert retriever.config.algorithm == IndexAlgorithm.IVF
    
    def test_add_documents(self, temp_cache_dir, sample_documents):
        """Test adding documents to index."""
        retriever = create_retriever(
            algorithm="hnsw",
            cache_dir=temp_cache_dir
        )
        
        retriever.add_documents(sample_documents)
        
        assert retriever.index is not None
        assert retriever.index.ntotal > 0
        assert len(retriever.documents) > 0
    
    def test_search(self, temp_cache_dir, sample_documents):
        """Test searching documents."""
        retriever = create_retriever(
            algorithm="hnsw",
            cache_dir=temp_cache_dir
        )
        retriever.add_documents(sample_documents)
        
        results, scores, time_ms = retriever.search("What is machine learning?", top_k=2)
        
        assert len(results) == 2
        assert len(scores) == 2
        assert time_ms > 0
    
    def test_save_and_load(self, temp_cache_dir, sample_documents):
        """Test saving and loading index."""
        # Create and save
        retriever1 = create_retriever(
            algorithm="hnsw",
            cache_dir=temp_cache_dir
        )
        retriever1.add_documents(sample_documents)
        retriever1.save("test_index")
        
        original_count = retriever1.index.ntotal
        
        # Load in new retriever
        retriever2 = create_retriever(
            algorithm="hnsw",
            cache_dir=temp_cache_dir
        )
        loaded = retriever2.load("test_index")
        
        assert loaded is True
        assert retriever2.index.ntotal == original_count
    
    def test_batch_search(self, temp_cache_dir, sample_documents):
        """Test batch searching."""
        retriever = create_retriever(
            algorithm="hnsw",
            cache_dir=temp_cache_dir
        )
        retriever.add_documents(sample_documents)
        
        queries = ["machine learning", "Eiffel Tower", "Python programming"]
        results = retriever.batch_search(queries, top_k=2)
        
        assert len(results) == 3
        for docs, scores, time_ms in results:
            assert len(docs) <= 2
            assert len(scores) <= 2
    
    def test_get_stats(self, temp_cache_dir, sample_documents):
        """Test getting index statistics."""
        retriever = create_retriever(
            algorithm="hnsw",
            cache_dir=temp_cache_dir
        )
        retriever.add_documents(sample_documents)
        
        stats = retriever.get_stats()
        
        assert "algorithm" in stats
        assert "num_vectors" in stats
        assert stats["num_vectors"] > 0


class TestIVFRetriever:
    """Tests specific to IVF retriever."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def large_documents(self):
        """Generate larger document set for IVF testing."""
        return [
            {"id": str(i), "text": f"Document {i} with some content about topic {i % 10}."}
            for i in range(100)
        ]
    
    def test_ivf_training(self, temp_cache_dir, large_documents):
        """Test IVF index training."""
        retriever = create_retriever(
            algorithm="ivf",
            cache_dir=temp_cache_dir,
            ivf_nlist=10
        )
        
        retriever.add_documents(large_documents)
        
        assert retriever._is_trained is True
        assert retriever.index.ntotal > 0
    
    def test_ivf_search(self, temp_cache_dir, large_documents):
        """Test IVF search."""
        retriever = create_retriever(
            algorithm="ivf",
            cache_dir=temp_cache_dir,
            ivf_nlist=10,
            ivf_nprobe=5
        )
        retriever.add_documents(large_documents)
        
        results, scores, time_ms = retriever.search("topic 5", top_k=5)
        
        assert len(results) <= 5
        assert time_ms > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
