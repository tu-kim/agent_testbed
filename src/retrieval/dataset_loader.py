"""
Dataset Loader for RAG Pipeline

Loads and processes HotpotQA dataset for both corpus indexing and QA evaluation.
Supports train/validation split selection for corpus building.
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    # HotpotQA dataset configuration
    qa_name: str = "hotpotqa/hotpot_qa"
    qa_subset: str = "distractor"
    
    # Split for corpus building (train or validation)
    corpus_split: str = "train"
    
    # Split for evaluation queries (typically validation)
    eval_split: str = "validation"
    
    # Processing options
    min_text_length: int = 50
    max_text_length: int = 10000
    cache_dir: Optional[str] = None


class HotpotQACorpusLoader:
    """
    Loader for building corpus from HotpotQA context documents.
    
    Extracts all context passages from HotpotQA to build the retrieval corpus.
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def load(self, split: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load corpus documents from HotpotQA context.
        
        Args:
            split: Dataset split to use ('train' or 'validation'). 
                   Defaults to config.corpus_split.
        
        Returns:
            List of documents with id, title, text, and source fields.
        """
        split = split or self.config.corpus_split
        logger.info(f"Loading HotpotQA corpus from split: {split}")
        logger.info(f"Dataset: {self.config.qa_name} (subset: {self.config.qa_subset})")
        
        dataset = load_dataset(
            self.config.qa_name,
            name=self.config.qa_subset,
            split=split,
            trust_remote_code=True
        )
        
        documents = []
        seen_doc_hashes = set()
        
        for item in tqdm(dataset, desc=f"Extracting corpus from {split}"):
            context = item.get("context", {})
            
            if isinstance(context, dict):
                titles = context.get("title", [])
                sentences_list = context.get("sentences", [])
                
                for title, sentences in zip(titles, sentences_list):
                    # Combine sentences into full text
                    if isinstance(sentences, list):
                        text = " ".join(sentences)
                    else:
                        text = str(sentences)
                    
                    # Skip too short documents
                    if len(text) < self.config.min_text_length:
                        continue
                    
                    # Truncate if too long
                    if len(text) > self.config.max_text_length:
                        text = text[:self.config.max_text_length]
                    
                    # Deduplicate by content hash
                    doc_hash = hashlib.md5(text.encode()).hexdigest()
                    if doc_hash in seen_doc_hashes:
                        continue
                    seen_doc_hashes.add(doc_hash)
                    
                    documents.append({
                        "id": f"hotpot_doc_{len(documents)}",
                        "title": title,
                        "text": text,
                        "source": "hotpotqa",
                        "split": split
                    })
        
        logger.info(f"Loaded {len(documents)} unique documents from HotpotQA {split} split")
        return documents


class HotpotQAQueryLoader:
    """
    Loader for HotpotQA questions for evaluation.
    """
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def load(self, split: Optional[str] = None, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load questions from HotpotQA for evaluation.
        
        Args:
            split: Dataset split to use. Defaults to config.eval_split.
            max_samples: Maximum number of samples to load.
        
        Returns:
            List of question dictionaries with id, question, answer, and supporting_facts.
        """
        split = split or self.config.eval_split
        logger.info(f"Loading HotpotQA questions from split: {split}")
        
        dataset = load_dataset(
            self.config.qa_name,
            name=self.config.qa_subset,
            split=split,
            trust_remote_code=True
        )
        
        questions = []
        
        for i, item in enumerate(tqdm(dataset, desc=f"Loading questions from {split}")):
            if max_samples and len(questions) >= max_samples:
                break
            
            # Extract supporting fact titles for evaluation
            supporting_facts = item.get("supporting_facts", {})
            gold_titles = []
            if isinstance(supporting_facts, dict):
                gold_titles = supporting_facts.get("title", [])
            
            questions.append({
                "id": item.get("id", f"hotpot_q_{i}"),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "type": item.get("type", ""),
                "level": item.get("level", ""),
                "gold_titles": list(set(gold_titles)),  # Unique titles
                "supporting_facts": supporting_facts
            })
        
        logger.info(f"Loaded {len(questions)} questions from HotpotQA {split} split")
        return questions


class DatasetManager:
    """
    Unified manager for HotpotQA dataset operations.
    
    Handles both corpus loading (for indexing) and query loading (for evaluation).
    """
    
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.corpus_loader = HotpotQACorpusLoader(self.config)
        self.query_loader = HotpotQAQueryLoader(self.config)
    
    def load_corpus(self, split: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Load corpus documents for indexing.
        
        Args:
            split: 'train' or 'validation'. Defaults to config.corpus_split.
        
        Returns:
            List of documents for FAISS indexing.
        """
        return self.corpus_loader.load(split=split)
    
    def load_queries(self, split: Optional[str] = None, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load queries for evaluation/profiling.
        
        Args:
            split: 'train' or 'validation'. Defaults to config.eval_split.
            max_samples: Maximum number of queries to load.
        
        Returns:
            List of question dictionaries.
        """
        return self.query_loader.load(split=split, max_samples=max_samples)
    
    def get_evaluation_data(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get evaluation data in a simplified format.
        
        Args:
            num_samples: Maximum number of samples.
        
        Returns:
            List of evaluation items with question, answer, and gold_titles.
        """
        questions = self.load_queries(max_samples=num_samples)
        return [{
            "id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            "gold_titles": q.get("gold_titles", []),
            "type": q.get("type", ""),
            "level": q.get("level", "")
        } for q in questions]


def create_dataset_manager(
    corpus_split: str = "train",
    eval_split: str = "validation",
    qa_subset: str = "distractor",
    cache_dir: Optional[str] = None,
    **kwargs
) -> DatasetManager:
    """
    Factory function to create a DatasetManager.
    
    Args:
        corpus_split: Split to use for corpus building ('train' or 'validation').
        eval_split: Split to use for evaluation queries.
        qa_subset: HotpotQA subset ('distractor' or 'fullwiki').
        cache_dir: Cache directory for datasets.
    
    Returns:
        Configured DatasetManager instance.
    """
    config = DatasetConfig(
        corpus_split=corpus_split,
        eval_split=eval_split,
        qa_subset=qa_subset,
        cache_dir=cache_dir,
        **kwargs
    )
    return DatasetManager(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test HotpotQA Dataset Loader")
    parser.add_argument("--corpus-split", type=str, default="train",
                        choices=["train", "validation"],
                        help="Split to use for corpus")
    parser.add_argument("--eval-split", type=str, default="validation",
                        choices=["train", "validation"],
                        help="Split to use for evaluation")
    parser.add_argument("--max-queries", type=int, default=10,
                        help="Maximum queries to load for testing")
    
    args = parser.parse_args()
    
    manager = create_dataset_manager(
        corpus_split=args.corpus_split,
        eval_split=args.eval_split
    )
    
    # Test corpus loading
    print("\n=== Loading Corpus ===")
    corpus = manager.load_corpus()
    print(f"Corpus size: {len(corpus)} documents")
    if corpus:
        print(f"Sample document: {corpus[0]['title'][:50]}...")
    
    # Test query loading
    print("\n=== Loading Queries ===")
    queries = manager.load_queries(max_samples=args.max_queries)
    print(f"Queries loaded: {len(queries)}")
    if queries:
        print(f"Sample question: {queries[0]['question']}")
        print(f"Sample answer: {queries[0]['answer']}")
