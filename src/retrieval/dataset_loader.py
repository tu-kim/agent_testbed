"""
Dataset Loader for RAG Pipeline

Loads and processes corpus (C4) and QA (HotpotQA) datasets.
Optimized for fast loading of pre-extracted JSON files using multiprocessing.
"""

import os
import logging
import json
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import time

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    # Corpus dataset (C4)
    corpus_name: str = "allenai/c4"
    corpus_subset: str = "en"
    corpus_split: str = "train"
    corpus_local_dir: Optional[str] = None
    
    # QA dataset (HotpotQA)
    qa_name: str = "hotpotqa/hotpot_qa"
    qa_subset: str = "fullwiki"
    qa_split: str = "validation"
    
    # Processing options
    min_text_length: int = 100
    max_text_length: int = 10000
    cache_dir: Optional[str] = None
    num_workers: int = os.cpu_count() or 4


def _process_single_json_file(args: Tuple[str, int, int]) -> List[Dict[str, Any]]:
    """
    Worker function to process a single JSON/JSONL file.
    This function runs in a separate process.
    """
    file_path, min_len, max_len = args
    docs = []
    
    try:
        file_size = os.path.getsize(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if it's JSON array or JSONL
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                # JSON array - load entire file
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
            else:
                # JSONL (one JSON object per line) - process line by line
                items = []
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            
            for item in items:
                text = item.get("text", "")
                if len(text) < min_len:
                    continue
                if len(text) > max_len:
                    text = text[:max_len]
                
                docs.append({
                    "text": text,
                    "url": item.get("url", ""),
                    "timestamp": item.get("timestamp", "")
                })
                
    except Exception as e:
        # Log error but don't crash the worker
        print(f"[Worker] Error processing {file_path}: {e}")
    
    return docs


def _process_file_batch(args: Tuple[List[str], int, int]) -> List[Dict[str, Any]]:
    """
    Process a batch of files in a single worker.
    More efficient for many small files.
    """
    file_paths, min_len, max_len = args
    all_docs = []
    
    for file_path in file_paths:
        docs = _process_single_json_file((file_path, min_len, max_len))
        all_docs.extend(docs)
    
    return all_docs


class C4CorpusLoader:
    """Loader for the C4 dataset from pre-extracted JSON files."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def _load_from_local_directory(self, local_dir: str) -> List[Dict[str, Any]]:
        """Load C4 corpus from local directory containing JSON/JSONL files."""
        logger.info(f"Loading C4 corpus from local directory: {local_dir}")
        
        if not os.path.exists(local_dir):
            raise FileNotFoundError(f"Local directory not found: {local_dir}")
        
        # Find all JSON files (including .json and .jsonl)
        json_files = []
        for pattern in ["*.json", "*.jsonl"]:
            json_files.extend(glob.glob(os.path.join(local_dir, pattern)))
            json_files.extend(glob.glob(os.path.join(local_dir, "**", pattern), recursive=True))
        
        # Remove duplicates and sort for consistent ordering
        json_files = sorted(list(set(json_files)))
        
        if not json_files:
            logger.error("No JSON files found in the directory")
            return []

        num_workers = self.config.num_workers
        logger.info(f"Found {len(json_files)} JSON files")
        logger.info(f"Using {num_workers} parallel workers")
        
        # Calculate total file size for progress estimation
        total_size = sum(os.path.getsize(f) for f in json_files)
        logger.info(f"Total data size: {total_size / (1024**3):.2f} GB")
        
        start_time = time.time()
        all_documents = []
        seen_hashes = set()
        processed_files = 0
        
        # Determine processing strategy based on file count
        if len(json_files) > num_workers * 10:
            # Many files: batch them for each worker
            batch_size = max(1, len(json_files) // num_workers)
            file_batches = [
                json_files[i:i + batch_size] 
                for i in range(0, len(json_files), batch_size)
            ]
            
            logger.info(f"Processing in {len(file_batches)} batches (batch size: ~{batch_size} files)")
            
            batch_args = [
                (batch, self.config.min_text_length, self.config.max_text_length)
                for batch in file_batches
            ]
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_file_batch, args): i for i, args in enumerate(batch_args)}
                
                with tqdm(total=len(file_batches), desc="Processing batches", unit="batch") as pbar:
                    for future in as_completed(futures):
                        try:
                            batch_docs = future.result()
                            for doc in batch_docs:
                                text_hash = hashlib.md5(doc["text"].encode()).hexdigest()
                                if text_hash not in seen_hashes:
                                    seen_hashes.add(text_hash)
                                    doc["id"] = f"c4_{len(all_documents)}"
                                    doc["source"] = "c4"
                                    all_documents.append(doc)
                            pbar.update(1)
                            pbar.set_postfix({"docs": len(all_documents)})
                        except Exception as e:
                            logger.error(f"Batch processing error: {e}")
        else:
            # Few files: process individually
            file_args = [
                (f, self.config.min_text_length, self.config.max_text_length)
                for f in json_files
            ]
            
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(_process_single_json_file, args): args[0] for args in file_args}
                
                with tqdm(total=len(json_files), desc="Processing files", unit="file") as pbar:
                    for future in as_completed(futures):
                        file_path = futures[future]
                        try:
                            file_docs = future.result()
                            for doc in file_docs:
                                text_hash = hashlib.md5(doc["text"].encode()).hexdigest()
                                if text_hash not in seen_hashes:
                                    seen_hashes.add(text_hash)
                                    doc["id"] = f"c4_{len(all_documents)}"
                                    doc["source"] = "c4"
                                    all_documents.append(doc)
                            processed_files += 1
                            pbar.update(1)
                            pbar.set_postfix({"docs": len(all_documents)})
                        except Exception as e:
                            logger.error(f"Error processing {file_path}: {e}")
        
        elapsed = time.time() - start_time
        docs_per_sec = len(all_documents) / elapsed if elapsed > 0 else 0
        mb_per_sec = (total_size / (1024**2)) / elapsed if elapsed > 0 else 0
        
        logger.info(f"Successfully loaded {len(all_documents)} unique C4 documents")
        logger.info(f"Processing time: {elapsed:.2f}s ({docs_per_sec:.0f} docs/s, {mb_per_sec:.1f} MB/s)")
        
        return all_documents
    
    def load(self) -> List[Dict[str, Any]]:
        if self.config.corpus_local_dir:
            return self._load_from_local_directory(self.config.corpus_local_dir)
        
        # HF streaming fallback
        logger.info(f"Streaming C4 from HuggingFace: {self.config.corpus_name}")
        dataset = load_dataset(self.config.corpus_name, self.config.corpus_subset, 
                               split=self.config.corpus_split, streaming=True, trust_remote_code=True)
        
        documents = []
        seen_hashes = set()
        for item in tqdm(dataset, desc="Streaming C4"):
            text = item.get("text", "")
            if len(text) < self.config.min_text_length:
                continue
            text = text[:self.config.max_text_length]
            
            h = hashlib.md5(text.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            
            documents.append({
                "id": f"c4_{len(documents)}",
                "text": text,
                "source": "c4",
                "url": item.get("url", ""),
                "timestamp": item.get("timestamp", "")
            })
        return documents


class HotpotQALoader:
    """Loader for HotpotQA dataset."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    def load(self, max_samples: Optional[int] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        logger.info(f"Loading HotpotQA: {self.config.qa_name}")
        dataset = load_dataset(self.config.qa_name, self.config.qa_subset, 
                               split=self.config.qa_split, trust_remote_code=True)
        
        questions = []
        supporting_docs = []
        seen_doc_ids = set()
        
        for i, item in enumerate(dataset):
            if max_samples and len(questions) >= max_samples:
                break
            
            questions.append({
                "id": item.get("id", f"hotpot_{i}"),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "supporting_facts": item.get("supporting_facts", {})
            })
            
            context = item.get("context", {})
            if isinstance(context, dict):
                for title, sents in zip(context.get("title", []), context.get("sentences", [])):
                    doc_id = hashlib.md5(title.encode()).hexdigest()[:8]
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        supporting_docs.append({
                            "id": f"hotpot_doc_{doc_id}",
                            "title": title,
                            "text": " ".join(sents) if isinstance(sents, list) else str(sents),
                            "source": "hotpotqa"
                        })
        return questions, supporting_docs


class DatasetManager:
    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()
        self.corpus_loader = C4CorpusLoader(self.config)
        self.qa_loader = HotpotQALoader(self.config)
    
    def load_corpus(self) -> List[Dict[str, Any]]:
        return self.corpus_loader.load()
    
    def get_evaluation_data(self, num_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        questions, _ = self.qa_loader.load(max_samples=num_samples)
        return [{
            "id": q["id"],
            "question": q["question"],
            "answer": q["answer"],
            "gold_docs": q.get("supporting_facts", {}).get("title", [])
        } for q in questions]


def create_dataset_manager(corpus_local_dir=None, cache_dir=None, num_workers=None, **kwargs) -> DatasetManager:
    config_kwargs = {"corpus_local_dir": corpus_local_dir, "cache_dir": cache_dir}
    if num_workers is not None:
        config_kwargs["num_workers"] = num_workers
    config_kwargs.update(kwargs)
    config = DatasetConfig(**config_kwargs)
    return DatasetManager(config)
