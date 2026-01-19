"""
IRCoT (Interleaving Retrieval with Chain-of-Thought) RAG Pipeline

Implements the IRCoT algorithm for multi-hop question answering.
Optimized with asynchronous execution support.
"""

import logging
import time
import re
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..retrieval.faiss_retriever import FAISSRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============== Prompt Templates ==============

IRCOT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions by reasoning step-by-step.
When you need to search for more information, output: [SEARCH: <query>]
When you have enough information to answer, output: [ANSWER: <your answer>]
Always explain your reasoning before giving the final answer."""

IRCOT_COT_PROMPT = """Question: {question}
Retrieved Context:
{context}
Previous reasoning steps:
{previous_reasoning}
Based on the context and your previous reasoning, continue thinking.
If you need more information, use [SEARCH: <query>]. If you can answer, use [ANSWER: <answer>].
Your reasoning:"""

# ============== Data Classes ==============

@dataclass
class IRCoTConfig:
    llm_endpoint: str = "http://localhost:8000"
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    max_tokens: int = 512
    temperature: float = 0.7
    max_iterations: int = 3
    retrieval_per_step: int = 3

class IRCoTRAGPipeline:
    def __init__(self, retriever: FAISSRetriever, config: IRCoTConfig):
        self.retriever = retriever
        self.config = config
        self.llm = ChatOpenAI(
            base_url=f"{self.config.llm_endpoint}/v1",
            api_key="not-needed",
            model=self.config.model_name,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        return "\n".join([f"[{doc.get('title', 'Doc')}] {doc.get('text', '')}" for doc in documents])

    def _parse_output(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        search_match = re.search(r'\[SEARCH:\s*(.+?)\]', output, re.IGNORECASE)
        if search_match: return search_match.group(1).strip(), None
        answer_match = re.search(r'\[ANSWER:\s*(.+?)\]', output, re.IGNORECASE | re.DOTALL)
        if answer_match: return None, answer_match.group(1).strip()
        return None, None

    async def arun(self, question: str, max_iterations: int = None, retrieval_per_step: int = None) -> Dict[str, Any]:
        """Asynchronous execution of IRCoT pipeline."""
        max_iters = max_iterations or self.config.max_iterations
        top_k = retrieval_per_step or self.config.retrieval_per_step
        
        start_time = time.perf_counter()
        reasoning_steps = []
        all_docs = []
        total_retrieval_time = 0.0
        total_llm_time = 0.0
        
        current_query = question
        previous_reasoning = ""
        
        for i in range(max_iters):
            # Retrieval
            docs, _, r_time = self.retriever.search(current_query, top_k=top_k)
            total_retrieval_time += r_time
            all_docs.extend(docs)
            
            # Reasoning
            context_str = self._format_context(all_docs)
            prompt = IRCOT_COT_PROMPT.format(
                question=question,
                context=context_str,
                previous_reasoning=previous_reasoning
            )
            
            llm_start = time.perf_counter()
            messages = [SystemMessage(content=IRCOT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
            response = await self.llm.ainvoke(messages)
            llm_time = (time.perf_counter() - llm_start) * 1000
            total_llm_time += llm_time
            
            output = response.content
            previous_reasoning += f"\nStep {i+1}: {output}"
            search_query, answer = self._parse_output(output)
            
            reasoning_steps.append({
                "step": i + 1,
                "reasoning": output,
                "search_query": search_query,
                "llm_time_ms": llm_time
            })
            
            if answer:
                return {
                    "answer": answer,
                    "context": [d["text"] for d in all_docs],
                    "num_iterations": i + 1,
                    "steps": reasoning_steps,
                    "profile": {"retrieval_time_ms": total_retrieval_time, "llm_time_ms": total_llm_time}
                }
            
            current_query = search_query or question

        return {
            "answer": "Could not find a definitive answer.",
            "context": [d["text"] for d in all_docs],
            "num_iterations": max_iters,
            "steps": reasoning_steps,
            "profile": {"retrieval_time_ms": total_retrieval_time, "llm_time_ms": total_llm_time}
        }
