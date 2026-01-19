"""
IRCoT (Interleaving Retrieval with Chain-of-Thought) RAG Pipeline

Implements the IRCoT algorithm for multi-hop question answering
using ReAct-style prompting (Question → Retrieval → Reason/Answer loop).
"""

import logging
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from ..retrieval.faiss_retriever import FAISSRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== Data Classes ==============

@dataclass
class IRCoTConfig:
    """Configuration for IRCoT pipeline."""
    llm_endpoint: str = "http://localhost:8000"
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    max_tokens: int = 512
    temperature: float = 0.7
    max_iterations: int = 3
    retrieval_per_step: int = 3


@dataclass
class IRCoTResult:
    """Result from IRCoT pipeline execution."""
    answer: str
    context: List[str]
    num_iterations: int
    steps: List[Dict[str, Any]]
    retrieval_time_ms: float
    llm_time_ms: float
    total_time_ms: float


# ============== ReAct-Style Prompt Templates with Few-Shot Examples ==============

REACT_SYSTEM_PROMPT = """You are a helpful assistant that answers multi-hop questions by reasoning step-by-step.

You will be given a conversation history that follows this pattern:
- Question: The user's question
- Retrieval: Retrieved documents from the knowledge base
- Thought: Your reasoning about the retrieved information
- Action: Either [SEARCH: <query>] to retrieve more information, or [ANSWER: <final answer>] to provide the final answer

Rules:
1. Analyze the retrieved documents carefully
2. If you need more information to answer the question, use [SEARCH: <specific query>]
3. If you have enough information, use [ANSWER: <your final answer>]
4. Always explain your reasoning in the Thought section before taking an action
5. Be concise but thorough in your reasoning

Here are some examples of how to reason through multi-hop questions:

===== Example 1 =====
Question: Who was the director of the movie that featured the song "My Heart Will Go On"?

Retrieval 1:
  [1] Titanic (1997 film): Titanic is a 1997 American epic romance and disaster film directed by James Cameron. The film features the song "My Heart Will Go On" performed by Celine Dion.
  [2] My Heart Will Go On: "My Heart Will Go On" is a song recorded by Canadian singer Celine Dion. It serves as the main theme song to the 1997 film Titanic.

Thought 1: The retrieved documents clearly state that "My Heart Will Go On" was featured in the 1997 film Titanic, and that Titanic was directed by James Cameron. I have enough information to answer the question.
Action 1: [ANSWER: James Cameron]

===== Example 2 =====
Question: What is the capital of the country where the Eiffel Tower is located?

Retrieval 1:
  [1] Eiffel Tower: The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel.
  [2] Paris: Paris is the capital and most populous city of France, with an estimated population of 2,161,000 residents.

Thought 1: From the retrieved documents, I can see that the Eiffel Tower is located in Paris, France. The documents also confirm that Paris is the capital of France. I have all the information needed to answer.
Action 1: [ANSWER: Paris]

===== Example 3 =====
Question: What university did the founder of Microsoft graduate from?

Retrieval 1:
  [1] Microsoft: Microsoft Corporation is an American multinational technology corporation. It was founded by Bill Gates and Paul Allen on April 4, 1975.
  [2] Bill Gates: William Henry Gates III (born October 28, 1955) is an American business magnate, software developer, and philanthropist. He is best known as the co-founder of Microsoft Corporation.

Thought 1: I found that Microsoft was founded by Bill Gates and Paul Allen. However, I don't have information about which university Bill Gates graduated from. I need to search for more specific information about Bill Gates' education.
Action 1: [SEARCH: Bill Gates education university]

Retrieval 2:
  [1] Bill Gates education: Bill Gates attended Harvard University in 1973 but dropped out two years later to start Microsoft with Paul Allen. He was awarded an honorary degree from Harvard in 2007.
  [2] Harvard University: Harvard University is a private Ivy League research university in Cambridge, Massachusetts.

Thought 2: Now I have the information. Bill Gates attended Harvard University, though he dropped out before graduating. Since the question asks about the university the founder attended (not necessarily graduated from), Harvard University is the answer.
Action 2: [ANSWER: Harvard University]

===== Example 4 =====
Question: In which year was the company that created the iPhone founded?

Retrieval 1:
  [1] iPhone: The iPhone is a line of smartphones designed and marketed by Apple Inc. The first iPhone was announced by Steve Jobs on January 9, 2007.
  [2] Smartphone history: The first smartphone was IBM Simon, introduced in 1992.

Thought 1: I found that the iPhone was created by Apple Inc., but I don't have information about when Apple was founded. I need to search for Apple's founding year.
Action 1: [SEARCH: Apple Inc founding year history]

Retrieval 2:
  [1] Apple Inc.: Apple Inc. is an American multinational technology company. It was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976.
  [2] Steve Jobs: Steve Jobs co-founded Apple Computer in 1976 with Steve Wozniak and Ronald Wayne.

Thought 2: The documents clearly state that Apple Inc. was founded on April 1, 1976. Since Apple created the iPhone, the answer to the question is 1976.
Action 2: [ANSWER: 1976]

===== End of Examples =====

Now, apply this reasoning approach to answer the following question:"""


class IRCoTRAGPipeline:
    """
    IRCoT RAG Pipeline with ReAct-style prompting.
    
    The prompt is built incrementally in the following pattern:
    Question: <user question>
    Retrieval 1: <retrieved docs>
    Thought 1: <reasoning>
    Action 1: [SEARCH: <query>] or [ANSWER: <answer>]
    Retrieval 2: <more docs>
    Thought 2: <more reasoning>
    Action 2: ...
    """
    
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

    def _format_retrieval(self, documents: List[Dict[str, Any]], step: int) -> str:
        """Format retrieved documents for the prompt."""
        if not documents:
            return f"Retrieval {step}: No relevant documents found."
        
        doc_texts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', f'Document {i}')
            text = doc.get('text', '')[:500]  # Truncate long texts
            doc_texts.append(f"  [{i}] {title}: {text}")
        
        return f"Retrieval {step}:\n" + "\n".join(doc_texts)

    def _parse_action(self, output: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse the action from LLM output."""
        # Look for SEARCH action
        search_match = re.search(r'\[SEARCH:\s*(.+?)\]', output, re.IGNORECASE)
        if search_match:
            return search_match.group(1).strip(), None
        
        # Look for ANSWER action
        answer_match = re.search(r'\[ANSWER:\s*(.+?)\]', output, re.IGNORECASE | re.DOTALL)
        if answer_match:
            return None, answer_match.group(1).strip()
        
        return None, None

    def _build_prompt_history(self, question: str, history: List[Dict[str, Any]]) -> str:
        """Build the accumulated prompt history in ReAct format."""
        prompt_parts = [f"Question: {question}"]
        
        for entry in history:
            step = entry["step"]
            
            # Add retrieval results
            if entry.get("retrieval_text"):
                prompt_parts.append(entry["retrieval_text"])
            
            # Add thought and action from previous iterations
            if entry.get("thought"):
                prompt_parts.append(f"Thought {step}: {entry['thought']}")
            
            if entry.get("action"):
                prompt_parts.append(f"Action {step}: {entry['action']}")
        
        return "\n\n".join(prompt_parts)

    async def arun(
        self, 
        question: str, 
        max_iterations: int = None, 
        retrieval_per_step: int = None
    ) -> IRCoTResult:
        """
        Asynchronous execution of IRCoT pipeline with ReAct-style prompting.
        
        Args:
            question: The user's question
            max_iterations: Maximum number of retrieval-reasoning iterations
            retrieval_per_step: Number of documents to retrieve per step
            
        Returns:
            IRCoTResult dataclass with answer, context, and profiling info
        """
        max_iters = max_iterations or self.config.max_iterations
        top_k = retrieval_per_step or self.config.retrieval_per_step
        
        start_time = time.perf_counter()
        history: List[Dict[str, Any]] = []
        all_docs: List[Dict[str, Any]] = []
        reasoning_steps: List[Dict[str, Any]] = []
        total_retrieval_time = 0.0
        total_llm_time = 0.0
        
        current_query = question
        final_answer = None
        
        for iteration in range(1, max_iters + 1):
            step_entry = {"step": iteration}
            
            # ===== RETRIEVAL PHASE =====
            docs, scores, retrieval_time = self.retriever.search(current_query, top_k=top_k)
            total_retrieval_time += retrieval_time
            all_docs.extend(docs)
            
            # Format retrieval for prompt
            retrieval_text = self._format_retrieval(docs, iteration)
            step_entry["retrieval_text"] = retrieval_text
            step_entry["retrieved_docs"] = len(docs)
            step_entry["retrieval_time_ms"] = retrieval_time
            
            # ===== BUILD ACCUMULATED PROMPT =====
            # Add current retrieval to history for prompt building
            history.append(step_entry)
            
            accumulated_prompt = self._build_prompt_history(question, history)
            accumulated_prompt += f"\n\nThought {iteration}:"
            
            # ===== REASONING PHASE (LLM Call) =====
            llm_start = time.perf_counter()
            
            messages = [
                SystemMessage(content=REACT_SYSTEM_PROMPT),
                HumanMessage(content=accumulated_prompt)
            ]
            
            response = await self.llm.ainvoke(messages)
            llm_time = (time.perf_counter() - llm_start) * 1000
            total_llm_time += llm_time
            
            output = response.content.strip()
            
            # Parse thought and action from output
            # The output should contain reasoning followed by an action
            thought = output
            search_query, answer = self._parse_action(output)
            
            # Determine the action text
            if search_query:
                action_text = f"[SEARCH: {search_query}]"
            elif answer:
                action_text = f"[ANSWER: {answer}]"
            else:
                action_text = "[SEARCH: " + question + "]"  # Default to searching original question
                search_query = question
            
            # Update history entry with thought and action
            step_entry["thought"] = thought
            step_entry["action"] = action_text
            step_entry["llm_time_ms"] = llm_time
            
            # Record reasoning step
            reasoning_steps.append({
                "step": iteration,
                "query": current_query,
                "retrieved_docs": len(docs),
                "reasoning": thought,
                "action": action_text,
                "search_query": search_query,
                "retrieval_time_ms": retrieval_time,
                "llm_time_ms": llm_time
            })
            
            # ===== CHECK FOR ANSWER =====
            if answer:
                final_answer = answer
                break
            
            # Update query for next iteration
            current_query = search_query or question
        
        # If no answer was found, generate a summary answer
        if final_answer is None:
            final_answer = "Could not find a definitive answer after exhausting all iterations."
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        # Deduplicate context
        seen_texts = set()
        unique_context = []
        for doc in all_docs:
            text = doc.get("text", "")
            if text and text not in seen_texts:
                seen_texts.add(text)
                unique_context.append(text)
        
        return IRCoTResult(
            answer=final_answer,
            context=unique_context,
            num_iterations=len(reasoning_steps),
            steps=reasoning_steps,
            retrieval_time_ms=total_retrieval_time,
            llm_time_ms=total_llm_time,
            total_time_ms=total_time
        )

    def run(
        self, 
        question: str, 
        max_iterations: int = None, 
        retrieval_per_step: int = None
    ) -> IRCoTResult:
        """Synchronous wrapper for arun."""
        import asyncio
        return asyncio.run(self.arun(question, max_iterations, retrieval_per_step))
