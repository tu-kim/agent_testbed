"""
vLLM Prefill Worker

This worker handles the prefill phase of the LLM inference,
processing the input prompt and generating the initial KV cache.

Compatible with vLLM v0.14+ (V1 engine architecture).
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# vLLM v0.14+ uses V1 engine (AsyncLLM instead of AsyncLLMEngine)
try:
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import RequestOutputKind
    
    # Try V1 engine first (vLLM v0.14+)
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        VLLM_V1 = True
        VLLM_AVAILABLE = True
        logging.info("Using vLLM V1 engine (AsyncLLM)")
    except ImportError:
        # Fallback to legacy engine
        from vllm.engine.async_llm_engine import AsyncLLMEngine as AsyncLLM
        VLLM_V1 = False
        VLLM_AVAILABLE = True
        logging.info("Using vLLM legacy engine (AsyncLLMEngine)")
        
except ImportError:
    VLLM_AVAILABLE = False
    VLLM_V1 = False
    logging.warning("vLLM not available. Running in mock mode.")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PrefillRequest(BaseModel):
    """Request model for prefill operation."""
    request_id: str
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop_sequences: Optional[List[str]] = None


class PrefillResponse(BaseModel):
    """Response model for prefill operation."""
    request_id: str
    prefill_tokens: int
    kv_cache_id: str
    prefill_time_ms: float
    status: str


class KVCacheMetadata(BaseModel):
    """Metadata for KV cache transfer."""
    request_id: str
    kv_cache_id: str
    num_layers: int
    num_heads: int
    head_dim: int
    seq_len: int
    dtype: str


def resolve_model_path(model_name_or_path: str) -> str:
    """
    Resolve model name or path.
    
    If the input is a local directory path, return it as-is.
    If it's a HuggingFace model ID, return it as-is.
    
    Args:
        model_name_or_path: Model name (e.g., 'meta-llama/Llama-2-7b-chat-hf')
                           or local path (e.g., '/path/to/model')
    
    Returns:
        Resolved model path or name
    """
    # Check if it's a local directory
    if os.path.isdir(model_name_or_path):
        abs_path = os.path.abspath(model_name_or_path)
        logger.info(f"Using local model directory: {abs_path}")
        return abs_path
    
    # Check if it's a relative path that exists
    if os.path.exists(model_name_or_path):
        abs_path = os.path.abspath(model_name_or_path)
        logger.info(f"Using local model path: {abs_path}")
        return abs_path
    
    # Assume it's a HuggingFace model ID
    logger.info(f"Using HuggingFace model: {model_name_or_path}")
    return model_name_or_path


@dataclass
class PrefillWorkerConfig:
    """Configuration for prefill worker."""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    host: str = "0.0.0.0"
    port: int = 8001
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_model_len: int = 4096
    dtype: str = "auto"
    trust_remote_code: bool = True
    tokenizer: Optional[str] = None  # Optional separate tokenizer path
    enforce_eager: bool = False  # Set True for faster startup


class PrefillWorker:
    """
    Prefill Worker for distributed vLLM inference.
    
    Handles the prefill phase which processes input prompts
    and generates KV cache for decode workers.
    
    Compatible with vLLM v0.14+ (V1 engine).
    """
    
    def __init__(self, config: PrefillWorkerConfig):
        self.config = config
        self.engine: Optional[AsyncLLM] = None
        self.kv_cache_store: Dict[str, Any] = {}
        self.app = FastAPI(title="vLLM Prefill Worker")
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/prefill", response_model=PrefillResponse)
        async def prefill(request: PrefillRequest):
            return await self.process_prefill(request)
        
        @self.app.get("/kv_cache/{kv_cache_id}")
        async def get_kv_cache(kv_cache_id: str):
            return await self.get_kv_cache_metadata(kv_cache_id)
        
        @self.app.delete("/kv_cache/{kv_cache_id}")
        async def delete_kv_cache(kv_cache_id: str):
            return await self.release_kv_cache(kv_cache_id)
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy", 
                "worker_type": "prefill",
                "vllm_version": "v1" if VLLM_V1 else "legacy"
            }
        
        @self.app.get("/stats")
        async def get_stats():
            return {
                "active_kv_caches": len(self.kv_cache_store),
                "model_name": self.config.model_name,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "vllm_v1_engine": VLLM_V1
            }
    
    async def initialize(self):
        """Initialize the vLLM engine."""
        if VLLM_AVAILABLE:
            try:
                # Resolve model path (local directory or HuggingFace ID)
                model_path = resolve_model_path(self.config.model_name)
                
                # Determine tokenizer path
                tokenizer_path = self.config.tokenizer
                if tokenizer_path is None and os.path.isdir(model_path):
                    # For local models, use the same path for tokenizer
                    tokenizer_path = model_path
                
                engine_args_dict = {
                    "model": model_path,
                    "tensor_parallel_size": self.config.tensor_parallel_size,
                    "gpu_memory_utilization": self.config.gpu_memory_utilization,
                    "max_num_seqs": self.config.max_num_seqs,
                    "max_model_len": self.config.max_model_len,
                    "dtype": self.config.dtype,
                    "trust_remote_code": self.config.trust_remote_code,
                    "enforce_eager": self.config.enforce_eager,
                }
                
                if tokenizer_path:
                    engine_args_dict["tokenizer"] = tokenizer_path
                
                engine_args = AsyncEngineArgs(**engine_args_dict)
                self.engine = AsyncLLM.from_engine_args(engine_args)
                logger.info(f"Initialized vLLM {'V1' if VLLM_V1 else 'legacy'} engine with model: {model_path}")
            except Exception as e:
                logger.error(f"Failed to initialize vLLM engine: {e}")
                raise
        else:
            logger.warning("Running in mock mode without vLLM")
    
    async def process_prefill(self, request: PrefillRequest) -> PrefillResponse:
        """
        Process prefill request.
        
        Args:
            request: PrefillRequest containing prompt and parameters
            
        Returns:
            PrefillResponse with prefill results and KV cache ID
        """
        start_time = time.perf_counter()
        kv_cache_id = str(uuid.uuid4())
        
        try:
            if VLLM_AVAILABLE and self.engine is not None:
                # Real vLLM prefill
                sampling_params_kwargs = {
                    "max_tokens": 1,  # Only prefill, minimal decode
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                }
                
                if request.stop_sequences:
                    sampling_params_kwargs["stop"] = request.stop_sequences
                
                # For V1 engine, use DELTA output kind for streaming
                if VLLM_V1:
                    try:
                        sampling_params_kwargs["output_kind"] = RequestOutputKind.DELTA
                    except:
                        pass  # output_kind not available in this version
                
                sampling_params = SamplingParams(**sampling_params_kwargs)
                
                # Generate a unique request ID for this prefill operation
                internal_request_id = f"prefill_{request.request_id}_{uuid.uuid4().hex[:8]}"
                
                prefill_tokens = 0
                
                try:
                    # vLLM v0.14+ (V1 engine) API:
                    # engine.generate(request_id=..., prompt=..., sampling_params=...)
                    results_generator = self.engine.generate(
                        request_id=internal_request_id,
                        prompt=request.prompt,
                        sampling_params=sampling_params
                    )
                    
                    async for result in results_generator:
                        # Get prompt token count from the result
                        if hasattr(result, 'prompt_token_ids') and result.prompt_token_ids:
                            prefill_tokens = len(result.prompt_token_ids)
                        elif hasattr(result, 'outputs') and len(result.outputs) > 0:
                            # Try to get from outputs
                            output = result.outputs[0]
                            if hasattr(output, 'token_ids'):
                                prefill_tokens = max(prefill_tokens, 1)
                        
                        # Check if finished
                        if hasattr(result, 'finished') and result.finished:
                            break
                        
                        # For prefill, we only need the first result
                        break
                        
                except Exception as e:
                    logger.error(f"vLLM generate error: {e}", exc_info=True)
                    # Fallback to estimation based on prompt length
                    prefill_tokens = len(request.prompt.split()) * 2
                
                # Store KV cache metadata
                self.kv_cache_store[kv_cache_id] = {
                    "request_id": request.request_id,
                    "internal_request_id": internal_request_id,
                    "prompt": request.prompt,
                    "prefill_tokens": prefill_tokens,
                    "sampling_params": {
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "stop_sequences": request.stop_sequences
                    },
                    "created_at": time.time()
                }
            else:
                # Mock mode
                prefill_tokens = len(request.prompt.split()) * 2  # Rough estimate
                self.kv_cache_store[kv_cache_id] = {
                    "request_id": request.request_id,
                    "prompt": request.prompt,
                    "prefill_tokens": prefill_tokens,
                    "sampling_params": {
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "stop_sequences": request.stop_sequences
                    },
                    "created_at": time.time()
                }
            
            prefill_time_ms = (time.perf_counter() - start_time) * 1000
            
            return PrefillResponse(
                request_id=request.request_id,
                prefill_tokens=prefill_tokens,
                kv_cache_id=kv_cache_id,
                prefill_time_ms=prefill_time_ms,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Prefill error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_kv_cache_metadata(self, kv_cache_id: str) -> Dict[str, Any]:
        """Get KV cache metadata for transfer to decode worker."""
        if kv_cache_id not in self.kv_cache_store:
            raise HTTPException(status_code=404, detail="KV cache not found")
        return self.kv_cache_store[kv_cache_id]
    
    async def release_kv_cache(self, kv_cache_id: str) -> Dict[str, str]:
        """Release KV cache after decode is complete."""
        if kv_cache_id in self.kv_cache_store:
            del self.kv_cache_store[kv_cache_id]
            return {"status": "released", "kv_cache_id": kv_cache_id}
        return {"status": "not_found", "kv_cache_id": kv_cache_id}
    
    def shutdown(self):
        """Shutdown the engine gracefully."""
        if self.engine is not None:
            try:
                if hasattr(self.engine, 'shutdown'):
                    self.engine.shutdown()
                logger.info("Engine shutdown complete")
            except Exception as e:
                logger.warning(f"Error during engine shutdown: {e}")
    
    def run(self):
        """Run the prefill worker server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )


def create_prefill_worker(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    host: str = "0.0.0.0",
    port: int = 8001,
    **kwargs
) -> PrefillWorker:
    """Factory function to create a prefill worker."""
    config = PrefillWorkerConfig(
        model_name=model_name,
        host=host,
        port=port,
        **kwargs
    )
    return PrefillWorker(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Prefill Worker")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Model name (HuggingFace ID) or local directory path")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer path (optional, defaults to model path)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Enforce eager execution for faster startup")
    
    args = parser.parse_args()
    
    worker = create_prefill_worker(
        model_name=args.model,
        host=args.host,
        port=args.port,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
        tokenizer=args.tokenizer,
        enforce_eager=args.enforce_eager
    )
    
    # Initialize and run
    asyncio.run(worker.initialize())
    try:
        worker.run()
    finally:
        worker.shutdown()
