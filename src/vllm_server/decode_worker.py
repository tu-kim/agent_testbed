"""
vLLM Decode Worker

This worker handles the decode phase of the LLM inference,
generating tokens autoregressively using the KV cache from prefill.

Compatible with vLLM v0.14+ (V1 engine architecture).
"""

import asyncio
import logging
import os
from typing import Dict, Any, Optional, List, AsyncGenerator
from dataclasses import dataclass
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import httpx

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


class DecodeRequest(BaseModel):
    """Request model for decode operation."""
    request_id: str
    kv_cache_id: str
    prefill_worker_url: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


class DecodeResponse(BaseModel):
    """Response model for decode operation."""
    request_id: str
    generated_text: str
    generated_tokens: int
    decode_time_ms: float
    tokens_per_second: float
    status: str


class StreamToken(BaseModel):
    """Model for streaming token response."""
    token: str
    token_id: int
    finish_reason: Optional[str] = None


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
class DecodeWorkerConfig:
    """Configuration for decode worker."""
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    host: str = "0.0.0.0"
    port: int = 8002
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_num_seqs: int = 256
    max_model_len: int = 4096
    dtype: str = "auto"
    trust_remote_code: bool = True
    tokenizer: Optional[str] = None  # Optional separate tokenizer path
    enforce_eager: bool = False  # Set True for faster startup


class DecodeWorker:
    """
    Decode Worker for distributed vLLM inference.
    
    Handles the decode phase which generates tokens autoregressively
    using KV cache from prefill workers.
    
    Compatible with vLLM v0.14+ (V1 engine).
    """
    
    def __init__(self, config: DecodeWorkerConfig):
        self.config = config
        self.engine: Optional[AsyncLLM] = None
        self.active_requests: Dict[str, Any] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.app = FastAPI(title="vLLM Decode Worker")
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/decode", response_model=DecodeResponse)
        async def decode(request: DecodeRequest):
            return await self.process_decode(request)
        
        @self.app.post("/decode/stream")
        async def decode_stream(request: DecodeRequest):
            return StreamingResponse(
                self.process_decode_stream(request),
                media_type="text/event-stream"
            )
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy", 
                "worker_type": "decode",
                "vllm_version": "v1" if VLLM_V1 else "legacy"
            }
        
        @self.app.get("/stats")
        async def get_stats():
            return {
                "active_requests": len(self.active_requests),
                "model_name": self.config.model_name,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "vllm_v1_engine": VLLM_V1
            }
        
        @self.app.on_event("startup")
        async def startup():
            self.http_client = httpx.AsyncClient(timeout=60.0)
        
        @self.app.on_event("shutdown")
        async def shutdown():
            if self.http_client:
                await self.http_client.aclose()
            self.shutdown_engine()
    
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
    
    async def fetch_kv_cache_metadata(
        self, 
        prefill_worker_url: str, 
        kv_cache_id: str
    ) -> Dict[str, Any]:
        """Fetch KV cache metadata from prefill worker."""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(timeout=60.0)
        
        try:
            response = await self.http_client.get(
                f"{prefill_worker_url}/kv_cache/{kv_cache_id}"
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch KV cache: {e}")
            raise HTTPException(status_code=500, detail=f"KV cache fetch failed: {e}")
    
    async def release_kv_cache(
        self, 
        prefill_worker_url: str, 
        kv_cache_id: str
    ):
        """Release KV cache on prefill worker."""
        if not self.http_client:
            return
        
        try:
            await self.http_client.delete(
                f"{prefill_worker_url}/kv_cache/{kv_cache_id}"
            )
        except Exception as e:
            logger.warning(f"Failed to release KV cache: {e}")
    
    async def process_decode(self, request: DecodeRequest) -> DecodeResponse:
        """
        Process decode request.
        
        Args:
            request: DecodeRequest containing KV cache reference and parameters
            
        Returns:
            DecodeResponse with generated text and metrics
        """
        start_time = time.perf_counter()
        
        try:
            # Fetch KV cache metadata from prefill worker
            kv_metadata = await self.fetch_kv_cache_metadata(
                request.prefill_worker_url,
                request.kv_cache_id
            )
            
            prompt = kv_metadata.get("prompt", "")
            
            if VLLM_AVAILABLE and self.engine is not None:
                # Real vLLM decode
                sampling_params_kwargs = {
                    "max_tokens": request.max_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                }
                
                if request.stop_sequences:
                    sampling_params_kwargs["stop"] = request.stop_sequences
                
                sampling_params = SamplingParams(**sampling_params_kwargs)
                
                # Generate unique internal request ID
                internal_request_id = f"decode_{request.request_id}_{uuid.uuid4().hex[:8]}"
                
                generated_text = ""
                generated_tokens = 0
                
                try:
                    # vLLM v0.14+ (V1 engine) API:
                    # engine.generate(request_id=..., prompt=..., sampling_params=...)
                    results_generator = self.engine.generate(
                        request_id=internal_request_id,
                        prompt=prompt,
                        sampling_params=sampling_params
                    )
                    
                    async for result in results_generator:
                        if result.outputs:
                            generated_text = result.outputs[0].text
                            if hasattr(result.outputs[0], 'token_ids'):
                                generated_tokens = len(result.outputs[0].token_ids)
                        
                        # Check if finished
                        if hasattr(result, 'finished') and result.finished:
                            break
                            
                except Exception as e:
                    logger.error(f"vLLM generate error: {e}", exc_info=True)
                    generated_text = f"[Generation error: {e}]"
                    generated_tokens = 0
                    
            else:
                # Mock mode - simulate generation
                await asyncio.sleep(0.1)  # Simulate processing time
                generated_text = f"[Mock response for: {prompt[:50]}...]"
                generated_tokens = len(generated_text.split())
            
            decode_time_ms = (time.perf_counter() - start_time) * 1000
            tokens_per_second = (generated_tokens / decode_time_ms) * 1000 if decode_time_ms > 0 else 0
            
            # Release KV cache
            await self.release_kv_cache(request.prefill_worker_url, request.kv_cache_id)
            
            return DecodeResponse(
                request_id=request.request_id,
                generated_text=generated_text,
                generated_tokens=generated_tokens,
                decode_time_ms=decode_time_ms,
                tokens_per_second=tokens_per_second,
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Decode error: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_decode_stream(
        self, 
        request: DecodeRequest
    ) -> AsyncGenerator[str, None]:
        """
        Process decode request with streaming.
        
        Args:
            request: DecodeRequest containing KV cache reference and parameters
            
        Yields:
            Server-sent events with generated tokens
        """
        try:
            # Fetch KV cache metadata
            kv_metadata = await self.fetch_kv_cache_metadata(
                request.prefill_worker_url,
                request.kv_cache_id
            )
            
            prompt = kv_metadata.get("prompt", "")
            
            if VLLM_AVAILABLE and self.engine is not None:
                # Configure sampling params with DELTA output for streaming
                sampling_params_kwargs = {
                    "max_tokens": request.max_tokens,
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
                
                # Generate unique internal request ID
                internal_request_id = f"stream_{request.request_id}_{uuid.uuid4().hex[:8]}"
                
                try:
                    # vLLM v0.14+ (V1 engine) API
                    results_generator = self.engine.generate(
                        request_id=internal_request_id,
                        prompt=prompt,
                        sampling_params=sampling_params
                    )
                    
                    previous_text = ""
                    async for result in results_generator:
                        if result.outputs:
                            current_text = result.outputs[0].text
                            
                            # In DELTA mode, current_text is already the new text
                            # In non-DELTA mode, we need to compute the diff
                            if VLLM_V1:
                                new_text = current_text  # DELTA mode
                            else:
                                new_text = current_text[len(previous_text):]
                                previous_text = current_text
                            
                            if new_text:
                                yield f"data: {new_text}\n\n"
                        
                        # Check if finished
                        if hasattr(result, 'finished') and result.finished:
                            yield f"data: [DONE]\n\n"
                            break
                        
                        # Also check finish_reason in outputs
                        if result.outputs and hasattr(result.outputs[0], 'finish_reason'):
                            if result.outputs[0].finish_reason:
                                yield f"data: [DONE]\n\n"
                                break
                                
                except Exception as e:
                    logger.error(f"Stream generate error: {e}", exc_info=True)
                    yield f"data: [ERROR] {str(e)}\n\n"
                    yield f"data: [DONE]\n\n"
            else:
                # Mock streaming
                mock_tokens = ["This ", "is ", "a ", "mock ", "response."]
                for token in mock_tokens:
                    await asyncio.sleep(0.05)
                    yield f"data: {token}\n\n"
                yield f"data: [DONE]\n\n"
            
            # Release KV cache
            await self.release_kv_cache(request.prefill_worker_url, request.kv_cache_id)
            
        except Exception as e:
            logger.error(f"Stream decode error: {e}", exc_info=True)
            yield f"data: [ERROR] {str(e)}\n\n"
    
    def shutdown_engine(self):
        """Shutdown the engine gracefully."""
        if self.engine is not None:
            try:
                if hasattr(self.engine, 'shutdown'):
                    self.engine.shutdown()
                logger.info("Engine shutdown complete")
            except Exception as e:
                logger.warning(f"Error during engine shutdown: {e}")
    
    def run(self):
        """Run the decode worker server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )


def create_decode_worker(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    host: str = "0.0.0.0",
    port: int = 8002,
    **kwargs
) -> DecodeWorker:
    """Factory function to create a decode worker."""
    config = DecodeWorkerConfig(
        model_name=model_name,
        host=host,
        port=port,
        **kwargs
    )
    return DecodeWorker(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Decode Worker")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf",
                        help="Model name (HuggingFace ID) or local directory path")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer path (optional, defaults to model path)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Enforce eager execution for faster startup")
    
    args = parser.parse_args()
    
    worker = create_decode_worker(
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
        worker.shutdown_engine()
