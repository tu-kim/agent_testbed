"""
vLLM Proxy Server

This server acts as a load balancer and coordinator between
prefill workers and decode workers for distributed vLLM inference.
Provides OpenAI-compatible API endpoint.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import time
import uuid
import json
from enum import Enum

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============== OpenAI Compatible Models ==============

class ChatMessage(BaseModel):
    """OpenAI-compatible chat message."""
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatCompletionChoice(BaseModel):
    """OpenAI-compatible chat completion choice."""
    index: int
    message: ChatMessage
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    """OpenAI-compatible usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    """OpenAI-compatible completion choice."""
    index: int
    text: str
    finish_reason: str


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: ChatCompletionUsage


# ============== Worker Management ==============

class WorkerType(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class WorkerInfo:
    """Information about a worker."""
    worker_id: str
    worker_type: WorkerType
    host: str
    port: int
    is_healthy: bool = True
    active_requests: int = 0
    last_health_check: float = 0.0
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class ProxyServerConfig:
    """Configuration for proxy server."""
    host: str = "0.0.0.0"
    port: int = 8000
    prefill_workers: List[Dict[str, Any]] = field(default_factory=list)
    decode_workers: List[Dict[str, Any]] = field(default_factory=list)
    health_check_interval: float = 30.0
    request_timeout: float = 120.0
    model_name: str = "meta-llama/Llama-2-7b-chat-hf"


class ProxyServer:
    """
    Proxy Server for distributed vLLM inference.
    
    Coordinates between prefill and decode workers,
    providing load balancing and OpenAI-compatible API.
    """
    
    def __init__(self, config: ProxyServerConfig):
        self.config = config
        self.prefill_workers: Dict[str, WorkerInfo] = {}
        self.decode_workers: Dict[str, WorkerInfo] = {}
        self.http_client: Optional[httpx.AsyncClient] = None
        self.health_check_task: Optional[asyncio.Task] = None
        self.app = FastAPI(
            title="vLLM Proxy Server",
            description="OpenAI-compatible API for distributed vLLM inference"
        )
        self._setup_routes()
        self._register_workers()
    
    def _register_workers(self):
        """Register workers from configuration."""
        for i, worker_config in enumerate(self.config.prefill_workers):
            worker_id = f"prefill_{i}"
            self.prefill_workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                worker_type=WorkerType.PREFILL,
                host=worker_config.get("host", "localhost"),
                port=worker_config.get("port", 8001 + i)
            )
        
        for i, worker_config in enumerate(self.config.decode_workers):
            worker_id = f"decode_{i}"
            self.decode_workers[worker_id] = WorkerInfo(
                worker_id=worker_id,
                worker_type=WorkerType.DECODE,
                host=worker_config.get("host", "localhost"),
                port=worker_config.get("port", 8002 + i)
            )
        
        logger.info(f"Registered {len(self.prefill_workers)} prefill workers")
        logger.info(f"Registered {len(self.decode_workers)} decode workers")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        # OpenAI-compatible endpoints
        @self.app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            if request.stream:
                return StreamingResponse(
                    self.stream_chat_completion(request),
                    media_type="text/event-stream"
                )
            return await self.process_chat_completion(request)
        
        @self.app.post("/v1/completions")
        async def completions(request: CompletionRequest):
            if request.stream:
                return StreamingResponse(
                    self.stream_completion(request),
                    media_type="text/event-stream"
                )
            return await self.process_completion(request)
        
        @self.app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.config.model_name,
                        "object": "model",
                        "owned_by": "vllm",
                        "permission": []
                    }
                ]
            }
        
        # Health and management endpoints
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "prefill_workers": len([w for w in self.prefill_workers.values() if w.is_healthy]),
                "decode_workers": len([w for w in self.decode_workers.values() if w.is_healthy])
            }
        
        @self.app.get("/workers")
        async def list_workers():
            return {
                "prefill_workers": [
                    {
                        "id": w.worker_id,
                        "url": w.url,
                        "healthy": w.is_healthy,
                        "active_requests": w.active_requests
                    }
                    for w in self.prefill_workers.values()
                ],
                "decode_workers": [
                    {
                        "id": w.worker_id,
                        "url": w.url,
                        "healthy": w.is_healthy,
                        "active_requests": w.active_requests
                    }
                    for w in self.decode_workers.values()
                ]
            }
        
        @self.app.on_event("startup")
        async def startup():
            self.http_client = httpx.AsyncClient(timeout=self.config.request_timeout)
            self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        @self.app.on_event("shutdown")
        async def shutdown():
            if self.health_check_task:
                self.health_check_task.cancel()
            if self.http_client:
                await self.http_client.aclose()
    
    def _select_worker(self, workers: Dict[str, WorkerInfo]) -> Optional[WorkerInfo]:
        """Select a healthy worker with least active requests."""
        healthy_workers = [w for w in workers.values() if w.is_healthy]
        if not healthy_workers:
            return None
        return min(healthy_workers, key=lambda w: w.active_requests)
    
    def _format_chat_prompt(self, messages: List[ChatMessage]) -> str:
        """Format chat messages into a prompt string."""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    async def _health_check_loop(self):
        """Periodically check worker health."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._check_all_workers()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    async def _check_all_workers(self):
        """Check health of all workers."""
        all_workers = list(self.prefill_workers.values()) + list(self.decode_workers.values())
        
        for worker in all_workers:
            try:
                response = await self.http_client.get(
                    f"{worker.url}/health",
                    timeout=5.0
                )
                worker.is_healthy = response.status_code == 200
                worker.last_health_check = time.time()
            except Exception:
                worker.is_healthy = False
                worker.last_health_check = time.time()
    
    async def process_chat_completion(
        self, 
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """Process a chat completion request."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Format prompt
        prompt = self._format_chat_prompt(request.messages)
        
        # Process through workers
        generated_text, prompt_tokens, completion_tokens = await self._process_request(
            request_id=request_id,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop
        )
        
        return ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(start_time),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=generated_text),
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    
    async def process_completion(
        self, 
        request: CompletionRequest
    ) -> CompletionResponse:
        """Process a completion request."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        generated_text, prompt_tokens, completion_tokens = await self._process_request(
            request_id=request_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop
        )
        
        return CompletionResponse(
            id=f"cmpl-{request_id}",
            created=int(start_time),
            model=request.model,
            choices=[
                CompletionChoice(
                    index=0,
                    text=generated_text,
                    finish_reason="stop"
                )
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        )
    
    async def _process_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]]
    ) -> tuple:
        """
        Process request through prefill and decode workers.
        
        Returns:
            Tuple of (generated_text, prompt_tokens, completion_tokens)
        """
        # Select workers
        prefill_worker = self._select_worker(self.prefill_workers)
        decode_worker = self._select_worker(self.decode_workers)
        
        if not prefill_worker or not decode_worker:
            raise HTTPException(
                status_code=503, 
                detail="No healthy workers available"
            )
        
        try:
            # Step 1: Prefill
            prefill_worker.active_requests += 1
            prefill_response = await self.http_client.post(
                f"{prefill_worker.url}/prefill",
                json={
                    "request_id": request_id,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop_sequences": stop_sequences
                }
            )
            prefill_worker.active_requests -= 1
            prefill_response.raise_for_status()
            prefill_data = prefill_response.json()
            
            # Step 2: Decode
            decode_worker.active_requests += 1
            decode_response = await self.http_client.post(
                f"{decode_worker.url}/decode",
                json={
                    "request_id": request_id,
                    "kv_cache_id": prefill_data["kv_cache_id"],
                    "prefill_worker_url": prefill_worker.url,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop_sequences": stop_sequences,
                    "stream": False
                }
            )
            decode_worker.active_requests -= 1
            decode_response.raise_for_status()
            decode_data = decode_response.json()
            
            return (
                decode_data["generated_text"],
                prefill_data["prefill_tokens"],
                decode_data["generated_tokens"]
            )
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during request processing: {e}")
            raise HTTPException(status_code=502, detail=str(e))
        except Exception as e:
            logger.error(f"Error during request processing: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def stream_chat_completion(self, request: ChatCompletionRequest):
        """Stream chat completion response."""
        request_id = str(uuid.uuid4())
        prompt = self._format_chat_prompt(request.messages)
        
        async for chunk in self._stream_request(
            request_id=request_id,
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop,
            model=request.model
        ):
            yield chunk
    
    async def stream_completion(self, request: CompletionRequest):
        """Stream completion response."""
        request_id = str(uuid.uuid4())
        
        async for chunk in self._stream_request(
            request_id=request_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stop_sequences=request.stop,
            model=request.model
        ):
            yield chunk
    
    async def _stream_request(
        self,
        request_id: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        stop_sequences: Optional[List[str]],
        model: str
    ):
        """Stream request through workers."""
        prefill_worker = self._select_worker(self.prefill_workers)
        decode_worker = self._select_worker(self.decode_workers)
        
        if not prefill_worker or not decode_worker:
            yield f"data: {json.dumps({'error': 'No healthy workers available'})}\n\n"
            return
        
        try:
            # Prefill
            prefill_response = await self.http_client.post(
                f"{prefill_worker.url}/prefill",
                json={
                    "request_id": request_id,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop_sequences": stop_sequences
                }
            )
            prefill_data = prefill_response.json()
            
            # Stream decode
            async with self.http_client.stream(
                "POST",
                f"{decode_worker.url}/decode/stream",
                json={
                    "request_id": request_id,
                    "kv_cache_id": prefill_data["kv_cache_id"],
                    "prefill_worker_url": prefill_worker.url,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stop_sequences": stop_sequences,
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        content = line[6:]
                        if content == "[DONE]":
                            yield f"data: [DONE]\n\n"
                        else:
                            chunk_data = {
                                "id": f"chatcmpl-{request_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    def run(self):
        """Run the proxy server."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            log_level="info"
        )


def create_proxy_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    prefill_workers: Optional[List[Dict[str, Any]]] = None,
    decode_workers: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    **kwargs
) -> ProxyServer:
    """Factory function to create a proxy server."""
    config = ProxyServerConfig(
        host=host,
        port=port,
        prefill_workers=prefill_workers or [{"host": "localhost", "port": 8001}],
        decode_workers=decode_workers or [{"host": "localhost", "port": 8002}],
        model_name=model_name,
        **kwargs
    )
    return ProxyServer(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Proxy Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--prefill-workers", type=str, nargs="+", 
                        default=["localhost:8001"],
                        help="Prefill worker addresses (host:port)")
    parser.add_argument("--decode-workers", type=str, nargs="+",
                        default=["localhost:8002"],
                        help="Decode worker addresses (host:port)")
    
    args = parser.parse_args()
    
    # Parse worker addresses
    prefill_workers = []
    for addr in args.prefill_workers:
        host, port = addr.split(":")
        prefill_workers.append({"host": host, "port": int(port)})
    
    decode_workers = []
    for addr in args.decode_workers:
        host, port = addr.split(":")
        decode_workers.append({"host": host, "port": int(port)})
    
    server = create_proxy_server(
        host=args.host,
        port=args.port,
        prefill_workers=prefill_workers,
        decode_workers=decode_workers,
        model_name=args.model_name
    )
    
    server.run()
