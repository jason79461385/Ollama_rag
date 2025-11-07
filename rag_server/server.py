from __future__ import annotations
import os, time, argparse, traceback, json
from typing import List, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    Chroma = None

app = FastAPI(title="Unified RAG/LLM Server")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["rag", "llm"], default=os.getenv("MODE", "rag"))
args, _ = parser.parse_known_args()
MODE = args.mode


OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")
EMBED_MODEL = os.getenv("EMBED_MODEL", "mxbai-embed-large")

DB_DIR = os.getenv("DB_PATH", "./chroma_db")
TOP_K = int(os.getenv("TOP_K", "8"))

SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "If context is provided, use it to answer accurately.\n"
    "If unknown, say you don't know.\n\nContext:\n{context}\n"
)

_state = {"emb": None, "vs": None, "err": None, "mode": MODE}

class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.2
    top_k: Optional[int] = None
    stream: Optional[bool] = False 

def ensure_rag_ready():
    if _state["vs"] is not None:
        return
    if Chroma is None:
        raise RuntimeError("Chroma not available; please install langchain_community")

    try:
        emb = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        _ = emb.embed_query("ping")
        vs = Chroma(persist_directory=DB_DIR, embedding_function=emb, collection_name="local_docs")
        _state["emb"] = emb
        _state["vs"] = vs
        _state["err"] = None
    except Exception as e:
        _state["err"] = f"init failed: {e}"
        raise

@app.get("/health")
def health():
    return {
        "ok": True,
        "mode": MODE,
        "llm_model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "db_path": DB_DIR,
        "ready": (_state["vs"] is not None) if MODE == "rag" else True,
        "init_error": _state["err"],
    }

@app.get("/debug/files")
def debug_files():
    ensure_rag_ready()
    metas = _state["vs"]._collection.get(include=["metadatas"])
    sources = sorted(set(m.get("source") for m in metas["metadatas"] if m.get("source")))
    return {"count": len(sources), "sources": sources}

@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": "rag1-ollama", "object": "model"}]}

def retrieve_context(query: str, k: int) -> str:
    try:
        ensure_rag_ready()
        docs = _state["vs"].similarity_search(query, k=k)
        return "\n\n".join(
            f"[DOC {i+1}] {d.metadata.get('source','')}\n{d.page_content}"
            for i, d in enumerate(docs)
        )
    except Exception as e:
        print("[retrieve] error:", e)
        traceback.print_exc()
        return ""

@app.post("/v1/chat/completions")
def chat(req: ChatCompletionRequest):
    try:
        last_user = ""
        for m in req.messages:
            if m.role == "user":
                last_user = m.content

        llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=req.temperature or 0.2)

        
        if req.stream:
            if MODE == "rag":
                context = retrieve_context(last_user, req.top_k or TOP_K)
                sys_prompt = SYSTEM_PROMPT.format(context=context)
                messages = [SystemMessage(content=sys_prompt), HumanMessage(content=last_user)]
            else:
                messages = [HumanMessage(content=last_user)]

            
            def stream_generator():
                try:
                    for chunk in llm.stream(messages):
                        text_chunk = chunk.content or ""
                        
                        chunk_data = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": req.model,
                            "choices": [{
                                "index": 0,
                                "delta": {"role": "assistant", "content": text_chunk},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    
                    final_chunk = {
                        "id": f"chatcmpl-{int(time.time())}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [{
                            "index": 0,
                            "delta": {}, 
                            "finish_reason": "stop" 
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except Exception as e:
                    print(f"[stream_generator] error: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    yield "data: [DONE]\n\n"

            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            if MODE == "rag":
                context = retrieve_context(last_user, req.top_k or TOP_K)
                sys_prompt = SYSTEM_PROMPT.format(context=context)
                ans = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=last_user)]).content
            else:
                ans = llm.invoke(last_user).content

            return JSONResponse(content={
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": req.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": ans}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            })
            
    except Exception as e:
        print("[/v1/chat/completions] error:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": traceback.format_exc()})
