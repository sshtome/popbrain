"""
PopBrain Synaptic LLM OpenAI-Compatible API Server
==================================================
Serves Qwen2.5-Coder-1.5B with real-time zero-backprop PopBrainScale10kMemory.
Supports standard OpenAI /v1/chat/completions and /v1/completions endpoints.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import time
import json
from typing import List, Dict, Optional, Any
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

from popbrain.dopamine.scalable_memory import PopBrainScale10kMemory

app = FastAPI(title="PopBrain Synaptic LLM Server", version="1.3.0")

# Global runtime state
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = r"C:\Users\ilya\.cache\huggingface\hub\models--Qwen--Qwen2.5-Coder-1.5B-Instruct\snapshots\2e1fd397ee46e1388853d2af2c993145b0f1098a"
memory_checkpoint = "assets/popbrain_qwen_synaptic_memory.pt"

print("Initializing PopBrain LLM Engine...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map=device,
    torch_dtype=torch.bfloat16
)
hidden_dim = model.config.hidden_size

print("Loading Synaptic Memory...")
memory = PopBrainScale10kMemory(
    hidden_dim=hidden_dim,
    num_clusters=128,
    nprobe=4,
    inhibition_threshold=0.82,
    margin_threshold=0.015,
    modulation_strength=0.25,
    device=device
)
if os.path.exists(memory_checkpoint):
    memory.load_memory(memory_checkpoint)
    print(f"Loaded {memory.num_memories} synaptic transitions from {memory_checkpoint}!")
else:
    print("Warning: Synaptic memory checkpoint not found!")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2.5-coder-popbrain"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.0
    synaptic_modulation: Optional[bool] = True


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    synaptic_diagnostics: Optional[Dict[str, Any]] = None


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Qwen2.5-Coder-1.5B-Instruct",
        "synaptic_memories": memory.num_memories,
        "vram_gb": round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
    }


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest):
    # Format chat using tokenizer chat template
    hf_msgs = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        prompt = tokenizer.apply_chat_template(hf_msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback formatting
        prompt = "\n".join([f"{m.role.capitalize()}: {m.content}" for m in req.messages]) + "\nAssistant:"

    curr_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]
    initial_len = curr_ids.shape[0]

    gen_tokens = []
    fired_steps = 0
    max_similarity = 0.0
    t0 = time.time()

    max_toks = req.max_tokens or 256

    for _ in range(max_toks):
        with torch.no_grad():
            out = model(curr_ids.unsqueeze(0), output_hidden_states=True)
            h_last = out.hidden_states[-1][0, -1, :]

            if req.synaptic_modulation:
                delta_h, sim = memory.compute_synaptic_modulation(h_last)
                max_similarity = max(max_similarity, sim)
                if delta_h.norm() > 0:
                    fired_steps += 1
                    h_mod = h_last + delta_h
                    logits = model.lm_head(h_mod.unsqueeze(0))[0]
                else:
                    logits = out.logits[0, -1, :]
            else:
                logits = out.logits[0, -1, :]

            if req.temperature and req.temperature > 0.0:
                probs = torch.softmax(logits / req.temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            else:
                next_tok = torch.argmax(logits).unsqueeze(0)

            curr_ids = torch.cat([curr_ids, next_tok])
            gen_tokens.append(next_tok.item())

            if next_tok.item() == tokenizer.eos_token_id:
                break

    dt = time.time() - t0
    response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    return ChatCompletionResponse(
        id=f"chatcmpl-{int(time.time()*1000)}",
        created=int(time.time()),
        model=req.model,
        choices=[{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop" if (gen_tokens and gen_tokens[-1] == tokenizer.eos_token_id) else "length"
        }],
        usage={
            "prompt_tokens": initial_len,
            "completion_tokens": len(gen_tokens),
            "total_tokens": initial_len + len(gen_tokens)
        },
        synaptic_diagnostics={
            "synaptic_active": req.synaptic_modulation,
            "fired_steps": fired_steps,
            "max_cosine_similarity": round(max_similarity, 4),
            "generation_time_sec": round(dt, 3),
            "tokens_per_sec": round(len(gen_tokens) / max(0.001, dt), 2)
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8088, log_level="info")
