"""
PopBrain Ingest Codebase Knowledge Engine for Qwen2.5-Coder-1.5B
================================================================
Ingests the entire PopBrain documentation, Python code signatures, and architecture
into PopBrainScale10kMemory using frozen Qwen2.5-Coder-1.5B representations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))

import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from popbrain.dopamine.scalable_memory import PopBrainScale10kMemory
from popbrain.knowledge_harvester import harvest_all


def ingest_popbrain_codebase(
    model_path: str = r"C:\Users\ilya\.cache\huggingface\hub\models--Qwen--Qwen2.5-Coder-1.5B-Instruct\snapshots\2e1fd397ee46e1388853d2af2c993145b0f1098a",
    save_path: str = "assets/popbrain_qwen_synaptic_memory.pt"
):
    print("=" * 80)
    print("PopBrain Codebase & Documentation Ingestion into Synaptic Memory")
    print(f"Target Model: Qwen2.5-Coder-1.5B-Instruct")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("1. Harvesting all codebase entities and documentation...")
    items = harvest_all("D:/projects/popbrain")
    print(f"Total knowledge items harvested: {len(items)}")

    print("\n2. Loading Qwen2.5-Coder-1.5B-Instruct on CUDA...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    embed_layer = model.get_input_embeddings()
    hidden_dim = model.config.hidden_size # 1536

    print(f"Model hidden dimension: {hidden_dim}, Base VRAM: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

    # Initialize scalable synaptic memory with Qwen hidden dimension (1536)
    memory = PopBrainScale10kMemory(
        hidden_dim=hidden_dim,
        num_clusters=128,
        nprobe=4,
        inhibition_threshold=0.82,
        margin_threshold=0.015,
        modulation_strength=0.20,
        device=device
    )

    print("\n3. Ingesting knowledge items via single forward passes (0 backprop)...")
    t0_start = time.time()
    batch_size = 32

    for b_start in range(0, len(items), batch_size):
        b_chunk = items[b_start : b_start + batch_size]
        prompts = [f"Question: {it['question']}\nAnswer: {it['answer']}" for it in b_chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512, add_special_tokens=False).to(device)

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            hidden_states = out.hidden_states[-1]
            embs = embed_layer(enc.input_ids)

            chunk_keys = []
            chunk_vals = []
            for bi in range(len(b_chunk)):
                seq_len = int(enc.attention_mask[bi].sum().item())
                if seq_len >= 2:
                    h_seq = hidden_states[bi, :seq_len]
                    e_seq = embs[bi, :seq_len]

                    k = h_seq[:-1]
                    k = k / (torch.norm(k, dim=-1, keepdim=True) + 1e-6)
                    v = e_seq[1:]
                    v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)

                    chunk_keys.append(k)
                    chunk_vals.append(v)

            del out, hidden_states, embs, enc
            if chunk_keys:
                batch_k = torch.cat(chunk_keys, dim=0)
                batch_v = torch.cat(chunk_vals, dim=0)
                memory.ingest_batch(batch_k, batch_v)
                del batch_k, batch_v, chunk_keys, chunk_vals

        if b_start % 128 == 0:
            torch.cuda.empty_cache()

    dt_ingest = time.time() - t0_start
    print(f"\nIngested {len(items)} knowledge items in {dt_ingest:.2f}s!")
    print(f"Total synaptic transitions stored: {memory.num_memories}")

    print("\n4. Building Spherical Voronoi Inverted Index (IVF-128)...")
    t0_ivf = time.time()
    memory.build_ivf_index(iterations=3)
    dt_ivf = time.time() - t0_ivf
    print(f"IVF Index built in {dt_ivf:.3f}s. IVF Status: {memory.ivf_built}")

    vram_total = torch.cuda.memory_allocated() / (1024**3)
    print(f"Total VRAM Allocated (Model + Synaptic Memory): {vram_total:.2f} GB")

    print(f"\n5. Saving synaptic memory state to {save_path}...")
    memory.save_memory(save_path)
    print("Memory checkpoint saved successfully!")

    # 6. Verify retrieval on 5 spot-check queries
    print("\n6. Running validation spot-check queries:")
    test_queries = [
        "What is PopBrain and what are its core capabilities?",
        "What CUDA kernel accelerates the SNN engine in PopBrain?",
        "What is PopBrainScale10kMemory and how does it index 10,000 facts?",
        "How does TieredLayerStreamer run 32B and 70B LLMs on 12GB GPUs in PopBrain?",
        "What optimizer is used for population evolution in PopBrain core?"
    ]

    for q in test_queries:
        prompt = f"Question: {q}\nAnswer:"
        curr_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)[0]

        gen_tokens = []
        max_seen_sim = 0.0
        fired_count = 0

        for _ in range(40):
            with torch.no_grad():
                out = model(curr_ids.unsqueeze(0), output_hidden_states=True)
                h_last = out.hidden_states[-1][0, -1, :]
                delta_h, sim = memory.compute_synaptic_modulation(h_last)
                max_seen_sim = max(max_seen_sim, sim)
                if delta_h.norm() > 0:
                    fired_count += 1
                    h_mod = h_last + delta_h
                    logits = model.lm_head(h_mod.unsqueeze(0))[0]
                else:
                    logits = out.logits[0, -1, :]
                next_tok = torch.argmax(logits).unsqueeze(0)
                curr_ids = torch.cat([curr_ids, next_tok])
                gen_tokens.append(next_tok.item())
                if next_tok.item() == tokenizer.eos_token_id or "\n" in tokenizer.decode(next_tok):
                    break

        ans = tokenizer.decode(gen_tokens).strip()
        print(f"\n[Q]: {q}")
        print(f"[Sim: {max_seen_sim:.4f} | Fired steps: {fired_count}] -> [A]: {ans}")

    return memory


if __name__ == "__main__":
    ingest_popbrain_codebase()
