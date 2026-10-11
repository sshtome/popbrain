import sys
import os
sys.path.insert(0, os.path.abspath("."))

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from popbrain.dopamine.synaptic_memory import PopBrainSynapticMemory

def main():
    model_id = 'D:/projects/models/bitnet-b1.58-2B-4T'
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map='cuda', torch_dtype=torch.bfloat16)
    embed_layer = model.get_input_embeddings()

    with open('assets/datasets/squad_10k.json', 'r', encoding='utf-8') as f:
        squad_pairs = json.load(f)

    mem = PopBrainSynapticMemory(hidden_dim=2560, inhibition_threshold=0.85, modulation_strength=0.18)

    for i in range(100):
        text = f"Question: {squad_pairs[i]['question']}\nAnswer: {squad_pairs[i]['answer']}"
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')[0]
        with torch.no_grad():
            out = model(ids.unsqueeze(0), output_hidden_states=True)
            h = out.hidden_states[-1][0]
            emb = embed_layer(ids)
            mem.ingest_sequence(h, emb)

    cross_talk = 0
    max_sims = []
    for i in range(1000, 1050):
        q = squad_pairs[i]['question']
        prompt = f"Question: {q}\nAnswer:"
        curr_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')[0]
        with torch.no_grad():
            out = model(curr_ids.unsqueeze(0), output_hidden_states=True)
            h_last = out.hidden_states[-1][0, -1, :]
            delta_h, max_sim = mem.compute_synaptic_modulation(h_last)
            max_sims.append(max_sim)
            if delta_h.norm() > 0:
                cross_talk += 1

    print(f"Unlearned distractors tested: 50")
    print(f"Cross-talk spurious activations: {cross_talk}/50 ({cross_talk*2}%)")
    print(f"Max distractor sim observed: {max(max_sims):.4f}, Avg: {sum(max_sims)/len(max_sims):.4f}")

if __name__ == '__main__':
    main()
