import torch
import json
import time
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

    t0 = time.time()
    for i in range(100):
        item = squad_pairs[i]
        text = f"Question: {item['question']}\nAnswer: {item['answer']}"
        ids = tokenizer(text, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')[0]
        with torch.no_grad():
            out = model(ids.unsqueeze(0), output_hidden_states=True)
            h = out.hidden_states[-1][0]
            emb = embed_layer(ids)
            mem.ingest_sequence(h, emb, reward=1.0)
    print(f"Ingested 100 facts in {time.time()-t0:.2f}s. Stored transitions: {mem.num_memories}")

    recalled = 0
    for i in range(10):
        item = squad_pairs[i]
        q = item['question']
        target = item['answer'].strip().lower()
        prompt = f"Question: {q}\nAnswer:"
        curr_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')[0]
        
        gen_tokens = []
        for _ in range(8):
            with torch.no_grad():
                out = model(curr_ids.unsqueeze(0), output_hidden_states=True)
                h_last = out.hidden_states[-1][0, -1, :]
                delta_h, max_sim = mem.compute_synaptic_modulation(h_last)
                if delta_h.norm() > 0:
                    h_mod = h_last + delta_h
                    logits = model.lm_head(h_mod.unsqueeze(0))[0]
                else:
                    logits = out.logits[0, -1, :]
                next_tok = torch.argmax(logits).unsqueeze(0)
                curr_ids = torch.cat([curr_ids, next_tok])
                gen_tokens.append(next_tok.item())
                if next_tok.item() == tokenizer.eos_token_id or '\n' in tokenizer.decode(next_tok):
                    break
        pred = tokenizer.decode(gen_tokens).strip()
        is_ok = target in pred.lower()
        if is_ok: recalled += 1
        print(f"[{i+1}] Q: {q[:45]} | Target: '{target}' | Pred: '{pred}' | Match: {is_ok}")
    print(f"Recall: {recalled}/10 ({recalled*10}%)")

if __name__ == '__main__':
    main()
