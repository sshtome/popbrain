"""
PopBrain Knowledge Harvester v2
===============================
Extracts both macro-level architectural facts and fine-grained micro-level
code transitions from all Python modules, README, tests, and CUDA sources.
"""

import os
import re
from typing import List, Dict


def harvest_all(repo_root: str = "D:/projects/popbrain") -> List[Dict[str, str]]:
    knowledge_items = []

    # 1. Macro documentation facts from README
    readme_path = os.path.join(repo_root, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        current_section = "General"
        section_lines = []
        for line in lines:
            line_str = line.strip()
            if line_str.startswith("#"):
                if section_lines:
                    text_block = " ".join(section_lines)[:250]
                    if len(text_block) > 40:
                        knowledge_items.append({
                            "question": f"What is documented in PopBrain README under section '{current_section}'?",
                            "answer": text_block
                        })
                    section_lines = []
                current_section = line_str.lstrip("#").strip()
            elif line_str and not line_str.startswith("```") and not line_str.startswith("<"):
                section_lines.append(line_str)

    # 2. Extract detailed docstrings, functions, signatures from all .py files
    for root, dirs, files in os.walk(repo_root):
        if any(ex in root for ex in [".git", "__pycache__", "build", "dist"]):
            continue
        for file in files:
            if file.endswith(".py"):
                rel_path = os.path.relpath(os.path.join(root, file), repo_root).replace("\\", "/")
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Module level docstring
                mod_doc = re.match(r'^\s*"""(.*?)"""', content, re.DOTALL)
                if mod_doc:
                    doc_summary = " ".join(mod_doc.group(1).split())[:300]
                    knowledge_items.append({
                        "question": f"What is the module {rel_path} responsible for in PopBrain?",
                        "answer": f"Module {rel_path}: {doc_summary}"
                    })

                # Class definitions
                classes = re.findall(r'class\s+([A-Za-z0-9_]+)(?:\((.*?)\))?:\s*(?:"""(.*?)""")?', content, re.DOTALL)
                for cname, cbases, cdoc in classes:
                    cdoc_str = " ".join(cdoc.split())[:250] if cdoc else "No docstring."
                    knowledge_items.append({
                        "question": f"How is class {cname} defined in PopBrain module {rel_path}?",
                        "answer": f"Class {cname} in {rel_path} inherits from {cbases or 'object'}. Functionality: {cdoc_str}"
                    })

                # Function and method definitions
                funcs = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\((.*?)\)(?:\s*->\s*(.*?))?:\s*(?:"""(.*?)""")?', content, re.DOTALL)
                for fname, fargs, fret, fdoc in funcs:
                    if fname.startswith("__") and fname != "__init__":
                        continue
                    fdoc_str = " ".join(fdoc.split())[:200] if fdoc else "Executes internal logic."
                    fret_str = fret.strip() if fret else "Any"
                    knowledge_items.append({
                        "question": f"What are the arguments and purpose of {fname} in {rel_path}?",
                        "answer": f"In {rel_path}, function {fname}({fargs.strip()[:60]}) returns {fret_str}. Logic: {fdoc_str}"
                    })

                # Code blocks / definitions
                lines = content.splitlines()
                for i in range(0, len(lines), 20):
                    chunk = "\n".join(lines[i:i+20]).strip()
                    if len(chunk) > 80:
                        knowledge_items.append({
                            "question": f"What is the code implementation around line {i+1} in {rel_path}?",
                            "answer": f"In {rel_path} (lines {i+1}-{min(len(lines), i+20)}): {chunk[:350]}"
                        })

    # Deduplicate by question
    seen = set()
    deduped = []
    for it in knowledge_items:
        q = it["question"].strip()
        if q not in seen:
            seen.add(q)
            deduped.append(it)

    return deduped


if __name__ == "__main__":
    k = harvest_all()
    print(f"Total extracted granular knowledge items: {len(k)}")
