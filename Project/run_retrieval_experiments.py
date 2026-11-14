"""
run_retrieval_experiments.py
1. Compares k=1, k=3, and k=5 across 6 representative questions.
2. Tests chunk size / overlap variations (e.g., 300/30 vs 600/100) across 3 questions.
Exports the comparison findings to 'retrieval_experiments_report.csv' and '.xlsx'.
"""

import pandas as pd
from rag_engine import (
    DEFAULT_KB_DIR,
    build_or_rebuild_vector_store,
    retrieve_chunks,
    generate_grounded_answer,
)

SIX_QUESTIONS = [
    ("Q1 (Direct Product)", "What are the specs and price of the BQE AeroBook Pro 14 Laptop?", "02_product_catalog.txt"),
    ("Q2 (Current Policy)", "What is the current free shipping threshold and return window?", "03_current_policy.txt"),
    ("Q3 (Policy Exception)", "Can I return an opened AquaPure UV Flask if I change my mind?", "03_current_policy.txt"),
    ("Q4 (Multi-Document)", "If I order a BQE VoltMax Power Bank at 2 PM in Lahore, is shipping free and when does it dispatch?", "02_product_catalog.txt"),
    ("Q5 (Old vs Current)", "Compare the old v1.0 and current v2.0 return window and refund method.", "04_previous_policy.txt"),
    ("Q6 (Missing Info)", "Does Bano Qabil Ecommerce ship internationally to Dubai or London?", "None"),
]


def run_k_comparison():
    print("=== PART A: COMPARING k=1, k=3, k=5 (chunk_size=600, overlap=100) ===")
    collection = build_or_rebuild_vector_store(
        kb_dir=DEFAULT_KB_DIR,
        chroma_dir="chroma_exp_k",
        chunk_size=600,
        chunk_overlap=100,
        force_rebuild=True
    )

    rows = []
    for label, q, expected_file in SIX_QUESTIONS:
        for k in [1, 3, 5]:
            chunks = retrieve_chunks(collection, q, k=k)
            sources = [c["source"] for c in chunks]
            required_found = (expected_file in sources) if expected_file != "None" else (
                not chunks or chunks[0]["similarity"] < 0.22
            )
            irrelevant_found = len(set(sources)) > 2 and expected_file != "None"

            try:
                ans_obj = generate_grounded_answer(q, chunks)
                ans_text = ans_obj["answer"]
            except Exception as err:
                ans_text = f"[LLM Skipped - No API Key: {err}]"

            rows.append({
                "Question_ID": label,
                "Question": q,
                "K_Value": k,
                "Retrieved_Sources": ", ".join(sources),
                "Required_Evidence_Retrieved": "Yes" if required_found else "No",
                "Irrelevant_Noise_Retrieved": "Yes" if irrelevant_found else "No",
                "Top_Similarity": chunks[0]["similarity"] if chunks else 0.0,
                "Answer_Preview": ans_text[:180].replace("\n", " ")
            })

    return pd.DataFrame(rows)


def run_chunk_comparison():
    print("=== PART B: COMPARING CHUNK SIZE (300/30 vs 600/100) ===")
    three_qs = [SIX_QUESTIONS[2], SIX_QUESTIONS[3], SIX_QUESTIONS[4]]
    rows = []

    for c_size, c_overlap in [(300, 30), (600, 100)]:
        col = build_or_rebuild_vector_store(
            kb_dir=DEFAULT_KB_DIR,
            chroma_dir=f"chroma_exp_c_{c_size}",
            chunk_size=c_size,
            chunk_overlap=c_overlap,
            force_rebuild=True
        )
        for label, q, expected_file in three_qs:
            chunks = retrieve_chunks(col, q, k=3)
            sources = [c["source"] for c in chunks]
            rows.append({
                "Chunk_Size": c_size,
                "Chunk_Overlap": c_overlap,
                "Question_ID": label,
                "Question": q,
                "Retrieved_Sources": ", ".join(sources),
                "Top_Similarity": chunks[0]["similarity"] if chunks else 0.0,
                "Top_Chunk_Preview": chunks[0]["text"][:160].replace("\n", " ") if chunks else ""
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df_k = run_k_comparison()
    df_chunk = run_chunk_comparison()

    df_k.to_csv("retrieval_k_comparison.csv", index=False)
    df_chunk.to_csv("retrieval_chunk_comparison.csv", index=False)

    with pd.ExcelWriter("retrieval_experiments_report.xlsx", engine="openpyxl") as writer:
        df_k.to_excel(writer, index=False, sheet_name="K_Value_Comparison")
        df_chunk.to_excel(writer, index=False, sheet_name="Chunk_Size_Comparison")

    print("Saved retrieval_experiments_report.xlsx successfully!")
