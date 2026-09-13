"""Test suite for the PRISM retrieval pipeline.

Files are grouped into categories A-H, one concern per file, so four people can
add tests in parallel without touching the same file:

    A  test_a_query_preprocessing.py   query cleaning + classification
    B  test_b_snippet_preprocessing.py snippet normalization, id preservation
    C  test_c_indexing.py              FAISS / BM25 index construction
    D  test_d_dense_retrieval.py       bi-encoder search
    E  test_e_bm25_retrieval.py        lexical search
    F  test_f_fusion.py                RRF / weighted fusion
    G  test_g_reranking.py             cross-encoder reordering
    H  test_h_pipeline_e2e.py          end-to-end + MTEB contract

Run everything:      pytest
Run one category:    pytest tests/test_f_fusion.py
Skip the slow ones:  pytest -m "not slow"
"""
