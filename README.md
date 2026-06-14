# CAS NLP final project: tRAG

Final project for the CAS Natural language processing, June 2026.

## Summary
Comparison of two Retrieval-Augmented Generation (RAG) pipelines for the Swiss railway operating regulations (FDV/PCT): one indexing German-only documentation, one indexing German, French and Italian. The project investigates whether multilingual indexing improves retrieval accuracy and terminology precision for content Q&A and cross-lingual terminology queries, in the context of a multilingual organisation (SBB) with strict terminology requirements.

## Results and Report
The full methodology, evaluation design and discussion are available in the [accompanying report](https://github.com/WpnSta/CAS_Project_tRAG/blob/main/report/CAS_NLP_tRAG_report_v1.pdf). 

The multilingual RAG setup substantially outperformed the monolingual setup on retrieval accuracy for French and Italian queries and this directly improved terminology precision in generated answers. Cross-lingual terminology queries, used as an exploratory stress test, revealed a more fundamental mismatch between the RAG retrieval paradigm and queries about a corpus's linguistic structure rather than its content, suggesting that terminology lookup tasks may require a dedicated architecture (e.g. Terminology-Augmented Generation) rather than standard RAG.

## Project structure and data

The project is made up of the following files:

**Notebooks:**
(Recommended pipeline order)
* *01_explore.ipynb*: corpus analysis across all three languages.
* *02_mono_rag.ipynb*: complete RAG pipeline indexing only the German documents.
* *03_multi_rag.ipynb*: identical pipeline indexing all three language versions.
* *04_model_comparison.ipynb*: parallel generation pipeline to test and compare two local models on a subset of queries from the test set. 
* *05_evaluation.ipynb*: side-by-side comparison of both setups using the test set, retrieval performance, answer quality and terminology precision.
* *06_crosslingual_eval.ipynb*: exploratory pipeline stress test using crosslingual meta-level queries.

Several settings and supporting functions are managed separately and loaded in all or parts of the notebooks from the following **python files:**
* *config.py*: parameters shared by all setups and notebooks.
* *chunking.py*: document loading, text cleaning, and chunking logic.
* *retrieval.py*: three-pass retrieval pipeline, semantic search via ChromaDB, BM25 hybrid scoring, and cross-encoder reranking.
* *generation.py*: language detection, prompt assembly, and LLM answer generation via Ollama.

**Data:**
The documents used for this project are the Swiss Railway operating regulation documents as available in German, French and Italian from the [Federal office of transport FOT](https://www.bav.admin.ch/de/fahrdienstvorschriften-fdv).
* *questions.csv*: manually curated, multilingual test set (queries and terminology) used for evaluation.
* *cross_lingual_questions.csv*: manually curated multilingual test set (queries, source and target terms, based on the contents of questions.csv) used for crosslingual exploration in notebook 06.
* *data*: txt files used for the RAG setups in subfolders per language (de/fr/it).

## Requirements & Setup

The project runs on Python 3.10+ and depends on [Ollama](https://ollama.com/) for local LLM inference. Install it separately and pull the models used in `config.py` before running the notebooks. By default, the project uses `gemma3:4b` for answer generation and `phi4-mini` as the LLM judge for evaluation, but both can be swapped for any Ollama-compatible model by editing `config.py`. Python dependencies are listed in `requirements.txt`; install them with `pip install -r requirements.txt`. PyTorch is not listed there because the right wheel depends on your hardware: for CPU-only, run `pip install torch --index-url https://download.pytorch.org/whl/cpu`; for GPU, follow the instructions at [pytorch.org](https://pytorch.org/get-started/locally/). ChromaDB vector stores and BM25 indexes are built on first run of the respective notebooks and persisted locally — no pre-built indexes are included in the repository.
