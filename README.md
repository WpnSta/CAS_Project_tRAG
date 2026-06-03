# CAS NLP final project: tRAG

Final project for the CAS Natural language processing. Comparison of two identical RAG setups, one based on German language only documentation, one on multilingual (DE/FR/IT) documentation.

## Summary
Goal and Content / Final report

## Project structure and data

The project is made up of the following files:

**Notebooks:**
(Recommended pipeline order)
* *01_explore.ipynb*: corpus analysis across all three languages.
* *02_mono_rag.ipynb*: complete RAG pipeline indexing only the German documents.
* *03_multi_rag.ipynb*: identical pipeline indexing all three language versions.
* *04_model_comparison.ipynb*: parallel generation pipeline to test and compare two local models on a subset of queries from the test set. 
* *05_evaluation.ipynb*: side-by-side comparison of both setups using the test set, retrieval performance, answer quality and terminology.precision.
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
* *data*: txt files used for the RAG setups.

