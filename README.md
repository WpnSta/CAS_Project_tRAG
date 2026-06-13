# CAS NLP final project: tRAG

Final project for the CAS Natural language processing. Comparison of two identical RAG setups, one based on German language only documentation, one on multilingual (DE/FR/IT) documentation.


## Summary
Retrieval-Augmented Generation (RAG) is increasingly used to ground LLM outputs in domain-specific knowledge, but much research and tooling has focused on monolingual, English-centric settings. Results cannot necessarily be transferred to multilingual settings which are often very different, depending on the languages and documents involved. For organisations operating in high-stakes domains, such as the Swiss Federal Railways (SBB), precise and consistent terminology across German, French and Italian is required. This project investigates whether a multilingual RAG setup (indexing documents in all three languages) produces more precise and consistent terminology than a monolingual (German-only) setup, applied to the Swiss railway operating regulations (FDV/PCT). Two RAG pipelines combining semantic search, BM25 lexical scoring and cross-encoder reranking were built and compared on two use cases: classic content Q&A in three languages and cross-lingual terminology queries, the latter used as an exploratory stress test of the pipelines’ limits. The use cases were evaluated for retrieval accuracy, terminology precision and overall answer quality. 
Retrieval performance proved to be the decisive factor: the multilingual setup achieves a hit rate of 95.5% for French and Italian queries, against 36% and 54.5% for the monolingual setup, and terminology precision in the generated answers follows directly from this. When the correct source document is unavailable in the query language, the generative model fails to produce the correct domain-specific term, confirming that an LLM’s parametric knowledge cannot always substitute for in-language grounding. The cross-lingual terminology queries expose a more fundamental limitation: because these queries interrogate the linguistic structure of the corpus rather than its factual content, the RAG retrieval paradigm itself is mismatched to the task. The findings suggest that multilingual RAG is a viable tool for content-oriented Q&A in regulated multilingual environments, but that terminology lookup for language professionals requires a different architecture, such as newer approaches like Terminology-Augmented Generation (TAG) which integrate dedicated terminology resources dynamically. 


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
* *data*: txt files used for the RAG setups in subfolders per language (de/fr/it).

