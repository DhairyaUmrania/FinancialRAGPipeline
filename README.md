Overview of the System
This is a RAG system that uses different retrieval methods (BM25, vector search, and hybrid approaches) along with language models to answer questions about financial documents. The system includes components for document ingestion, retrieval, reranking, and text generation.
Individual File Analysis
1. main.py
This is the entry point for the RAG system that ties all components together:
•	Loads environment variables using dotenv
•	Initializes document ingestion, retrieval, reranking, and language model components
•	Implements an interactive query loop where users can ask financial questions
•	Demonstrates four approaches for comparison: 
o	Baseline 1: BM25 retrieval with a base language model
o	Baseline 2: Vector retrieval with a base language model
o	Improvement 1: Hybrid retrieval with reranking
o	Improvement 2: Finetuned model with hybrid retrieval
2. ingestion.py
Handles the loading and processing of financial documents:
•	Uses LangChain's DirectoryLoader and PyPDFLoader to load PDF documents
•	Splits documents into smaller chunks using RecursiveCharacterTextSplitter
•	Returns the document chunks for further processing
3. retriever.py
Implements document retrieval functionality:
•	Sets up two retrieval methods: BM25 and Vector search using Pinecone
•	Initializes a HuggingFace embedding model for vector embeddings
•	Manages the Pinecone index creation and document ingestion
•	Provides methods to retrieve documents using either BM25 or vector similarity
4. hybrid_retriever.py
Implements a reranking system to improve retrieval quality:
•	Uses a cross-encoder model to score document relevance to a query
•	Takes results from both BM25 and vector retrievers and reranks them
•	Outputs a list of scores for candidate documents
5. llm_model.py
Implements the base language model for generating answers:
•	Loads a seq2seq model (default: google/flan-t5-small)
•	Wraps the model in a LangChain pipeline
•	Implements a generate_answer method that: 
o	Takes a retriever and query
o	Constructs a prompt with retrieved context
o	Generates an answer based on the context and question
6. finetuned_llm.py
Similar to llm_model.py but uses a finetuned model:
•	Loads a finetuned seq2seq model from a specified path
•	Implements the same generate_answer interface as the base model
•	Uses a prompt template designed for the finetuned model
7. NLP_Finetuned.py
Contains the code for finetuning a language model:
•	Loads the FinLang/investopedia-instruction-tuning-dataset
•	Preprocesses the dataset for finetuning a seq2seq model
•	Implements a training loop with early stopping
•	Tracks and plots training and validation losses
•	Saves checkpoints and the best model
•	Tests the model on sample financial questions
8. evaluation.py
Implements evaluation metrics for the RAG system:
•	Loads a test dataset from FinLang/investopedia-instruction-tuning-dataset
•	Implements F1 and ROUGE metrics for answer evaluation
•	Compares four different retrieval-generation combinations: 
o	BM25 + base model
o	Vector retrieval + base model
o	Hybrid retrieval + base model
o	Hybrid retrieval + finetuned model
•	Outputs performance metrics for each approach
9. embedding_visualization.py
Visualizes embeddings for different retrieval methods:
•	Generates sample financial queries
•	Retrieves documents using BM25, vector, and hybrid approaches
•	Reduces embedding dimensions using t-SNE, PCA, or UMAP
•	Creates visualizations showing the relationship between: 
o	Query embeddings
o	BM25 retrieved document embeddings
o	Vector retrieved document embeddings
o	Hybrid retrieved document embeddings with reranker scores
•	Saves the visualizations as PNG files
10. requirements.txt
Lists all the Python package dependencies for the project:
•	Document processing: pypdf, langchain
•	Vector databases: pinecone, faiss-cpu
•	ML/NLP libraries: transformers, torch, sentence-transformers
•	Text retrieval: rank_bm25
•	Utility libraries: numpy, requests, python-dotenv
System Architecture and Flow
1.	Document Ingestion: PDFs are loaded and split into chunks using ingestion.py
2.	Retrieval: Documents are retrieved using BM25, vector search, or a hybrid approach
3.	Reranking: The hybrid approach uses hybrid_retriever.py to rerank candidate documents
4.	Answer Generation: Either llm_model.py (base model) or finetuned_llm.py (finetuned model) generates answers
5.	Evaluation: The system is evaluated using metrics in evaluation.py
6.	Visualization: Embeddings are visualized using embedding_visualization.py

