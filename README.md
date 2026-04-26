# LinkedOut AI Hiring Platform

An AI-powered recruiting and candidate matching platform built in Python using Streamlit, FAISS, and HuggingFace LLMs.

## Features
- **Smart Uploads**: Automatically parses PDFs and Word Docs.
- **AI Extraction**: Uses Local AI models (`TinyLlama`) to intelligently pull technical capabilities. 
- **Semantic Vector DB**: Computes full embeddings of resumes via `sentence-transformers` and intelligently indexes them locally using Facebook AI Similarity Search (`FAISS`).
- **Employer Search**: Match job descriptions instantly using AI semantic capabilities without needing strict keyword matching.

## Getting Started

Follow these steps when downloading or cloning this repository to run it on your own machine.

### 1. Create a Virtual Environment
It is highly recommended to isolate the packages using a python virtual environment:
```bash
python -m venv venv
```

**Activate it:**
- On Windows: `.\venv\Scripts\activate`
- On Mac/Linux: `source venv/bin/activate`

### 2. Install Dependencies
Install all the required AI models and frameworks using the included requirements file:
```bash
pip install -r requirements.txt
```

### 3. Run the Platform
To launch the platform in your web browser, execute the standard Streamlit interface command:
```bash
streamlit run her.py
```

*Note: Upon your first launch and execution, the platform will take an extra few moments to download the local Open-Source AI models (`TinyLlama` and `all-MiniLM-L6-v2`) into your machine's memory.*
