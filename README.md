# Word Analogy Calculator

A Streamlit application that performs word analogies using various embedding models. This tool allows you to explore word relationships through vector operations using different embedding services like GloVe, Cohere, OpenAI, Azure OpenAI, and SBERT.

## Features

- Support for multiple embedding services:
  - GloVe
  - Cohere
  - OpenAI
  - Azure OpenAI
  - SBERT
- Vector-based word analogy calculations
- Database storage for API-based embeddings
- Interactive web interface

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```
   uv pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```
   COHERE_API_KEY=your_cohere_key
   OPENAI_API_KEY=your_openai_key
   OPENAI_MODEL=your_model_name
   AZURE_OPENAI_API_KEY=your_azure_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   AZURE_OPENAI_DEPLOYMENT=your_deployment_name
   GEMINI_API_KEY=your_gemini_key
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run src/main.py
   ```
2. Select your preferred embedding service
3. Enter words with operations (- or +) between them
   - Example: `king - man + woman`
4. View the results showing similar words based on the analogy

## Requirements

- Python 3.x
- Streamlit
- Gensim
- NumPy
- PyYAML
- Loguru
- Various embedding service SDKs (Cohere, OpenAI, etc.)


Thanks to English data from 'eyturner'