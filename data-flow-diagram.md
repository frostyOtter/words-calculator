```
graph LR
A[User pastes text into input] --> B(Streamlit application)
B --> C{Select Embedding Service}
C -- GloVe --> D[GloVe embeddings]
C -- Cohere --> E[Cohere embeddings]
C -- OpenAI --> F[OpenAI embeddings]
C -- Azure_OpenAI --> G[Azure_OpenAI embeddings]
C -- SBERT --> H[SBERT embeddings]
D --> I{Process text}
E --> I
F --> I
G --> I
H --> I
I -- GloVe --> J[GloVe model]
I -- Other services --> K[Embeddings DB]
K -- Ingest --> L[Ingest new words]
J --> M{Find similar words}
K --> M
L --> K
M --> N[Display similar words]
```