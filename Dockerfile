# Install uv
FROM python:3.12-slim-bookworm
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
# Copy application code
COPY src/ ./src/

# Create data directory and copy only necessary data files
RUN mkdir -p data
COPY data/20k-english-words.txt ./data/

# Expose Streamlit port
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app

# Run the application
CMD ["streamlit", "run", "src/main.py"]
