FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and seed files
COPY db.py graph.py vector_store.py agent.py main.py seed.json ./

# Create directories for persistence
RUN mkdir -p /app/chroma_db

EXPOSE 8000

# Run uvicorn on container startup
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
