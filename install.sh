#!/usr/bin/env bash

pip install --upgrade pip

pip install llama-index
pip install llama_index_core
pip install llama-index-embeddings-huggingface

pip install weaviate-client
pip install llama-index-vector-stores-weaviate
pip install torch sentence-transformers

# Already installed
pip install llama-index-llms-anthropic

pip install llama-index-vector-stores-qdrant
pip install llama-index-embeddings-gemini
pip install llama-index-llms-gemini
# pip install -q llama-index google-generativeai

pip install llama-index-llms-ollama
