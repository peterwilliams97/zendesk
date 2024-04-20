
import sys
import time
from argparse import ArgumentParser
from utils import print_exit
from ticket_processor import ZendeskData, describe_tickets
from rag_summariser import SUMMARISER_TYPES, SUMMARISER_DEFAULT

DO_TEMPURATURE = False
TEMPURATURE = 0.0
OLLAMA_TIMEOUT = 600

class ModelOllama:
    "Load the Ollama llama2:7b LLM"
    models = {
        "llama2": "llama2",
        #  "llama2:7b": "llama2:7b",
        "zephyr": "zephyr",
    }
    default_key = "zephyr"

    def load(self, key=None):
        from llama_index.llms.ollama import Ollama
        if not key:
            key = self.default_key
        model = self.models[key]
        assert model == "zephyr", f"Only Zephyr is supported. {model} not supported."
        if DO_TEMPURATURE:
            llm = Ollama(model=model, request_timeout=OLLAMA_TIMEOUT, temperature=TEMPURATURE)
        else:
            llm = Ollama(model=model, request_timeout=OLLAMA_TIMEOUT)
        return llm, model

class ModelGemini:
    models = {"pro": "models/gemini-1.5-pro-latest"}
    default_key = "pro"

    def load(self, key=None):
        "Load the Gemini LLM. I found models/gemini-1.5-pro-latest with explore_gemini.py. "
        from llama_index.llms.gemini import Gemini
        if not key:
            key = self.default_key
        model = self.models[key]
        if DO_TEMPURATURE:
            return Gemini(model_name=model, temperature=TEMPURATURE), "Gemini"
        else:
            return Gemini(model_name=model, request_timeout=10_000), "Gemini"

class ModelClaude:
    models = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-3-sonnet-20240229",
        "opus": "claude-3-opus-20240229",
    }
    default_key = "haiku"

    def load(self, key=None):
        "Load the Claude (Haiku | Sonnet | Opus) LLM."
        from llama_index.llms.anthropic import Anthropic
        if not key:
            key = self.default_key
        model = self.models[key]
        if DO_TEMPURATURE:
            llm = Anthropic(model=model, max_tokens=4024, temperature=TEMPURATURE)
        else:
            llm = Anthropic(model=model, max_tokens=4024)
        # Settings.llm = llm
        return llm, model

class ModelOpenAI:
    models = {"openai": "openai"}
    default_key = "openai"

    def load(self, key=None):
        "Load the OpenAI GPT-3 LLM."
        assert False, "OpenAI not supported"
        if not key:
            key = self.default_key
        from llama_index.llms.openai import OpenAI
        model = self.models[key]
        if DO_TEMPURATURE:
            llm = OpenAI(model=model, temperature=TEMPURATURE)
        else:
            llm = OpenAI(model=model)
        return llm, model

LLM_MODELS = {
    "llama":  ModelOllama,
    "gemini": ModelGemini,
    "claude": ModelClaude,
    "openai": ModelOpenAI,
}

def sub_models(key):
    "Return the submodels of the specified model."
    model = LLM_MODELS[key]
    return  f"{key}: ({' | '.join(model.models.keys())})"

# https://huggingface.co/Snowflake/snowflake-arctic-embed-m
EMBEDDING_NAME = "Snowflake/snowflake-arctic-embed-m"

def set_best_embedding():
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.settings import Settings

    embed_model = HuggingFaceEmbedding(
            model_name=EMBEDDING_NAME,
            trust_remote_code=True
        )
    Settings.embedding = embed_model
