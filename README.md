# Zendesk
Anaylse zendesk tickets with RAG frameworks.
Starting with [LlamaIndex 🦙](https://www.llamaindex.ai/).

**Q: I just want to use the best model. Which one(s) should I use?**

The models currently performing best on my test Zendesk tickets are

* [test_query.py](test_query.py) --model claude (Anthropic One prompt per query) and
* [test_query.py](test_query.py) --model gemini (Gemini One prompt per query)

## Setup

Set the following environment variables
```
export ZENDESK_USER="user@mydomain.com"          # e.g. peter@papercut.com
export ZENDESK_TOKEN="<redacted>"                # e.g. password1234
export ZENDESK_SUBDOMAIN="my zendesk subdomain"  # e.g. papercut

python -m venv .zdenv
source .zdenv/bin/activate

pip install --upgrade pip

pip install llama-index
pip install llama_index_core
pip install llama-index-embeddings-huggingface
```

For Claude

```
pip install llama-index-llms-anthropic
```

For Ollama

```
pip install llama-index-llms-ollama

ollama pull mistral:instruct
ollama run mistral
ollama run llama2
```

For Gemini

```
# pip install llama-index-multi-modal-llms-gemini
pip install llama-index-vector-stores-qdrant
pip install llama-index-embeddings-gemini
pip install llama-index-llms-gemini
# pip install -q llama-index google-generativeai
```


Set API keys for the LLMs you are using. You only need to set the ones you are running.
```
export OPENAI_API_KEY="sk-..."
export COHERE_API_KEY="Qht... "
export LLAMA_CLOUD_API_KEY="llx-..."
export ANTHROPIC_API_KEY="sk-ant..."
```


## Run

### Read Zendesk Tickets

Download the comments the Zendesk tickets `TICKET_NUMBERS` in [download_tickets.py.py](download_tickets.py.py) and
write them to
the `data` directory.


```
python download_tickets.py.py.
```

### Summarise ticket comments

Summarise the Zendesk tickets the `data` directory and write the summaries to the `summaries`
directory

There are several versions of code for doing this
```
test_query.py --model llama  --plain # Ollama    One multi-query prompt.  Runs open source LLM locally!
test_query.py --model claude --plain # Anthropic One multi-query prompt.
test_query.py --model gemini --plain # Gemini    One multi-query prompt.
test_query.py --model llama          # Ollama    One prompt per query.
test_query.py --model claude         # Anthropic    One prompt per query.
test_query.py --model gemini         # Gemini    One prompt per query.
```

`test_query.py --model llama`requires the Ollama server to be running.
```
ollama serve
```

## Test cases

Tested with the following support tickets. The bottom ones are the largest support ticket histories
from my work and are test cases for how big a set of documents we can search.

```
   0: 1259693    7 comments   2.888 kb
   1: 1216136   26 comments  20.715 kb
   2: 1196141  122 comments  81.527 kb
   3: 1260221  106 comments 126.619 kb
   4: 1116722  288 comments 190.168 kb
   5: 1280919  216 comments 731.220 kb
```

## Observations

1. Ollama: Generally slower than the commercial LLMs. Inconvenient for the very large tickets.
1. Ollama: llama2 gave the best results.
1. Ollama: llama2:text didn't follow instructions.
1. Ollama: llama2:13b used so much memory it barely ran on my Macbook.
1. One prompt-per-query `test_query.py --model claude` gave cleanly structured results and  better
recall than one multi-query `test_query.py --model claude --plain` but used more Anthropic tokens.
1. Anthropic: Consistent [anthropic.RateLimitError](claude.png) errors after first set of tests.
1. Gemini finds more instances of facts than Anthropic Haiku but doesn't follow formatting
instructions as precisely
