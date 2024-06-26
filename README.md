# Zendesk
Analyse zendesk tickets with RAG frameworks.
Starting with [LlamaIndex 🦙](https://www.llamaindex.ai/).

**Q: I just want to use the best model. Which one(s) should I use?**

The models currently performing best on my test Zendesk tickets are

```
# Reasonably priced and gives decent results
summarise_tickets.py --model claude --sub haiku <ticket number>

# Possibly the best results this month.
summarise_tickets.py --model gemini <ticket number>
```

## Setup

Replace the settings in [config.py](config.py) with your own Zendesk tickets and names. At the least
you should replace `COMPANY`.

Set the following environment variables.
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

# For finding close tickets
pip install --q chromadb haystack-ai jina-haystack chroma-haystack

# For clustering
pip install umap-learn
pip install hdbscan
python -m pip install -U matplotlib
```

For Claude

```
pip install llama-index-llms-anthropic
```

For Ollama

```
pip install llama-index-llms-ollama


ollama run mistral
ollama run llama2
ollama run llama3
ollama run zephyr

ollama pull mistral:instruct
ollama pull llama3:8b-instruct-q5_1

pip install dspy-ai
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

### Download Zendesk Tickets

Download all the Zendesk tickets created after [download_tickets.py.py](download_tickets.py.py)
`START_DATE` (currently `2000-01-01`) and write them to the `data` directory.

```
python download_tickets.py
```

### Summarise ticket comments

`summarise_tickets.py` summarises Zendesk tickets writes the summaries to the `summaries`
directory.

```
python summarise_tickets.py --model <llm model> <ticket number>
```

e.g.

```
python summarise_tickets.py --model llama  518539  # Summarise ticket 518539 using llama2 model
```

**NOTE:**

`summarise_tickets.py --model llama`requires the Ollama server to be running.
```
ollama serve
```

### Command Line Arguements
```
  --model MODEL         LLM model name. (llama | gemini | claude | openai)
  --sub SUB             Sub-model name. [claude: (haiku | sonnet | opus)]
  --overwrite           Overwrite existing summaries.
  --max_tickets MAX_TICKETS
                        Maximum number of tickets to process.
  --max_size MAX_SIZE   Maximum size of ticket comments in kilobytes.
  --pattern PATTERN     Select tickets with this pattern in the comments.
  --high                Process only high priority tickets.
  --all                 Process all tickets.
  --list                List tickets. Don't summarise.
```


e.g.
```
summarise_tickets.py --model llama  1234  # Ollama        Runs open source LLM locally!
summarise_tickets.py --model claude 1234  # Claude Haiku  The best cheap model!
summarise_tickets.py --model gemini 1234  # Gemini. Could be the best model this month!
summarise_tickets.py --model llama 1234   # Ollama Free and accurate but slow.
summarise_tickets.py --model claude -sub sonne 1234   # Even better than Haiku but costs more.
python summarise_tickets.py --model llama --max_size 10      # Summarise all tickets of ≤ 10 kb
python summarise_tickets.py --model llama --max_tickets 10   # Summarise your 10 tickets with the most comments
python summarise_tickets.py --model llama --high             # Summarise all your high priority tickets
python summarise_tickets.py --model llama --pattern "John\s+Doe" # Summarise all tickets containing the pattern John Doe
python summarise_tickets.py --model llama --pattern "John\s+Doe" --list # List all tickets containing the pattern John
```


## Observations

1. Anthropic Claude's Haiku gave great results.
1. Ollama: Generally slower than the commercial LLMs. Inconvenient for the very large tickets.
1. Zephyr gave the best results of the Ollama models.
1. Anthropic: Consistent [anthropic.RateLimitError](claude.png) errors after first set of tests.
1. Gemini finds more instances of facts than Anthropic Haiku.


# API Keys

## Gemini

https://aistudio.google.com/app/apikey
