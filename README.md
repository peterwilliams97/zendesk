# Zendesk
Analyse zendesk tickets with RAG frameworks.
Starting with [LlamaIndex 🦙](https://www.llamaindex.ai/).

**Q: I just want to use the best model. Which one(s) should I use?**

The models currently performing best on my test Zendesk tickets are

* [summarise_tickets.py](summarise_tickets.py) --model claude (Anthropic One prompt per query)


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

`summarise_tickets.py` summarises Zendesk tickets writes the summaries to the `summaries`
directory.

```
python summarise_tickets.py --model <llm model> <ticket number>
```

e.g.

```
python summarise_tickets.py --model llama 518539  # Summarise ticket 518539 using llama2 model
```

**NOTE:**

`summarise_tickets.py --model llama`requires the Ollama server to be running.
```
ollama serve
```

### Command Line Arguements
```
--model: LLM model name. (llama | gemini | claude)
--sub: LLM sub-model name. e.g, (opus | sonnet | haiku) for model claude.
--struct: Use structured summarisation.
--overwrite: Overwrite existing summaries.
--max_tickets: Maximum number of tickets to process.
--max_size: Maximum size of ticket comments in kilobytes.
--pattern: Select tickets with this pattern in the comments.
--list: List tickets. Don't summarise.
```


e.g.
```
summarise_tickets.py --model llama  1234  # Ollama    One multi-query prompt.  Runs open source LLM locally!
summarise_tickets.py --model claude 1234  # Anthropic One multi-query prompt.
summarise_tickets.py --model gemini 1234  # Gemini    One multi-query prompt.
summarise_tickets.py --model llama --struct 1234      # Ollama    One prompt per query.
python summarise_tickets.py --model llama --max_size 10 # Summarise all tickets of ≤ 10 kb
python summarise_tickets.py --model llama --max_tickets 10 # Summarise your 10 tickets with the most comments
python summarise_tickets.py --model llama --high # Summarise all your high priority tickets
python summarise_tickets.py --model llama --pattern "John\s+Doe" # Summarise all tickets containing the pattern John Doe
python summarise_tickets.py --model llama --pattern "John\s+Doe" --list # List all tickets containing the pattern John Doe
```



## Observations

1. Ollama: Generally slower than the commercial LLMs. Inconvenient for the very large tickets.
1. Ollama: llama2 gave the best results.
1. Ollama: llama2:text didn't follow instructions.
1. Ollama: llama2:13b used so much memory it barely ran on my Macbook.
1. One prompt-per-query `summarise_tickets.py --model claude` gave cleanly structured results and  better
recall than one multi-query `summarise_tickets.py --model claude --plain` but used more Anthropic tokens.
1. Anthropic: Consistent [anthropic.RateLimitError](claude.png) errors after first set of tests.
1. Gemini finds more instances of facts than Anthropic Haiku but doesn't follow formatting
instructions as precisely.
