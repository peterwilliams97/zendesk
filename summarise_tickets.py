""" Evaluate different LLMs and summarisation prompts.
    Tests the summariseTicketsObject function in process_tickets.py.
"""
import sys
import time
from argparse import ArgumentParser
from ticket_processor import ZendeskData, listTickets

TEMPURATURE = 0.0
DO_TEMPURATURE = False

def loadLlama2():
    "Load the Ollama llama2:7b LLM"
    from llama_index.llms.ollama import Ollama
    model = "llama2"
    if DO_TEMPURATURE:
        llm = Ollama(model="llama2", request_timeout=600, temperature=TEMPURATURE)
    else:
        llm = Ollama(model="llama2", request_timeout=600)
    return llm, model

def loadGemini():
    "Load the Gemini LLM. I found models/gemini-1.5-pro-latest with explore_gemini.py. "
    from llama_index.llms.gemini import Gemini
    model = "models/gemini-1.5-pro-latest"
    if DO_TEMPURATURE:
        return Gemini(temperature=TEMPURATURE), "Gemini"
    else:
        return Gemini(model_name=model, request_timeout=10_000), "Gemini"

def loadClaude(submodel):
    "Load the Claude Haiku|Sonnet|Opus LLM."
    from llama_index.llms.anthropic import Anthropic
    model = "claude-3-haiku-20240307"
    if submodel:
        submodel = submodel.lower()
        assert False, "Submodels not supported"
        if submodel.startswith("op"):
            model = "claude-3-opus-20240229"
        elif submodel.startswith("so"):
            model = "claude-3-sonnet-20240229"

    if DO_TEMPURATURE:
        llm = Anthropic(model=model, max_tokens=1024, temperature=TEMPURATURE)
    else:
        llm = Anthropic(model=model, max_tokens=1024)
    return llm, model

def loadOpenAI():
    "Load the OpenAI GPT-3 LLM."
    assert False, "OpenAI not supported"
    from llama_index.llms.openai import OpenAI
    model = "openai"
    if DO_TEMPURATURE:
        llm = OpenAI(model=model, temperature=TEMPURATURE)
    else:
        llm = OpenAI(model=model)
    return llm, model

from llama_index.core.settings import Settings

def main():
    """
    Evaluate different LLMs and summarisation prompts.

    This function takes command-line arguments to specify
        the LLM model to use,
        whether to use plain or structured summarisation,
        whether to overwrite existing summaries, and
        the maximum number of tickets and size of ticket comments to process.

    It then loads the specified LLM model, retrieves a list of ticket numbers, and summarise the
    tickets using the LLM model and structured or plain summarisation.
    It saves the resulting summaries to the "summaries" directory.

    Command-line arguments:
    --model: LLM model name. (llama | gemini | claude)
    --sub: LLM sub-model name. e.g, (opus | sonnet | haiku) for model claude.
    --struct: Us unstructured summarisation.
    --overwrite: Overwrite existing summaries.
    --max_tickets: Maximum number of tickets to process.
    --max_size: Maximum size of ticket comments in kilobytes.
    --pattern: Select tickets with this pattern in the comments.
    --list: List tickets. Don't summarise.
    """
    parser = ArgumentParser(description=("Evaluate different LLMs and summarisation prompts."))
    parser.add_argument('vars', nargs='*')
    parser.add_argument("--model", type=str, required=True,
        help="LLM model name. (llama | gemini | claude)"
    )
    parser.add_argument("--sub", type=str, required=False,
        help="Sub-model name. (e.g. somnet)"
    )
    parser.add_argument("--sum", type=str, required=True,
        help="Summarisation type. (plain | structured | composite)."
    )
    parser.add_argument("--overwrite", action="store_true",
        help="Overwrite existing summaries.")
    parser.add_argument("--max_tickets", type=int, default=0,
        help="Maximum number of tickets to process."
    )
    parser.add_argument("--max_size", type=int, default=0,
        help="Maximum size of ticket comments in kilobytes."
    )
    parser.add_argument("--pattern", type=str, required=False,
        help="Select tickets with this pattern in the comments."
    )
    parser.add_argument("--high", action="store_true",
        help="Process only high priority tickets.")
    parser.add_argument("--list", action="store_true",
        help="List tickets. Don't summarise.")

    args = parser.parse_args()

    positionals = args.vars

    zd = ZendeskData()

    if positionals:
        ticket_numbers = [int(x) for x in positionals if x.isdigit()]
    else:
        ticket_numbers = zd.ticketNumbers()
        priority = "high" if args.high else None
        ticket_numbers = zd.filterTickets(ticket_numbers, args.pattern, priority,
                    args.max_size, args.max_tickets)

    if args.list:
        metadatas = [zd.metadata(k) for k in ticket_numbers]
        listTickets(metadatas)
        exit()

    model_arg = args.model.lower()
    if model_arg.startswith("lla"):
        llm, model = loadLlama2()
    elif model_arg.startswith("gem"):
        llm, model = loadGemini()
    elif model_arg.startswith("cla"):
        llm, model = loadClaude(args.sub)
    elif model_arg.startswith("oai"):
        llm, model = loadOpenAI()
    else:
        raise ValueError(f"Unknown model '{model_arg}'")

    print(f"Processing {len(ticket_numbers)} tickets with {model} " +
        f"(max {args.max_size} kb {args.max_tickets} tickets)...  ")
    summaryPaths = zd.summariseTickets(ticket_numbers, llm, model,
        overwrite=args.overwrite, summariser_name=args.sum)
    print(f"{len(summaryPaths)} summary paths saved. {summaryPaths[:2]} ...")

if __name__ == "__main__":
    main()
