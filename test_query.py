""" Evaluate different LLMs and summarisation prompts.
    Tests the summariseTicketsObject function in process_tickets.py.
"""
from argparse import ArgumentParser
from zendesk_utils import storedTicketNumbers
from process_tickets import summariseTickets

def loadLlama2():
    "Load the Ollama llama2:7b LLM"
    from llama_index.llms.ollama import Ollama
    model = "llama2"
    llm =  Ollama(model="llama2", request_timeout=600)
    return llm, model

def loadGemini():
    "Load the Gemini LLM."
    from llama_index.llms.gemini import Gemini
    return Gemini(), "Gemini"

def loadClaude():
    "Load the Claude Haiku LLM."
    from llama_index.llms.anthropic import Anthropic
    # model = "claude-3-opus-20240229"
    # model = "claude-3-sonnet-20240229"
    model = "claude-3-haiku-20240307"
    llm = Anthropic(model=model, max_tokens=1024)
    return llm, model

def main():
    """
    Evaluate different LLMs and summarisation prompts.

    This function takes command-line arguments to specify
        the LLM model to use,
        whether to use plain or structured summarisation,
        whether to overwrite existing summaries, and
        the maximum number of tickets and size of ticket comments to process.

    It then loads the specified LLM model, retrieves a list of ticket numbers, and summarise the
    tickets using the LLM model and structured or plain summarisation
    It saves the resulting summaries to the structured.summaries directory.
    The resulting summary paths are wr

    Command-line arguments:
    --model: LLM model name. (llama | gemini | claude)
    --plain: Use plain (unstructured) summarisation.
    --overwrite: Overwrite existing summaries.
    --max_ticket: Maximum number of tickets to process.
    --max_size: Maximum size of ticket comments in kilobytes.
    """
    parser = ArgumentParser(description=("Evaluate different LLMs and summarisation prompts."))
    parser.add_argument("--model", type=str, required=True,
        help="LLM model name. (llama | gemini | claude)"
    )
    parser.add_argument("--plain", action="store_true",
        help="Use plain (unstructured) summarisation.")
    parser.add_argument("--overwrite", action="store_true",
        help="Overwrite existing summaries.")
    parser.add_argument("--max_ticket", type=int, default=1,
        help="Maximum number of tickets to process."
    )
    parser.add_argument("--max_size", type=int, default=100,
        help="Maximum size of ticket comments in kilobytes."
    )

    args = parser.parse_args()
    modelArg = args.model.lower()
    if modelArg.startswith("lla"):
        llm, model = loadLlama2()
    elif modelArg.startswith("gem"):
        llm, model = loadGemini()
    elif modelArg.startswith("cla"):
        llm, model = loadClaude()
    else:
        raise ValueError(f"Unknown model '{modelArg}'")

    ticketNumbers = storedTicketNumbers()
    print(f"Processing {len(ticketNumbers)} tickets with {model} " +
          f"(max {args.max_size} kb {args.max_ticket} tickets)...  ")

    summaryPaths = summariseTickets(ticketNumbers, llm, model,
        overwrite=args.overwrite, structured=not args.plain,
        max_size=args.max_size, max_tickets=args.max_ticket)

    print(f"{len(summaryPaths)} summary paths saved. {summaryPaths[:2]} ...")

if __name__ == "__main__":
    main()
