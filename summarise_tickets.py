""" Summarise Zendesk tickets using different LLMs and summarisation prompts.

    The script takes command-line arguments
        - to specify the LLM model to use,
        - whether to use plain, structured, composite or Pydantic JSON summarisation,
        - whether to overwrite existing summaries, and
        - the maximum number of tickets and size of ticket comments to process.
    It then loads the specified LLM model, retrieves a list of ticket numbers, and summarise the
    tickets using the LLM model and the selected summarisation method.

    It saves the resulting summaries to the "summaries" directory.

    Command-line arguments:
    --model: LLM model name. (llama | gemini | claude)
    --sub: LLM sub-model name. e.g, (opus | sonnet | haiku) for model claude.
    --method: Summarisation method. (plain | structured | composite | pydantic)
    --overwrite: Overwrite existing summaries.
    --max_tickets: Maximum number of tickets to process.
    --max_size: Maximum size of ticket comments in kilobytes.
    --pattern: Select tickets with this pattern in the comments.
    --list: List tickets. Don't summarise.
"""
import sys
import time
from argparse import ArgumentParser
from ticket_processor import ZendeskData, describeTickets
from rag_summariser import SUMMARISER_TYPES, SUMMARISER_DEFAULT

DO_TEMPURATURE = False
TEMPURATURE = 0.0

class modelOllama:
    "Load the Ollama llama2:7b LLM"
    models = {
        "llama2": "llama2",
        #  "llama2:7b": "llama2:7b",
    }
    default_key = "llama2"

    def load(self, key=None):
        from llama_index.llms.ollama import Ollama
        if not key:
            key = self.default_key
        model = self.models[key]
        if DO_TEMPURATURE:
            llm = Ollama(model=model, request_timeout=600, temperature=TEMPURATURE)
        else:
            llm = Ollama(model=model, request_timeout=600)
        return llm, model

class modelGemini:
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

class modelClaude:
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

class modelOpenAI:
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
    "llama":  modelOllama,
    "gemini": modelGemini,
    "claude": modelClaude,
    "openai": modelOpenAI,
}

def subModels(key):
    "Return the submodels of the specified model."
    model = LLM_MODELS[key]
    return  f"{key}: ({' | '.join(model.models.keys())})"

def matchKey(a_dict, key):
    """ Find a key in dictionary `a_dict` that starts with the given key (case-insensitive).
        Returns the matching key if found, None otherwise.
    """
    matches = [k for k in a_dict if k.startswith(key.lower())]
    if not matches:
        print(f"{key}' doesn't match any of {list(a_dict.keys())}", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(f"{key}' matches {matches}. Choose one.", file=sys.stderr)
        return None
    return matches[0]

def printExit(message):
    "Print a message and exit."
    print(message, file=sys.stderr)
    exit()

def main():
    "Summarise Zendesk tickets using different LLMs and summarisation prompts across the command line."
    model_names = f"({' | '.join(LLM_MODELS.keys())})"
    has_submodels = [key for key in LLM_MODELS.keys() if len(LLM_MODELS[key].models) > 1]
    sub_model_names = f"[{' | '.join(subModels(key) for key in has_submodels)}]"
    summariser_names = f"({' | '.join(SUMMARISER_TYPES.keys())})"

    parser = ArgumentParser(description=("Evaluate different LLMs and summarisation prompts."))
    parser.add_argument('vars', nargs='*')
    parser.add_argument("--model", type=str, required=False,
        help=f"LLM model name. {model_names}"
    )
    parser.add_argument("--sub", type=str, required=False,
        help=f"Sub-model name. {sub_model_names}"
    )
    parser.add_argument("--method", type=str, required=False,
        help=f"Summarisation type. {summariser_names}"
    )
    parser.add_argument("--overwrite", action="store_true",
        help="Overwrite existing summaries.")
    parser.add_argument("--max_tickets", type=int, default=0,
        help="Maximum number of tickets to process."
    )
    parser.add_argument("--max_size", type=int, default=0,
        help="Maximum size of ticket comments in kilobytes."
    )
    parser.add_argument("--pattern", type=str, required=False, default="",
        help="Select tickets with this pattern in the comments."
    )
    parser.add_argument("--high", action="store_true",
        help="Process only high priority tickets.")
    parser.add_argument("--all", action="store_true",
        help="Process all tickets.")
    parser.add_argument("--list", action="store_true",
        help="List tickets. Don't summarise.")

    args = parser.parse_args()
    positionals = args.vars

    zd = ZendeskData()

    if positionals:
        ticket_numbers = [int(x) for x in positionals if x.isdigit()]
        new_numbers, bad_numbers = zd.addNewTickets(ticket_numbers)
        if bad_numbers:
            print(f"Tickets not found: {bad_numbers}", file=sys.stderr)
    else:
        ticket_numbers = zd.ticketNumbers()
        priority = "high" if args.high else None
        ticket_numbers = zd.filterTickets(ticket_numbers, args.pattern, priority,
                    args.max_size, args.max_tickets)

    ticket_numbers = zd.existingTickets(ticket_numbers)

    if args.list:
        metadata_list = [(k, zd.metadata(k)) for k in ticket_numbers]
        describeTickets(metadata_list)
        exit()

    if not args.model:
        printExit("Model name not specified. Use --model to specify a model")

    model_name = matchKey(LLM_MODELS, args.model)
    model_type = LLM_MODELS[model_name]
    if not model_type:
        printExit(f"Unknown model '{args.model}'")

    submodel = None
    model_instance = model_type()
    if model_instance.models and args.sub:
        submodel = matchKey(model_type.models, args.sub)
        if not submodel:
            printExit(f"Unknown submodel '{args.sub}'")
        llm, model = model_instance.load(submodel)
    else:
        llm, model = model_instance.load()

    if not args.method:
        printExit("--method not specified.")
    summariser_name = matchKey(SUMMARISER_TYPES, args.method)
    if not summariser_name:
        printExit("Summariser method not specified.")
    summariser_type = SUMMARISER_TYPES[summariser_name]
    if not summariser_type:
        printExit(f"Unknown summariser '{args.method}'")

    print("Zendesk ticket summarisation ==========================================================")
    print(f"  LLM family: {model_name}")
    print(f"  LLM model: {model}")
    print(f"  Summarisation method: {summariser_name}")

    if not any((positionals, args.all, args.high, args.pattern, args.max_size, args.max_tickets)):
        printExit("Please select a ticket number(s), specify a filter or use the --all flage.")

    print(f"Processing {len(ticket_numbers)} tickets with {model} " +
        f"(max {args.max_size} kb {args.max_tickets} tickets)...  ")

    summaryPaths = zd.summariseTickets(ticket_numbers, llm, model, summariser_type,
        overwrite=args.overwrite)

    print(f"{len(summaryPaths)} summary paths saved. {summaryPaths[:2]} ...")

if __name__ == "__main__":
    main()
