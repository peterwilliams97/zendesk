""" Summarise Zendesk tickets using different LLMs and summarisation prompts.

    The script takes command-line arguments
        - to specify the LLM model to use,
        - whether to overwrite existing summaries, and
        - the maximum number of tickets and size of ticket comments to process.
    It then loads the specified LLM model, retrieves a list of ticket numbers, and summarise the
    tickets using the LLM model.

    It saves the resulting summaries to the "summaries" directory.

    Command-line arguments:
    --model: LLM model name. (llama | gemini | claude)
    --sub: LLM sub-model name. e.g, (opus | sonnet | haiku) for model claude.
    --overwrite: Overwrite existing summaries.
    --max_tickets: Maximum number of tickets to process.
    --max_size: Maximum size of ticket comments in kilobytes.
    --pattern: Select tickets with this pattern in the comments.
    --list: List tickets. Don't summarise.
"""
import sys
import time
from argparse import ArgumentParser
from utils import print_exit, match_key
from ticket_processor import ZendeskData, describe_tickets
from models import LLM_MODELS, sub_models, set_best_embedding
from classify_tfdidf import classify_tickets

def main():
    "Summarise Zendesk tickets using different LLMs and summarisation prompts across the command line."
    model_names = f"({' | '.join(LLM_MODELS.keys())})"
    has_submodels = [key for key in LLM_MODELS.keys() if len(LLM_MODELS[key].models) > 1]
    sub_model_names = f"[{' | '.join(sub_models(key) for key in has_submodels)}]"

    parser = ArgumentParser(description=("Evaluate different LLMs and summarisation prompts."))
    parser.add_argument('vars', nargs='*')
    parser.add_argument("--model", type=str, required=False,
        help=f"LLM model name. {model_names}"
    )
    parser.add_argument("--sub", type=str, required=False,
        help=f"Sub-model name. {sub_model_names}"
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
    parser.add_argument("--features", action="store_true",
        help="Summarise tickets for feature classification.")
    parser.add_argument("--classify", action="store_true",
        help="Classify tickets.")

    args = parser.parse_args()
    positionals = args.vars

    zd = ZendeskData()

    if positionals:
        ticket_numbers = [int(x) for x in positionals if x.isdigit()]
        new_numbers, bad_numbers = zd.add_new_tickets(ticket_numbers)
        if bad_numbers:
            print(f"Tickets not found: {bad_numbers}", file=sys.stderr)
    else:
        ticket_numbers = zd.ticket_numbers()
        priority = "high" if args.high else None
        ticket_numbers = zd.filter_tickets(ticket_numbers, args.pattern, priority,
                    args.max_size, args.max_tickets)

    ticket_numbers = zd.existing_tickets(ticket_numbers)

    if args.list:
        metadata_list = [(k, zd.metadata(k)) for k in ticket_numbers]
        describe_tickets(metadata_list)
        exit()

    if not args.model:
        print_exit("Model name not specified. Use --model to specify a model")

    model_name = match_key(LLM_MODELS, args.model)
    model_type = LLM_MODELS[model_name]
    if not model_type:
        print_exit(f"Unknown model '{args.model}'")

    set_best_embedding()
    submodel = None
    model_instance = model_type()
    if model_instance.models and args.sub:
        submodel = match_key(model_type.models, args.sub)
        if not submodel:
            print_exit(f"Unknown submodel '{args.sub}'")
        llm, model = model_instance.load(submodel)
    else:
        llm, model = model_instance.load()

    do_features = args.features or args.classify
    summariser = zd.get_summariser(llm, model, do_features)

    if args.classify:
        # assert ticket_numbers, "No tickets to classify."
        # data_list = [(zd.metadata(t), zd.comment_paths(t)) for t in ticket_numbers]
        # data_list = [both for both in data_list if both[1]]
        # assert ticket_numbers, f"No comments to classify in {len(ticket_numbers)} tickets."

        classify_tickets(zd, summariser, ticket_numbers)
        exit(0)

    print("Zendesk ticket summarisation ==========================================================")
    print(f"  LLM family: {model_name}")
    print(f"  LLM model: {model}")

    if not any((positionals, args.all, args.high, args.pattern, args.max_size, args.max_tickets)):
        print_exit("Please select a ticket number(s), specify a filter or use the --all flage.")

    print(f"Processing {len(ticket_numbers)} tickets with {model} " +
        f"(max {args.max_size} kb {args.max_tickets} tickets)...  ")

    summaryPaths = zd.summarise_tickets(ticket_numbers, summariser, overwrite=args.overwrite)

    print(f"{len(summaryPaths)} summary paths saved. {summaryPaths[:2]} ...")

if __name__ == "__main__":
    main()
