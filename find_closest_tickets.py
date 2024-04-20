"""
    Find the most similar Zendesk tickets based on the provided arguments.

    Usage:
        python find_closest_tickets.py [options] [ticket_numbers]

    Options:
        --model <model_name>        LLM model name. Available models: <model_names>
        --sub <sub_model_name>      Sub-model name. Available sub-models: <sub_model_names>
        --clusters                  Compute clusters.
        --tune_clusters             Find best cluster parameters.
        --verbose                   Show ticket summaries in output.
        --max_tickets <num>         Maximum number of tickets to process.
        --max_size <size>           Maximum size of ticket comments in kilobytes.
        --top_k <num>               Number of closest tickets to return.
        --recurse <num>             Recursively find closest tickets of tickets up to this total number.
        --pattern <pattern>         Select tickets with this pattern in the comments.
        --high                      Process only high priority tickets.
        --all                       Process all tickets.
        --list                      List tickets. Don't summarise.

    Arguments:
        ticket_numbers              Optional list of ticket numbers to process.

    Note:
        - The available LLM models are: <model_names>
        - The available sub-models are: <sub_model_names>
"""
import sys
import time
from argparse import ArgumentParser
from utils import print_exit, match_key, truncate, regex_compile
from ticket_processor import ZendeskData, describe_tickets
from models import LLM_MODELS, sub_models
from reranker import QueryEngine
from cluster_tickets import find_clusters

RE_SUMMARY = regex_compile("SUMMARY:\s*(.*)\s*\n\n")

def main():
    model_names = f"({' | '.join(LLM_MODELS.keys())})"
    has_submodels = [key for key in LLM_MODELS.keys() if len(LLM_MODELS[key].models) > 1]
    sub_model_names = f"[{' | '.join(sub_models(key) for key in has_submodels)}]"

    parser = ArgumentParser(description=("Find the most similar Zendesk tickets."))
    parser.add_argument('vars', nargs='*')
    parser.add_argument("--model", type=str, required=False,
        help=f"LLM model name. {model_names}"
    )
    parser.add_argument("--sub", type=str, required=False,
        help=f"Sub-model name. {sub_model_names}"
    )
    parser.add_argument("--clusters", action="store_true",
        help="Compute clusters.")
    parser.add_argument("--tune_clusters", action="store_true",
        help="Finb beset cluster params.")
    parser.add_argument("--verbose", action="store_true",
        help="Show ticket summaries in output.")
    parser.add_argument("--max_tickets", type=int, default=0,
        help="Maximum number of tickets to process."
    )
    parser.add_argument("--max_size", type=int, default=0,
        help="Maximum size of ticket comments in kilobytes."
    )
    parser.add_argument("--top_k", type=int, default=10,
        help="Number of closest tickets to return."
    )
    parser.add_argument("--recurse", type=int, default=0,
        help="Recursively find closest tickets of tickets up to this total number."
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

    submodel = None
    model_instance = model_type()
    if model_instance.models and args.sub:
        submodel = match_key(model_type.models, args.sub)
        if not submodel:
            print_exit(f"Unknown submodel '{args.sub}'")
        llm, model = model_instance.load(submodel)
    else:
        llm, model = model_instance.load()

    query_engine = QueryEngine(zd.df, llm, model)

    def describe_ticket(ticket_number, max_len=200):
        subject = zd.describe(ticket_number, max_len//2)
        content = query_engine.ticket_content(ticket_number, allow_not_exist=True)
        if args.verbose:
            lines = content.split("\n")
            lines = [line for line in lines if line.strip()]
            lines = [f"              {line}" for line in lines]
            content = "\n".join(lines)
            return f"{subject}\n{content}"

        m = RE_SUMMARY.search(content)
        if m:
            content = m.group(1)
        remaining = max(20, max_len - len(subject) - 5)
        content = repr(truncate(content, remaining))
        return f"{subject:50} :: {content}"

    if args.clusters or args.tune_clusters:
        ticket_numbers = query_engine.ticket_numbers()
        # ticket_numbers = ticket_numbers[:1_000]
        ticket_contents = [query_engine.ticket_content(t) for t in ticket_numbers]
        # n_neighbors, n_components, min_cluster_size = find_best_params(ticket_contents)
        # assert False, "Not implemented"
        label_cluster, labels = find_clusters(ticket_contents, tune_clusters=args.tune_clusters)
        cluster_sizes = [len(label_cluster[label]) for label in labels]
        print(f"Clusters: {cluster_sizes} ========================================================")
        for i, label in enumerate(labels):
            cluster = label_cluster[label]
            print(f"{i:4}:  {len(cluster):4} ----------------------------------------")
            for j, ticket_idx in enumerate(cluster[:10]):
                ticket_number = ticket_numbers[ticket_idx]
                print(f"{j:8}: {ticket_number:7}  {describe_ticket(ticket_number)}")
        exit()

    if args.recurse > 0:
        results = query_engine.find_closest_tickets_recurse(ticket_numbers,
            top_k=args.top_k, max_results=args.recurse)
    else:
        results = query_engine.find_closest_tickets(ticket_numbers, top_k=args.top_k)

    print(f"Similar tickets: {len(results)} test tickets")
    for i, (query_number, result) in enumerate(results):
        print(f"{i:4}:     {query_number:7} {len(result):2}     {describe_ticket(query_number)}")
        for j, (ticket_number, score) in enumerate(result):
            print(f"{j:8}: {ticket_number:7} ({score:4.2f}) {describe_ticket(ticket_number)}")

if __name__ == "__main__":
    main()
