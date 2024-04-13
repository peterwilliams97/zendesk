"""
Find the most similar Zendesk tickets based on the provided arguments.
"""
import sys
import time
from argparse import ArgumentParser
from utils import print_exit
from ticket_processor import ZendeskData, describe_tickets
from reranker import QueryEngine

def main():
    """
    Find the most similar Zendesk tickets based on the provided arguments.

    Args:
        vars (list): Additional command-line arguments.
        --overwrite (bool): Overwrite existing summaries.
        --recursive (bool): Search for similar tickets recursively.
        --max_tickets (int): Maximum number of tickets to process.
        --max_size (int): Maximum size of ticket comments in kilobytes.
        --pattern (str): Select tickets with this pattern in the comments.
        --high (bool): Process only high priority tickets.
        --all (bool): Process all tickets.
        --list (bool): List tickets. Don't summarize.
    """
    parser = ArgumentParser(description=("Find the most similar Zendesk tickets."))
    parser.add_argument('vars', nargs='*')
    parser.add_argument("--overwrite", action="store_true",
        help="Overwrite existing summaries.")
    parser.add_argument("--recursive", action="store_true",
        help="Search for similar tickets recursively.")
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
    query_engine = QueryEngine(zd.df)

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

    print("Zendesk ticket similar tickets =========================================================")

    if not any((positionals, args.all, args.high, args.pattern, args.max_size, args.max_tickets)):
        print_exit("Please select a ticket number(s), specify a filter or use the --all flage.")

    print(f"Processing {len(ticket_numbers)} " +
        f"(max {args.max_size} kb {args.max_tickets} tickets)...  ")

    top_k = 10
    min_score = 0.8
    if args.recursive:
        results = query_engine.find_closest_tickets_recursive(ticket_numbers,
                top_k=top_k, min_score=min_score, max_depth=10)
    else:
        results = query_engine.find_closest_tickets(ticket_numbers,
                top_k=top_k, min_score=min_score)

    print(f"Similar tickets: {len(results)}")
    for i, (query_number, result) in enumerate(results):
        print(f"{i:4}: {query_number:7} {len(result)}")
        for j, (ticket_number, score) in enumerate(result):
            print(f"{j:8}: {ticket_number:7} ({score:4.2f})")

if __name__ == "__main__":
    main()
