""" Functions for reading Zendesk data and summarising tickets.
    - The ZendeskData class provides methods to interact with the Zendesk data, such as filtering
      tickets, summarising tickets, and listing tickets.
    - ticket_has_pattern() returns True if any of the comments for a ticket contains a
       specified pattern.
    - describe_tickets() prints the ticket information for each metadata in a list.
"""
import glob
import os
import re
import time
import sys
from utils import total_size_kb, current_time, since, load_text
from zendesk_wrapper import comment_paths, add_tickets_to_index, load_existing_index
from evaluate_summary import summarise_ticket, summary_text
from rag_summariser import PydanticSummariser
from rag_classifier import PydanticFeatureGenerator

def ticket_has_pattern(ticket_number, pattern):
    "Returns True if any of the comments for ticket with number `ticket_number` contains `pattern`."
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    paths = comment_paths(ticket_number)
    return any(regex.search(load_text(path)) for path in paths)

TICKETS_SHOWN = 5   # Number of tickets to show in the summary.

def describe_tickets(metadata_list):
    "Prints the ticket information for each metadata in the given list."
    for i, (ticket_number, metadata) in enumerate(metadata_list):
        created_at = metadata.created_at
        status = metadata.status
        subject = metadata.subject
        priority = metadata.priority
        number = metadata.comments_num
        size = metadata.comments_size / 1024
        print(f"{i:3}: {ticket_number:8} {created_at.date()} {status:8} {priority:7} {number:3} {size:4.1f} {subject[:100]}")

def show_tickets(ticket_numbers, num_shown):
    """ Display summaries of the comments in the Zendesk tickets with numbers `ticket_numbers`.
        Show `num_shown` tickets from the start and end of the list.
    """
    def show_one(i):
        ticket_number = ticket_numbers[i]
        paths = comment_paths(ticket_number)
        print(f"{i:8}: {ticket_number:8} {len(paths):3} comments {total_size_kb(paths):6.2f} kb")

    if len(ticket_numbers) <= num_shown:
        for i in range(len(ticket_numbers)):
            show_one(i)
        return

    half = (num_shown + 1 ) // 2
    for i in range(half):
        show_one(i)
    print("    ...")
    for i in range(len(ticket_numbers) - half + 1, len(ticket_numbers)):
        show_one(i)

def summarise_one_ticket(summariser, i, ticket_number, metadata, overwrite):
    commentCount = len(comment_paths(ticket_number))
    commentSize = total_size_kb(comment_paths(ticket_number))

    print(f"{i:2}: ticket_number={ticket_number:8} {commentCount:3} comments {commentSize:7.3f} kb {current_time()}",
        flush=True)

    summary_path = summariser.summary_path(ticket_number)

    if not overwrite and os.path.exists(summary_path):
        print(f"   skipping ticket {ticket_number}. Already processed. '{summary_path}'",
            flush=True)
        return None   # Skip tickets that have already been summarised.

    t0 = time.time()
    try:
        summary_dict, err = summarise_ticket(summariser, ticket_number, metadata)
    except Exception as e:
        print(f"Error processing ticket {ticket_number}.", file=sys.stderr)
        raise

    if err:
        print(f"  Could not process {ticket_number}: {err}.", flush=True)
        return None

    summary = summary_text(summary_dict)

    description = (f"{commentCount} comments {commentSize:5.2f} kb {since(t0):4.1f} sec " +
                   f"summary={len(summary)} chars")

    with open(summary_path, "w") as f:
        print(f"Zendesk info: ticket {ticket_number}: {description} --------------------------------",
            file=f)
        print(summary, file=f)
    print(f"  {description} saved to {os.path.abspath(summary_path)}", flush=True)
    return summary_path

# Create a hierarchy of classes to handle different types of summarisation. !@#$
class ZendeskData:
    """
    A class that represents Zendesk data and provides methods to interact with it.

    Attributes:
        df (DataFrame): The Zendesk data as a pandas DataFrame.

    Methods:
        ticket_numbers(): Returns a list of ticket numbers.
        ticket(ticket_number): Returns the ticket with the specified ticket number.
        metadata(ticket_number): Returns the metadata of the ticket with the specified ticket number.
        ticket_has_priority(ticket_number, priority): Returns True if the ticket with the specified
                ticket number has the specified priority.
        summarise_tickets(ticket_numbers, llm, model, structured, overwrite=False): Summarizes the
                conversations from the Zendesk support tickets specified by `ticket_numbers`.
    """
    def __init__(self):
        df = load_existing_index()
        self.df = df

    def ticket_numbers(self):
        return list(self.df.index)

    def metadata(self, ticket_number):
        return self.df.loc[ticket_number]

    def describe(self, ticket_number, max_len=150):
        metadata = self.metadata(ticket_number)
        subject = metadata["subject"]
        return repr(subject[:max_len])

    def comment_paths(self, ticket_number: int):
        return comment_paths(ticket_number)

    def ticket_has_priority(self, ticket_number, priority):
        "Returns True if ticket with number `ticket_number` has priority `priority`."
        metadata = self.metadata(ticket_number)
        return metadata.priority == priority

    def existing_tickets(self, ticket_numbers):
        "Filters out ticket numbers that do not exist in the DataFrame index."
        reduced_numbers = []
        for t in ticket_numbers:
            if t not in self.df.index:
                print(f"    Ticket {t} not found. Skipping.")
                continue
            reduced_numbers.append(t)
        if len(reduced_numbers) < len(ticket_numbers):
            print(f"    Ticket numbers reduced to {len(reduced_numbers)} for existing tickets.")
        return reduced_numbers

    def add_new_tickets(self, ticket_numbers):
        """ Adds tickets with numbers `ticket_numbers` to the index.
            Returns new_ticket_numbers, bad_ticket_numbers where:
            - new_ticket_numbers: the ticket numbers that were added to the index.
            - bad_ticket_numbers: the numbers that were not Zendesk ticket numbers.
        """
        self.df, new_ticket_numbers, bad_ticket_numbers = add_tickets_to_index(self.df, ticket_numbers)
        return new_ticket_numbers, bad_ticket_numbers

    def filter_tickets(self, ticket_numbers, pattern, priority, max_size, max_tickets):
        print(f"  Filtering {len(ticket_numbers)} tickets.")
        if priority:
            reduced_numbers = [t for t in ticket_numbers if self.ticket_has_priority(t, priority)]
            print(f"    Ticket numbers reduced to {len(reduced_numbers)} to match priority '{priority}'.")
            ticket_numbers = reduced_numbers
        if max_size > 0:
            reduced_numbers = [k for k in ticket_numbers if total_size_kb(comment_paths(k)) <= max_size]
            if len(reduced_numbers) < len(ticket_numbers):
                print(f"    Ticket numbers reduced to {len(reduced_numbers)} for {max_size} kb size limit")
                ticket_numbers = reduced_numbers
        if pattern:
            reduced_numbers = [t for t in ticket_numbers if ticket_has_pattern(t, pattern)]
            print(f"    Ticket numbers reduced to {len(reduced_numbers)} to match '{pattern}'.")
            ticket_numbers = reduced_numbers
        if max_tickets > 0:
            reduced_numbers = ticket_numbers[:max_tickets]
            if len(reduced_numbers) < len(ticket_numbers):
                print(f"    Ticket numbers reduced to {len(reduced_numbers)} for {max_tickets} number limit")
                ticket_numbers = reduced_numbers
        ticket_numbers.sort(key=lambda k: (total_size_kb(comment_paths(k)), k))
        return ticket_numbers

    def get_summariser(self, llm, model, do_features):
        if do_features:
            summariser = PydanticFeatureGenerator(llm, model)
        else:
            summariser = PydanticSummariser(llm, model)
        return summariser

    def summarise_tickets(self, ticket_numbers, summariser, overwrite=False):
        """
        Summarises the conversations from the Zendesk support tickets specified by `ticket_numbers`.

        Args:
            ticket_numbers (list): A list of ticket numbers to process.
            llm (LLM): The LLM to use for summarisation.
            model (str): The LLM model name.
            do_features (bool): If True, generate features instead of summaries.
            overwrite (bool, optional): If True, overwrite existing summaries. Defaults to False.

        Returns:
            list: A list of paths to the generated summaries.
        """
        # summariser = self.get_summariser(llm, model, do_features)

        if not overwrite:
            reduced_numbers = [t for t in ticket_numbers
                            if not os.path.exists(summariser.summary_path(t))]
            print(f"    Ticket numbers reduced to {len(reduced_numbers)} unprocessed tickets.")
            ticket_numbers = reduced_numbers

        show_tickets(ticket_numbers, TICKETS_SHOWN)

        t00 = time.time()
        summary_paths = []

        for i, ticket_number in enumerate(ticket_numbers):
            metadata = self.metadata(ticket_number)
            path = summarise_one_ticket(summariser, i, ticket_number, metadata, overwrite)
            if path:
                summary_paths.append(path)

        print("==========================================^^^==========================================")
        print(f"Total duration: {since(t00):.1f} seconds")

        return summary_paths
