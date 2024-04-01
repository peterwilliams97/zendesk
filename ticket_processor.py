""" Functions to process Zendesk tickets.
"""
import glob
import os
import time
import sys
from utils import totalSizeKB, currentTime, since, loadText
from zendesk_wrapper import commentPaths, loadIndex
from evaluate_summary import summariseTicket

def ticketHasPattern(ticket_number, pattern):
    "Returns True if any of the comments for ticket with number `ticket_number` contains `pattern`."
    regex = re.compile(pattern, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    paths = commentPaths(ticket_number)
    return any(regex.search(loadText(path)) for path in paths)

TICKETS_SHOWN = 5   # Number of tickets to show in the summary.

def listTickets(metadatas):
    "Prints the ticket information for each metadata in the given list."
    for i, metadata in enumerate(metadatas):
        ticket_number = metadata.ticket_number.astype(int)
        created_at = metadata.created_at
        status = metadata.status
        subject = metadata.subject
        priority = metadata.priority
        number = metadata.comments_num
        size = metadata.comments_size / 1024
        print(f"{i:3}: {ticket_number:8} {created_at.date()} {status:8} {priority:7} {number:3} {size:4.1f} {subject[:100]}")

def showTickets(ticket_numbers, num_shown):
    """ Display summaries of the comments in the Zendesk tickets with numbers `ticket_numbers`.
        Show `num_shown` tickets from the start and end of the list.
    """
    def showOne(i):
        ticket_number = ticket_numbers[i]
        paths = commentPaths(ticket_number)
        print(f"{i:8}: {ticket_number:8} {len(paths):3} comments {totalSizeKB(paths):5.2f} kb")

    if len(ticket_numbers) <= num_shown:
        for i in range(len(ticket_numbers)):
            showOne(i)
        return

    half = (num_shown + 1 ) // 2
    for i in range(half):
        showOne(i)
    print("    ...")
    for i in range(len(ticket_numbers) - half + 1, len(ticket_numbers)):
        showOne(i)

def summariseOneTicket(summariser, i, ticket_number, metadata, overwrite):
    commentCount = len(commentPaths(ticket_number))
    commentSize = totalSizeKB(commentPaths(ticket_number))

    print(f"{i:2}: ticket_number={ticket_number:8} {commentCount:3} comments {commentSize:7.3f} kb {currentTime()}",
        flush=True)

    summaryPath = summariser.summaryPath(ticket_number)

    if not overwrite and os.path.exists(summaryPath):
        print(f"   skipping ticket {ticket_number}. Already processed. '{summaryPath}'",
            flush=True)
        return None   # Skip tickets that have already been summarised.

    t0 = time.time()
    try:
        summary, ok = summariseTicket(summariser, ticket_number, metadata)
    except Exception as e:
        print(f"Error processing ticket {ticket_number}: {e}", file=sys.stderr)
        raise

    if not ok:
        print(f"  Could not process {ticket_number}: {summary}.", flush=True)
        return None

    description = f"{commentCount} comments {commentSize:5.2f} kb {since(t0):4.1f} sec summary={len(summary)} chars"

    with open(summaryPath, "w") as f:
        print(f"Zendesk info: ticket {ticket_number}: {description} --------------------------------",
            file=f)
        print(summary, file=f)
    print(f"  {description} saved to {summaryPath}", flush=True)
    return summaryPath

class ZendeskData:
    """
    A class that represents Zendesk data and provides methods to interact with it.

    Attributes:
        df (DataFrame): The Zendesk data as a pandas DataFrame.

    Methods:
        ticketNumbers(): Returns a list of ticket numbers.
        ticket(ticket_number): Returns the ticket with the specified ticket number.
        metadata(ticket_number): Returns the metadata of the ticket with the specified ticket number.
        ticketHasPriority(ticket_number, priority): Returns True if the ticket with the specified
                ticket number has the specified priority.
        summariseTickets(ticket_numbers, llm, model, structured, overwrite=False): Summarizes the
                conversations from the Zendesk support tickets specified by `ticket_numbers`.
    """
    def __init__(self):
        df = loadIndex()
        self.df = df

    def ticketNumbers(self):
        return list(self.df.index)

    def ticket(self, ticket_number):
        return self.df.loc[ticket_number]

    def metadata(self, ticket_number):
        return self.df.loc[ticket_number]

    def ticketHasPriority(self, ticket_number, priority):
        "Returns True if ticket with number `ticket_number` has priority `priority`."
        metadata = self.metadata(ticket_number)
        return metadata.priority == priority

    def filterTickets(self, ticket_numbers, pattern, priority, max_size, max_tickets):
        print(f"  Filtering {len(ticket_numbers)} tickets.")
        if priority:
            reduced_numbers = [t for t in ticket_numbers if self.ticketHasPriority(t, priority)]
            print(f"    Ticket numbers reduced to {len(reduced_numbers)} to match priority '{priority}'.")
            ticket_numbers = reduced_numbers
        if max_size > 0:
            reduced_numbers = [k for k in ticket_numbers if totalSizeKB(commentPaths(k)) <= max_size]
            if len(reduced_numbers) < len(ticket_numbers):
                print(f"    Ticket numbers reduced to {len(reduced_numbers)} for {max_size} kb size limit")
                ticket_numbers = reduced_numbers
        if pattern:
            reduced_numbers = [t for t in ticket_numbers if ticketHasPattern(t, pattern)]
            print(f"    Ticket numbers reduced to {len(reduced_numbers)} to match '{pattern}'.")
            ticket_numbers = reduced_numbers
        if max_tickets > 0:
            reduced_numbers = ticket_numbers[:max_tickets]
            if len(reduced_numbers) < len(ticket_numbers):
                print(f"    Ticket numbers reduced to {len(reduced_numbers)} for {max_tickets} number limit")
                ticket_numbers = reduced_numbers
        return ticket_numbers

    def summariseTickets(self, ticket_numbers, llm, model, summariser_type, overwrite=False):
        """
        Summarises the conversations from the Zendesk support tickets specified by `ticket_numbers`.

        Args:
            ticket_numbers (list): A list of ticket numbers to process.
            llm (LLM): The LLM to use for summarisation.
            structured (bool): If True, use the StructuredSummariser class to summarise the tickets.
                               Otherwise, use the PlainSummariser class.
            overwrite (bool, optional): If True, overwrite existing summaries. Defaults to False.
            model (str): The LLM model name.

        Returns:
            list: A list of paths to the generated summaries.
        """
        summariser = summariser_type(llm, model)

        if not overwrite:
            reduced_numbers = [t for t in ticket_numbers
                            if not os.path.exists(summariser.summaryPath(t))]
            print(f"    Ticket numbers reduced to {len(reduced_numbers)} unprocessed tickets.")
            ticket_numbers = reduced_numbers

        showTickets(ticket_numbers, TICKETS_SHOWN)

        t00 = time.time()
        summaryPaths = []

        for i, ticket_number in enumerate(ticket_numbers):
            metadata = self.metadata(ticket_number)
            summaryPath = summariseOneTicket(summariser, i, ticket_number, metadata, overwrite)
            if summaryPath:
                summaryPaths.append(summaryPath)

        print("==========================================^^^==========================================")
        print(f"Total duration: {since(t00):.1f} seconds")

        return summaryPaths
