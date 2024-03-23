"""
    This script summarises the comments from the Zendesk support tickets specified by the ticket
    numbers. The summaries are written to the "structured.summaries" directory using the specified
    LLM model. The script is intended to be called from another script that specifies the ticket
    numbers and the LLM model to use.
"""
import glob
import os
import time
from config import DATA_DIR
from zendesk_utils import commentPaths, totalSizeKB, currentTime
from structured_queries import StructuredSummariser
from plain_queries import PlainSummariser

#
# Test case.
#
# ['1259693', '1216136', '1196141', '1260221', '1116722', '1280919']
#    0: 1259693    7 comments   2.888 kb
#    1: 1216136   26 comments  20.715 kb
#    2: 1196141  122 comments  81.527 kb
#    3: 1260221  106 comments 126.619 kb
#    4: 1116722  288 comments 190.168 kb
#    5: 1280919  216 comments 731.220 kb

# TODO Remove this when the script is ready to be used.
MAX_SIZE = 100  # Maximum size of ticket comments in kilobytes.
MAX_TICKETS = 1 # Maximum number of tickets to process.

def summariseTickets(ticketNumbers, llm, model, structured,
                     overwrite=False, max_size=MAX_SIZE, max_tickets=MAX_TICKETS):
    """
    Summarises the conversations from the Zendesk support tickets specified by `ticketNumbers`.

    ticketNumbers: A list of ticket numbers to process.
    llm: The LLM to use for summarisation.
    structured: If True, use the StructuredSummariser class to summarise the tickets. Otherwise, use
        the PlainSummariser class.
    owerwrite: If True, overwrite existing summaries.
    model: The LLM model name.
    max_size: The maximum size of ticket comments in kilobytes.
    max_tickets: The maximum number of tickets to process.
    """
    if structured:
        summariser = StructuredSummariser(llm, model)
    else:
        summariser = PlainSummariser(llm, model)

    ticketNumbers = sorted(ticketNumbers, key=lambda k: (totalSizeKB(commentPaths(k)), k))

    if max_size > 0:
        reducedNumbers = [k for k in ticketNumbers if totalSizeKB(commentPaths(k)) < max_size]
        if reducedNumbers < ticketNumbers:
            print(f"Ticket numbers reduced to {len(reducedNumbers)} for {max_size} kb size limit")
            ticketNumbers = reducedNumbers

    for i, ticketNumber in enumerate(ticketNumbers):
        paths = commentPaths(ticketNumber)
        print(f"{i:4}: {ticketNumber:8} {len(paths):3} comments {totalSizeKB(paths):5.2f} kb")

    if max_tickets > 0:
        reducedNumbers = ticketNumbers[:max_tickets]
        if reducedNumbers < ticketNumbers:
            print(f"Ticket numbers reduced to {len(reducedNumbers)} for {max_tickets} number limit")
            ticketNumbers = reducedNumbers

    print(ticketNumbers)

    t00 = time.time()
    summaries = {}
    durations = {}
    commentCounts = {}
    commentSizes = {}
    summaryPaths = []

    for i, ticketNumber in enumerate(ticketNumbers):
        commentCount = len(commentPaths(ticketNumber))
        commentSize = totalSizeKB(commentPaths(ticketNumber))

        print(f"{i:2}: ticketNumber={ticketNumber:8} {commentCount:3} comments {commentSize:7.3f} kb {currentTime()}",
            flush=True)

        summaryPath = summariser.summaryPath(ticketNumber)

        if not overwrite and os.path.exists(summaryPath):
            print(f"   skipping ticket {ticketNumber}. Already processed. '{summaryPath}'",
            flush=True)
            continue   # Skip tickets that have already been summarised.

        t0 = time.time()
        summary = summariser.summariseTicket(ticketNumber)
        duration = time.time() - t0
        description = f"{commentCount} comments {commentSize:5.2f} kb {duration:4.1f} sec summary={len(summary)} chars"

        print(f"  {description}", flush=True)

        with open(summaryPath, "w") as f:
            print(f"Summary: ticket {ticketNumber}: {description} --------------------------------",
                file=f)
            print(summary, file=f)
        summaryPaths.append(summaryPath)

        summaries[ticketNumber] = summary
        durations[ticketNumber] = duration
        commentCounts[ticketNumber] = commentCount
        commentSizes[ticketNumber] = commentSize

    duration = time.time() - t00
    print("==========================================^^^==========================================")
    print(f"Total duration: {duration:.1f} seconds")

    return summaryPaths
