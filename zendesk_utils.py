"""
    This module provides utility functions for working with Zendesk tickets.
"""
import os
import glob
import time
from config import DATA_DIR

def saveText(path, text):
    "Save `text` to a file `path`."
    with open(path, "w") as f:
        f.write(text)

def totalSizeKB(paths):
    "Returns the total size in kilobytes of the files specified by `paths`."
    return sum(os.path.getsize(path) for path in paths) / 1024

def currentTime():
    "Returns the current time in the format 'dd/mm/YYYY HH:MM:SS'."
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

def commentPaths(ticketNumber):
    "Returns a sorted list of file paths for the comments in Zendesk ticket `ticketNumber`."
    ticketDir = os.path.join(DATA_DIR, ticketNumber)
    return sorted(glob.glob(os.path.join(ticketDir, "*.txt")))

def storedTicketNumbers():
    "Returns a sorted list of the ticket numbers stored in `DATA_DIR`."
    assert DATA_DIR, "Set DATA_DIR in config.py before calling storedTicketNumbers()"
    return sorted(os.path.basename(path) for path in glob.glob(os.path.join(DATA_DIR, "*")))
