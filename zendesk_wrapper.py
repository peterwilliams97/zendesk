"""
    This module is used to fetch ticket comments from Zendesk.
    https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
"""
import requests
import json
import os
import time
import re
from collections import defaultdict
from functools import partial
import glob
import datetime
import pandas as pd
from config import (COMMENTS_DIR, TICKET_DIR, METADATA_DIR, TICKET_INDEX, TICKET_ALIASES,
                    METADATA_KEYS, FIELD_KEY_NAMES)
from utils import saveJson, saveText, loadJson, loadText, isoToDate

USER = os.environ.get("ZENDESK_USER")
TOKEN = os.environ.get("ZENDESK_TOKEN")
SUBDOMAIN = os.environ.get("ZENDESK_SUBDOMAIN")

assert USER, "missing ZENDESK_USER"
assert TOKEN, "missing ZENDESK_TOKEN"
assert SUBDOMAIN, "missing ZENDESK_SUBDOMAIN"

ZENDESK_API = f"https://{SUBDOMAIN}.zendesk.com/api/v2/"

def makeUrl(path):
    "Constructs and returns a URL by appending `path` to `ZENDESK_API.`"
    return f"{ZENDESK_API}/{path}"

def urlGet(url, params=None):
    """Sends a GET request to `url` with authentication and headers.
        Returns: The JSON response parsed as a dictionary.
        Raises exceptions if the GET request fails or the response is not valid JSON.
    """
    auth = (f"{USER}/token", TOKEN)
    headers = {"Content-Type": "application/json"}
    if params:
        response = requests.request("GET", url, auth=auth, headers=headers, params=params)
    else:
        response = requests.request("GET", url, auth=auth, headers=headers)
    return json.loads(response.text)

def zdGet(path, params=None):
    "Calls the Zendesk API with the specified path and returns the response parsed as a dictionary."
    url = makeUrl(path)
    return urlGet(url, params)

def fetchTicketFields():
    return zdGet("ticket_fields")

def fetchAuthor(author_id):
    return zdGet(f"users/{author_id}")

def fetchTicket(ticket_number):
    """ Fetches Zendesk ticket number `ticket_number`.
        Returns: A dictionary representing the ticket.
        https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
    """
    result = zdGet(f"tickets/{ticket_number}")
    assert "error" not in result, f"^^^ {ticket_number} {result}"
    return result

def fetchTicketComments(ticket_number):
    """ Fetches the comments for Zendesk ticket number `ticket_number`.
        Returns: A list of comments for the specified ticket number.
    """
    result = zdGet(f"tickets/{ticket_number}/comments?include=&include_inline_images=false")
    assert "error" not in result, f"^^^ {ticket_number} {result}"

    comments = []
    for i in range(1_000):
        comments.extend(result["comments"])
        url = result.get("next_page")
        if not url:
            break
        result = urlGet(url)

    return comments

def fetchTicketNumbersAfterDate(start_date):
    "Fetches the Zendesk tickts created after `start_date`."
    if start_date:
        query = f"type:ticket created>{start_date}"
    else:
        query = f"type:ticket"

    params = {
        "query": query,
        "sort_by": "created_at",
        "sort_order": "asc",
    }
    result = zdGet(f"search.json", params=params)
    assert "error" not in result, f"^^^ {result}"

    ticket_created = {}
    latest_created = start_date

    for i in range(1_000):
        assert "error" not in result, f"i={i}: {result}\n\turl={url}"
        for ticket in result['results']:
            created_at = isoToDate(ticket["created_at"])
            ticket_created[ticket["id"]] = created_at
            if not latest_created or created_at > latest_created:
                latest_created = created_at

        url = result.get("next_page")
        if not url:
            break
        result = urlGet(url)
        if "error" in result:
            break

    return ticket_created, latest_created

def fetchTicketNumbers(max_batches, start_date=None):
    """ Fetches up to `max_batches` of Zendesk ticket numbers starting from `start_date`.
        Set max_batches to 0 to fetch all tickets.   .
        Set start_date to None for remove the date filter.
        Returns: A list of ticket numbers sorted by creation date.
    """
    if max_batches == 0:
        max_batches = 1_000
    all_ticket_created = {}
    t0 = time.time()
    for download in range(max_batches):
        print(f"fetchTicketNumbers {download:2}: ", end="", flush=True)
        ticket_created, latest_created = fetchTicketNumbersAfterDate(start_date)
        for ticket_id, created_at in ticket_created.items():
            all_ticket_created[ticket_id] = created_at
        dt = time.time() - t0
        print(f"{start_date} - {latest_created} : {len(ticket_created):4}, {len(all_ticket_created):4} ({dt:.1f} secs)",
              flush=True)
        if latest_created == start_date:
            break
        start_date = latest_created
    return sorted(all_ticket_created.keys(), key=lambda k: (all_ticket_created[k], k))

os.makedirs(TICKET_DIR, exist_ok=True)
os.makedirs(COMMENTS_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)

def _pathFunc(ticket_number, dir_name, ext):
    "Returns the path give by joining directory name `dir_name` and `ticket_number`."
    return os.path.abspath(os.path.join(dir_name, f"{ticket_number}{ext}"))

def _listFunc(dir_name, ext):
    "Returns the files in `dir_name` matching `ext`."
    return sorted(glob.glob(os.path.join(dir_name, f"*{ext}")))

def _makeTicketPathFunc(dir_name, ext=""):
    """ Returns path_func, list_func where:
        path_func: function that returns the path give by joining directory name `dir_name` and
        `ticket_number`.
        list_func: function that returns the files in `dir_name` matching `ext`.
    """
    path_func = partial(_pathFunc, dir_name=dir_name, ext=ext)
    list_func = partial(_listFunc, dir_name=dir_name, ext=ext)
    return path_func, list_func

# The functions for constructing paths to ticket data.
commentsDir, listCommentDirs = _makeTicketPathFunc(COMMENTS_DIR)
metadataPath, listMetadata = _makeTicketPathFunc(METADATA_DIR, ".json")
ticketPath, listTickets = _makeTicketPathFunc(TICKET_DIR, ".json")

def commentPaths(ticket_number):
    "Returns a sorted list of file paths for the comments in Zendesk ticket `ticket_number`."
    comments_dir = commentsDir(ticket_number)
    return sorted(glob.glob(os.path.join(comments_dir, "*.txt")))

def loadRawMetadata(ticket_number):
    return loadJson(metadataPath(ticket_number))

RE_NON_ALPHANUM = re.compile(r"[^a-zA-Z0-9]+")

def compressKey(key):
    """ Compresses a `key` by replacing non-alphanumeric characters with underscores and converting
        it to lowercase.
    """
    return RE_NON_ALPHANUM.sub("_", key).lower()

def getVal(keyVals, key):
    val = keyVals.get(key)
    if not val:
        return "Unknown"
    if isinstance(val, str):
        val = val[:500]
    return val

def loadMetadata(ticket_number):
    keyVals = loadRawMetadata(ticket_number)
    metadata = {}
    for name in METADATA_KEYS:
        metadata[compressKey(name)] = getVal(keyVals, name)
    for key, name in FIELD_KEY_NAMES.items():
        metadata[compressKey(name)] = getVal(keyVals, key)
    return metadata

def metadataInfo():
    info = {}
    for name in METADATA_KEYS:
        info[compressKey(name)] = name
    for key, name in FIELD_KEY_NAMES.items():
        info[compressKey(name)] = name
    return info

def extractMetadata(ticket):
    fields = ticket["fields"]
    keys = [key for key in METADATA_KEYS if key in ticket]
    fields = [field for field in fields if field["id"] and field["value"]]

    metadata = {key: ticket[key] for key in keys}
    for field in fields:
        metadata[field["id"]] = field["value"]
    return metadata

def saveComments(ticket_number):
    """ Download the comments in Zendesk ticket `ticket_number` and save them as one text file per
        comment in directory `COMMENTS_DIR/ticket_number`.
    """
    comments = fetchTicketComments(ticket_number)
    comments_dir = os.path.join(COMMENTS_DIR, str(ticket_number))
    os.makedirs(comments_dir, exist_ok=True)
    has_paths = False
    for i, comment in enumerate(comments):
        body = comment["body"]
        body = body.strip()
        if not any(c.isalnum() for c in body):
            continue
        path = os.path.join(comments_dir, f"comment_{i:03d}.txt")
        saveText(path, body)
        has_paths = True
    if not has_paths:
        os.rmdir(comments_dir)

def saveRawTicket(ticket_number):
    """ Download the comments in Zendesk ticket `ticket_number` and save them as one text file per
        comment in directory `TICKET_DIR/ticket_number`.
    """
    whole_ticket = fetchTicket(ticket_number)
    ticket = whole_ticket["ticket"]
    # print(f"^^^t {ticket_number} {list(ticket.keys())}")
    saveJson(ticketPath(ticket_number), ticket)
    return ticket

def saveTicket(ticket_number, overwrite=False):
    if not overwrite and os.path.exists(metadataPath(ticket_number)):
        # print(f"^^^* {ticket_number} already saved")
        return None

    ticket = saveRawTicket(ticket_number)
    saveComments(ticket_number)
    metadata = extractMetadata(ticket)
    saveJson(metadataPath(ticket_number), metadata)

    return metadata

def readTicketDates():
    pathList = listMetadata()
    dates = []
    keyCounts = defaultdict(int)
    keySets = defaultdict(set)
    for path in pathList:
        keyVals = loadJson(path)
        created_at = keyVals["created_at"]
        if keyVals["status"] != "closed": # !@#$
            continue
        dates.append(isoToDate(created_at))
        for key in keyVals:
            keyCounts[key] += 1
            keySets[key].add(keyVals[key])
    dates.sort()
    keySets = {key: {v for v in keySets[key] if v} for key in keySets}
    keyLists = {key: sorted(vals) for key, vals in keySets.items()}
    return dates, keyCounts, keyLists

def readTicketNumbers():
    paths = listMetadata()
    return [int(os.path.basename(path)[:-5]) for path in paths]

def downloadTickets(ticket_numbers, overwrite):
    """ Downloads tickets and comments from Zendesk for the given ticket numbers.

        ticket_numbers: A list of ticket numbers to download.
        overwrite : Flag indicating whether to overwrite existing files.
    """
    assert len(ticket_numbers) == len(set(ticket_numbers)), f"duplicate ticket numbers"
    print(f"Downloading {len(ticket_numbers)} tickets")

    t0 = time.time()
    dt = lambda: time.time() - t0

    last_date = None
    for i, ticket_number in enumerate(ticket_numbers):
        metadata = saveTicket(ticket_number, overwrite)
        if metadata:
            last_date = metadata["created_at"]
        if i % 1000 == 10:
            print(f"\n  Downloaded {i} tickets to {isoToDate(last_date)} ({dt():.1f} secs) ", end="",
                  flush=True)
        elif i % 100 == 10:
            print(f"{i}, ", end="", flush=True)
    print(f"\n  Downloaded {len(ticket_numbers)} tickets to {isoToDate(last_date)} ({dt():.1f} secs)",
          flush=True)

def makeIndex():
    ticket_numbers = readTicketNumbers()
    print(f"Found {len(ticket_numbers)} tickets. {ticket_numbers[:3]}...{ticket_numbers[-3:]}")
    for i, ticket_number in enumerate(ticket_numbers):
        metadata2 = loadMetadata(ticket_number)
        comments_paths = commentPaths(ticket_number)
        metadata = {}
        for k, v in metadata2.items():
            if k != "description":
                metadata[k] = v
        metadata["comments_num"] = len(comments_paths)
        metadata["comments_size"] = sum(os.path.getsize(k) for k in comments_paths)
        if i == 0:
            df = pd.DataFrame(metadata, index=[ticket_number])
        else:
            df.loc[ticket_number] = metadata

    print(f"Writing {len(df)} tickets to {TICKET_INDEX}")
    df['ticket_number'] = df.index
    df = df.sort_values(by=['created_at', 'updated_at','ticket_number'])
    df.to_csv(TICKET_INDEX)
    print

    # Some Zendeck tickets have the same subject and description, so we need to create aliases
    key_index = {}
    reversed_aliases = defaultdict(list)
    for row in df.itertuples():
        idx = f"{row.comments_num:2}:{row.comments_size:5}:{row.subject}//{row.subject}"
        if idx in key_index:
            reversed_aliases[key_index[idx]].append(row.Index)
        else:
            key_index[idx] = row.Index
    saveJson(TICKET_ALIASES, reversed_aliases)
    print(f"Writing {len(reversed_aliases)} aliases to {TICKET_ALIASES}")

def loadIndex():
    print(f"Reading tickets from {TICKET_INDEX}")
    df = pd.read_csv(TICKET_INDEX, index_col=0)
    df['created_at']= pd.to_datetime(df['created_at'])
    df['updated_at']= pd.to_datetime(df['updated_at'])
    df['comments_num'] = df['comments_num'].astype(int)
    df['comments_size'] = df['comments_size'].astype(int)
    return df
