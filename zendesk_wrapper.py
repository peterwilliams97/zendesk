"""
    This module is used to fetch ticket comments from Zendesk.
    https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
"""
import requests
import json
import os
import time
import re
import shutil
from collections import defaultdict
from functools import partial
import glob
import datetime
import pandas as pd
from config import (COMMENTS_DIR, TICKET_INDEX_PATH, TICKET_ALIASES_PATH,
                     METADATA_KEYS, FIELD_KEY_NAMES, TICKET_BATCHES_DIR)
from utils import saveJson, saveText, loadJson, loadText, isoToDate, since

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
    return response.json()

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
    if "error" in result or "ticket" not in result:
        return None
    return result["ticket"]

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
        if i > 5:
            # Crazy number of comments.
            print(f" {len(comments)} comments fetched for {ticket_number}: {url}")
            break

    return comments

os.makedirs(COMMENTS_DIR, exist_ok=True)

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

def commentPaths(ticket_number):
    "Returns a sorted list of file paths for the comments in Zendesk ticket `ticket_number`."
    comments_dir = commentsDir(ticket_number)
    return sorted(glob.glob(os.path.join(comments_dir, "*.txt")))

def _batchPath(i): return os.path.join(TICKET_BATCHES_DIR, f"{i:05d}.json")
def _listBatches(): return sorted(glob.glob(os.path.join(TICKET_BATCHES_DIR, "*.json")))

def fetchAllTicketBatches(ticket_batches_dir, max_pages):
    """Fetches all the Zendesk tickets.

     https://developer.zendesk.com/api-reference/introduction/pagination/
     https://example.zendesk.com/api/v2/tickets.json?page[size]=100
    """
    params = {
        "page[size]": 100,
        "sort_by": "created_at",
    }
    result = zdGet("tickets.json", params=params)
    assert "error" not in result, f"^^^ {result}"

    t0 = time.time()
    num_tickets = 0
    for i in range(max_pages):
        assert "error" not in result, f"i={i}: {result}"
        assert "tickets" in result, f"i={i}: {list(result.keys())}"
        tickets = result["tickets"]
        saveJson(_batchPath(i), tickets)
        num_tickets += len(tickets)

        has_more = result.get("meta", {}).get("has_more")
        if not has_more:
            break
        url = result.get("links", {}).get("next")
        if not url:
            break
        result = urlGet(url)
        if "error" in result:
            break

        if i % 10 == 1:
            print(f"  Page {i:5}: fetched {num_tickets:6} tickets in {since(t0):6.1f} secs")

def extractMetadata(ticket, add_custom_fields=False):
    "Extracts metadata from a Zendesk ticket."
    key_list = [key for key in METADATA_KEYS if key in ticket]
    metadata = {key: ticket[key] for key in key_list}
    metadata["created_at"] = isoToDate(metadata["created_at"])
    metadata["updated_at"] = isoToDate(metadata["updated_at"])

    if add_custom_fields:
        field_list = ticket["custom_fields"]
        custom_dict = {field["id"]: field["value"] for field in field_list
                if field["id"] and field["value"]}
        metadata["custom_fields"] = custom_dict

    return metadata

def downloadComments(ticket_number, overwrite=False):
    """ Download the comments in Zendesk ticket `ticket_number` and save them as one text file per
        comment in directory `COMMENTS_DIR/ticket_number`.
    """
    comments_dir = commentsDir(ticket_number)
    if not overwrite and os.path.exists(comments_dir):
        return
    comments = fetchTicketComments(ticket_number)

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

def formatIndexDf(df):
    """
    Formats the `df` by sorting it based on specific columns, converting certain columns
    to appropriate data types, and filling missing values with empty strings.
    """
    df = df.sort_values(by=["created_at", "updated_at", "ticket_number"])
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df["comments_num"] = df["comments_num"].astype(int)
    df["comments_size"] = df["comments_size"].astype(int)
    for column in ["status", "priority", "problem_type", "product_name", "product_version",
                   "recipient", "customer", "region", "subject"]:
        df[column] = df[column].fillna("")
    return df

def makeEmptyIndex():
    "Creates an index of Zendesk tickets."
    columns = METADATA_KEYS + ["comments_num", "comments_size"]
    df = pd.DataFrame(columns=columns)
    df.index.name = "ticket_number"
    return df

MAX_PAGES = 10_000
MAX_TICKETS = 10_000

def updateIndex(df, min_date=None, max_date=None, clean=False):
    """
    Updates an index of Zendesk tickets.

    This function reads ticket numbers, loads metadata, calculates comments information,
    creates aliases for tickets with the same subject and description, and saves the index
    and aliases to files.
    """
    def inRange(ticket):
        created_at = isoToDate(ticket["created_at"])
        if min_date and created_at < min_date:
            return False
        if max_date and created_at > max_date:
            return False
        return True

    if clean and os.path.exists(TICKET_BATCHES_DIR):
       shutil.rmtree(TICKET_BATCHES_DIR)
    os.makedirs(TICKET_BATCHES_DIR, exist_ok=True)
    t0 = time.time()

    fetchAllTicketBatches(TICKET_BATCHES_DIR, MAX_PAGES)
    batch_paths = _listBatches()
    print(f"   Fetched  {len(batch_paths)} batches of tickets in {since(t0):.1f} secs")

    t0 = time.time()
    num_tickets = 0
    for i, batch_path in enumerate(batch_paths):
        ticket_list = loadJson(batch_path)
        for ticket in ticket_list[:MAX_TICKETS]:
            if not inRange(ticket):
                continue
            ticket_number = ticket["id"]
            downloadComments(ticket_number)
            num_tickets += 1
            if num_tickets % 1_000 == 10:
                dt = since(t0)
                rate = num_tickets / (60.0 * dt + 0.001)
                print(f"Downloaded batch {i:4} of {num_tickets:6} comments in {dt:.1f} secs ({rate:.1f} per min)")
    print(f"Downloaded {num_tickets} comments  in {since(t0):.1f} secs")

    t0 = time.time()
    num_tickets = 0
    num_range = 0
    num_processed = 0
    for i, batch_path in enumerate(batch_paths):
        ticket_list = loadJson(batch_path)
        for ticket in ticket_list[:MAX_TICKETS]:
            num_tickets += 1
            if not inRange(ticket):
                continue
            num_range += 1
            ticket_number = ticket["id"]
            if ticket_number in df.index:
                continue
            num_processed += 1
            metadata = extractMetadata(ticket)
            comments_paths = commentPaths(ticket_number)
            metadata["comments_num"] = len(comments_paths)
            metadata["comments_size"] = sum(os.path.getsize(k) for k in comments_paths)
            for key in metadata.keys():
                assert key in df.columns, f"bad key {key}"
            df.loc[ticket_number] = metadata

            if num_tickets % 100_000 == 10_000:
                dt = since(t0)
                period = 10_000 * dt / (num_tickets+1)
                print(f"Indexed batch {i:4}: {num_tickets:6} metadata in {dt:5.1f} secs ({period:.1f} per 10k)")
                df = formatIndexDf(df)
                n = num_tickets // 1_000
                path = f"aaa_{n:06d}.csv"
                df.to_csv(path)
                print(f"   Saved {path}")

    print(f"   Indexed {num_processed } of {num_range} of {num_tickets} metadatas in {since(t0):.1f} secs")
    print(f"   Index = {len(df)} tickets")
    assert num_range - num_processed <= len(df)

    df = formatIndexDf(df)

    # Some Zendeck tickets have the same subject and description, so we need to create aliases
    key_index = {}
    reversed_aliases = defaultdict(list)
    for row in df.itertuples():
        idx = f"{row.comments_num:2}:{row.comments_size:5}:{row.subject}//{row.subject}"
        if idx in key_index:
            reversed_aliases[key_index[idx]].append(row.Index)
        else:
            key_index[idx] = row.Index
    return df, reversed_aliases

def makeIndex(in_date=None, max_date=None, clean=False):
    "Creates a fresh index of Zendesk tickets."
    df = makeEmptyIndex()
    return updateIndex(df, in_date, max_date, clean)

def loadIndex(index_path):
    "Load the ticket index from a CSV file and perform necessary data transformations."
    print(f"  Reading tickets from {index_path}")
    if not os.path.exists(index_path):
        return makeEmptyIndex()
    df = pd.read_csv(index_path, index_col="ticket_number")
    df = formatIndexDf(df)
    return df

def addTicketsToIndex(df, ticket_numbers):
    "Adds the metadata for the specified `ticket_numbers` to the index `df`."
    ticket_numbers = sorted(set(ticket_numbers))
    new_ticket_numbers, bad_ticket_numbers = [], []
    for ticket_number in ticket_numbers:
        if ticket_number in df.index:
            continue
        ticket = fetchTicket(ticket_number)
        if not ticket:
            bad_ticket_numbers.append(ticket_number)
            continue
        new_ticket_numbers.append(ticket_number)
        metadata = extractMetadata(ticket)
        downloadComments(ticket_number)
        comments_paths = commentPaths(ticket_number)
        metadata["comments_num"] = len(comments_paths)
        metadata["comments_size"] = sum(os.path.getsize(k) for k in comments_paths)
        print(f"  {ticket_number}: {metadata}")
        df.loc[ticket_number] = metadata
    return new_ticket_numbers, bad_ticket_numbers
