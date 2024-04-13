"""
    This module is used to fetch ticket comments from Zendesk.
    https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
"""
from collections import defaultdict
import glob
import json
import os
import pytz
import requests
import shutil
import time
from functools import partial
import pandas as pd
from config import (COMMENTS_DIR, TICKET_INDEX_PATH, TICKET_ALIASES_PATH, TICKET_BATCHES_DIR,
                    METADATA_KEYS, FIELD_KEY_NAMES)
from utils import save_json, save_text, load_json, load_text, iso2date, since

USER = os.environ.get("ZENDESK_USER")
TOKEN = os.environ.get("ZENDESK_TOKEN")
SUBDOMAIN = os.environ.get("ZENDESK_SUBDOMAIN")

assert USER, "missing ZENDESK_USER"
assert TOKEN, "missing ZENDESK_TOKEN"
assert SUBDOMAIN, "missing ZENDESK_SUBDOMAIN"

ZENDESK_API = f"https://{SUBDOMAIN}.zendesk.com/api/v2/"

def make_url(path):
    "Constructs and returns a URL by appending `path` to `ZENDESK_API.`"
    return f"{ZENDESK_API}/{path}"

def url_get(url, params=None):
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

def zd_get(path, params=None):
    "Calls the Zendesk API with the specified path and returns the response parsed as a dictionary."
    url = make_url(path)
    return url_get(url, params)

def fetch_ticket_fields():
    return zd_get("ticket_fields")

def fetch_author(author_id):
    return zd_get(f"users/{author_id}")

def fetch_ticket(ticket_number: int) -> dict:
    """ Fetches Zendesk ticket number `ticket_number`.
        Returns: A dictionary representing the ticket.
        https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
    """
    result = zd_get(f"tickets/{ticket_number}")
    if "error" in result or "ticket" not in result:
        return None
    return result["ticket"]

def fetch_ticket_comments(ticket_number):
    """ Fetches the comments for Zendesk ticket number `ticket_number`.
        Returns: A list of comments for the specified ticket number.
    """
    result = zd_get(f"tickets/{ticket_number}/comments?include=&include_inline_images=false")
    assert "error" not in result, f"^^^ {ticket_number} {result}"

    comments = []
    for i in range(1_000):
        comments.extend(result["comments"])
        url = result.get("next_page")
        if not url:
            break

        result = url_get(url)
        if i > 5:
            # Crazy number of comments.
            print(f" {len(comments)} comments fetched for {ticket_number}: {url}")
            break

    return comments

os.makedirs(COMMENTS_DIR, exist_ok=True)

def _path_func(ticket_number, dir_name, ext):
    "Returns the path give by joining directory name `dir_name` and `ticket_number`."
    return os.path.abspath(os.path.join(dir_name, f"{ticket_number}{ext}"))

def _list_func(dir_name, ext):
    "Returns the files in `dir_name` matching `ext`."
    return sorted(glob.glob(os.path.join(dir_name, f"*{ext}")))

def _make_ticket_path_func(dir_name, ext=""):
    """ Returns path_func, list_func where:
        path_func: function that returns the path give by joining directory name `dir_name` and
        `ticket_number`.
        list_func: function that returns the files in `dir_name` matching `ext`.
    """
    path_func = partial(_path_func, dir_name=dir_name, ext=ext)
    list_func = partial(_list_func, dir_name=dir_name, ext=ext)
    return path_func, list_func

# The functions for constructing paths to ticket data.
get_comments_dir, list_comment_dirs = _make_ticket_path_func(COMMENTS_DIR)

def comment_paths(ticket_number):
    "Returns a sorted list of file paths for the comments in Zendesk ticket `ticket_number`."
    folder = get_comments_dir(ticket_number)
    return sorted(glob.glob(os.path.join(folder, "*.txt")))

# The functions for accessing TICKET_BATCHES_DIR which contains the raw ticket data.
def _batch_path(i): return os.path.join(TICKET_BATCHES_DIR, f"{i:05d}.json")
def _list_batches(): return sorted(glob.glob(os.path.join(TICKET_BATCHES_DIR, "*.json")))

def fetch_all_ticket_batches(ticket_batches_dir, max_pages):
    """Fetches all the Zendesk tickets.

     https://developer.zendesk.com/api-reference/introduction/pagination/
     https://example.zendesk.com/api/v2/tickets.json?page[size]=100
    """
    params = {
        "page[size]": 100,
        "sort_by": "created_at",
    }
    result = zd_get("tickets.json", params=params)
    assert "error" not in result, f"^^^ {result}"

    t0 = time.time()
    num_tickets = 0
    for i in range(max_pages):
        assert "error" not in result, f"i={i}: {result}"
        assert "tickets" in result, f"i={i}: {list(result.keys())}"
        tickets = result["tickets"]
        save_json(_batch_path(i), tickets)
        num_tickets += len(tickets)

        has_more = result.get("meta", {}).get("has_more")
        if not has_more:
            break
        url = result.get("links", {}).get("next")
        if not url:
            break
        result = url_get(url)
        if "error" in result:
            break

        if i % 10 == 1:
            print(f"  Page {i:5}: fetched {num_tickets:6} tickets in {since(t0):6.1f} secs")

def run_on_all_tickets(func):
    "Runs function `func` on all tickets in `TICKET_BATCHES_DIR`."
    batch_paths = _list_batches()
    for path in batch_paths:
        tickets = load_json(path)
        for ticket in tickets:
            func(ticket)

def panderise_date(date):
    "Converts a date to a string."
    return pytz.utc.localize(date)

def extract_metadata(ticket, add_custom_fields=False):
    "Extracts metadata from a Zendesk ticket."
    key_list = [key for key in METADATA_KEYS if key in ticket]
    metadata = {key: ticket[key] for key in key_list}
    metadata["created_at"] = panderise_date(iso2date(metadata["created_at"]))
    metadata["updated_at"] = panderise_date(iso2date(metadata["updated_at"]))

    if add_custom_fields:
        field_list = ticket["custom_fields"]
        custom_dict = {field["id"]: field["value"] for field in field_list
                if field["id"] and field["value"]}
        metadata["custom_fields"] = custom_dict

    return metadata

def download_comments(ticket_number, overwrite=False):
    """ Download the comments in Zendesk ticket `ticket_number` and save them as one text file per
        comment in directory `COMMENTS_DIR/ticket_number`.
    """
    folder = get_comments_dir(ticket_number)
    if not overwrite and os.path.exists(folder):
        return
    comments = fetch_ticket_comments(ticket_number)

    os.makedirs(folder, exist_ok=True)
    has_paths = False
    for i, comment in enumerate(comments):
        body = comment["body"]
        body = body.strip()
        if not any(c.isalnum() for c in body):
            continue
        path = os.path.join(folder, f"comment_{i:03d}.txt")
        save_text(path, body)
        has_paths = True
    if not has_paths:
        os.rmdir(folder)

def format_index_df(df):
    """
    Formats the `df` by sorting it based on specific columns, converting certain columns
    to appropriate data types, and filling missing values with empty strings.
    """
    df["comments_num"] = df["comments_num"].astype(int)
    df["comments_size"] = df["comments_size"].astype(int)
    for column in ["status", "priority", "problem_type", "product_name", "product_version",
                   "recipient", "customer", "region", "subject"]:
        df[column] = df[column].fillna("")
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["updated_at"] = pd.to_datetime(df["updated_at"])
    df = df.sort_values(by=["created_at", "updated_at", "ticket_number"])
    return df

def make_empty_index():
    "Creates an index of Zendesk tickets."
    columns = METADATA_KEYS + ["comments_num", "comments_size"]
    df = pd.DataFrame(columns=columns)
    df.index.name = "ticket_number"
    return df

MAX_PAGES = 10_000
MAX_TICKETS = 10_000

def update_index(df, min_date=None, max_date=None, clean=False):
    """
    Updates an index of Zendesk tickets.

    This function reads ticket numbers, loads metadata, calculates comments information,
    creates aliases for tickets with the same subject and description, and saves the index
    and aliases to files.
    """
    def inRange(ticket):
        created_at = iso2date(ticket["created_at"])
        if min_date and created_at < min_date:
            return False
        if max_date and created_at > max_date:
            return False
        return True

    if clean and os.path.exists(TICKET_BATCHES_DIR):
        shutil.rmtree(TICKET_BATCHES_DIR)
    os.makedirs(TICKET_BATCHES_DIR, exist_ok=True)
    t0 = time.time()

    fetch_all_ticket_batches(TICKET_BATCHES_DIR, MAX_PAGES)
    batch_paths = _list_batches()
    print(f"   Fetched  {len(batch_paths)} batches of tickets in {since(t0):.1f} secs")

    t0 = time.time()
    num_tickets = 0
    for i, batch_path in enumerate(batch_paths):
        ticket_list = load_json(batch_path)
        for ticket in ticket_list[:MAX_TICKETS]:
            if not inRange(ticket):
                continue
            ticket_number = ticket["id"]
            download_comments(ticket_number)
            num_tickets += 1
            if num_tickets % 1_000 == 10:
                dt = since(t0)
                rate = num_tickets / (60.0 * dt + 0.001)
                print(f"Downloaded batch {i:4} of {num_tickets:6} comments in {dt:.1f} secs " +
                      f"({rate:.1f} per min)")
    print(f"Downloaded {num_tickets} comments  in {since(t0):.1f} secs")

    t0 = time.time()
    num_tickets = 0
    num_range = 0
    num_processed = 0
    for i, batch_path in enumerate(batch_paths):
        ticket_list = load_json(batch_path)
        for ticket in ticket_list[:MAX_TICKETS]:
            num_tickets += 1
            if not inRange(ticket):
                continue
            num_range += 1
            ticket_number = ticket["id"]
            if ticket_number in df.index:
                continue
            num_processed += 1
            metadata = extract_metadata(ticket)
            paths = comment_paths(ticket_number)
            metadata["comments_num"] = len(paths)
            metadata["comments_size"] = sum(os.path.getsize(k) for k in paths)
            for key in metadata.keys():
                assert key in df.columns, f"bad key {key}"
            df.loc[ticket_number] = metadata

            if num_tickets % 100_000 == 10_000:
                dt = since(t0)
                period = 10_000 * dt / (num_tickets+1)
                print(f"Indexed batch {i:4}: {num_tickets:6} metadata in {dt:5.1f} secs ({period:.1f} per 10k)")

    print(f"   Indexed {num_processed } of {num_range} of {num_tickets} metadatas in {since(t0):.1f} secs")
    print(f"   Index = {len(df)} tickets")
    assert num_range - num_processed <= len(df)

    df = format_index_df(df)

    # Some Zendeck tickets have the same subject and description, so we need to create aliases
    key_index = {}
    reversed_aliases = defaultdict(list)
    for row in df.itertuples():
        idx = f"{row.comments_num:2}:{row.comments_size:5}:{row.subject}//{row.subject}"
        if idx in key_index:
            reversed_aliases[key_index[idx]].append(row.Index)
        else:
            key_index[idx] = row.Index

    print(f"Writing {len(df)} tickets to {TICKET_INDEX_PATH}")
    df.to_csv(TICKET_INDEX_PATH)
    print(f"Writing {len(reversed_aliases)} aliases to {TICKET_ALIASES_PATH}")
    save_json(TICKET_ALIASES_PATH, reversed_aliases)

    return df, reversed_aliases

def make_fresh_index(in_date=None, max_date=None, clean=False):
    "Creates a fresh index of Zendesk tickets."
    df = make_empty_index()
    return update_index(df, in_date, max_date, clean)

def load_index():
    "Load the ticket index from a TICKET_INDEX_PATH and perform necessary data transformations."
    print(f"  Reading tickets from {TICKET_INDEX_PATH}")
    if not os.path.exists(TICKET_INDEX_PATH):
        df = make_empty_index()
        df.to_csv(TICKET_INDEX_PATH)
        print(f"Created empty {TICKET_INDEX_PATH}")
    else:
        df = pd.read_csv(TICKET_INDEX_PATH, index_col="ticket_number")
        df = format_index_df(df)
        print(f"Loaded {len(df)} tickets from {TICKET_INDEX_PATH}")
    return df

def add_tickets_to_index(df, ticket_numbers):
    "Adds the metadata for the specified `ticket_numbers` to the index `df`."
    ticket_numbers = sorted(set(ticket_numbers))
    new_ticket_numbers, bad_ticket_numbers = [], []
    for ticket_number in ticket_numbers:
        if ticket_number in df.index:
            continue
        ticket = fetch_ticket(ticket_number)
        if not ticket:
            bad_ticket_numbers.append(ticket_number)
            continue
        new_ticket_numbers.append(ticket_number)
        metadata = extract_metadata(ticket)
        download_comments(ticket_number)
        comments_paths = comment_paths(ticket_number)
        metadata["comments_num"] = len(comments_paths)
        metadata["comments_size"] = sum(os.path.getsize(k) for k in comments_paths)
        df.loc[ticket_number] = metadata
    if new_ticket_numbers:
        print(f"Adding {len(new_ticket_numbers)} new tickets to {TICKET_INDEX_PATH}.")
        df = format_index_df(df)
        df.to_csv(TICKET_INDEX_PATH)
    return df, new_ticket_numbers, bad_ticket_numbers
