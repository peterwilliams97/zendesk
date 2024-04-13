"""
    This module provides utility functions for working with Zendesk tickets.
"""
import os
import json
import re
import time
import datetime

def save_text(path, text):
    "Save `text` to file `path`."
    with open(path, "w") as f:
        f.write(text)

def load_text(path):
    with open(path, "r") as f:
        return f.read()

def save_json(path, obj):
    "Save `obj` as JSON to file `path`."
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)

def load_json(path):
    "Load and return a JSON object from file `path`."
    with open(path, "r") as f:
        return json.load(f)

def total_size_kb(paths):
    "Returns the total size in kilobytes of the files specified by `paths`."
    return sum(os.path.getsize(path) for path in paths) / 1024

def current_time():
    "Returns the current time in the format 'dd/mm/YYYY HH:MM:SS'."
    now = datetime.datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

def iso2date(text):
    """Convert "2017-01-30T20:48:26Z" to a time tuple."""
    if not text:
        return "[Unknown date]"
    date = datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ")
    # Avoid ValueError: Cannot mix tz-aware with tz-naive values
    date = date.replace(tzinfo=None)
    return date

def list_index(arr, k):
    "Returns the index of `k` in `arr` or the length of `arr` if `k` is not found."
    try:
        i = arr.index(k)
    except ValueError:
        i = len(arr)
    return i

def disjunction(patterns):
    """
    Returns a regular expression pattern that matches any of the given patterns.

    Example:
        >>> patterns = ['apple', 'banana', 'cherry']
        >>> disjunction(patterns)
        '(?:apple|banana|cherry)'
    """
    disjunct = "|".join(patterns)
    return f"(?:{disjunct})"

def regex_compile(pattern):
    "Returns `pattern` compiled to a regular expression. The pattern is case-insensitive."
    return re.compile(pattern, re.IGNORECASE)

# Copied from https://github.com/madisonmay/CommonRegex
# License: MIT
PATTERN_DATE = '(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}'
PATTERN_TIME = '\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?'

RE_DATE = regex_compile(PATTERN_DATE)
RE_TIME = regex_compile(PATTERN_TIME)
RE_YEAR = regex_compile(r"\s+(\d{4})\b")

def text_lines(text):
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    return lines

def since(t0):
    return time.time() - t0

def deduplicate(array):
    "Removes duplicate elements from the given list."
    seen = set()
    result = []
    for item in array:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result

def print_exit(message):
    "Print a message and exit."
    print(message, file=sys.stderr)
    exit()
