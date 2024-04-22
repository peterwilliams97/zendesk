"""
    This module provides utility functions for working with Zendesk tickets.
"""
import os
import json
import re
import sys
import time
import datetime
from config import DIVIDER

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
    "Returns `pattern` compiled to a regular expression. The regular expression is case-insensitive."
    return re.compile(pattern, re.IGNORECASE)

# Copied from https://github.com/madisonmay/CommonRegex
# License: MIT
PATTERN_DATE = '(?:(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?\s+(?:of\s+)?(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)|(?:jan\.?|january|feb\.?|february|mar\.?|march|apr\.?|april|may|jun\.?|june|jul\.?|july|aug\.?|august|sep\.?|september|oct\.?|october|nov\.?|november|dec\.?|december)\s+(?<!\:)(?<!\:\d)[0-3]?\d(?:st|nd|rd|th)?)(?:\,)?\s*(?:\d{4})?|[0-3]?\d[-\./][0-3]?\d[-\./]\d{2,4}'
PATTERN_TIME = '\d{1,2}:\d{2} ?(?:[ap]\.?m\.?)?|\d[ap]\.?m\.?'

RE_DATE = regex_compile(PATTERN_DATE)
RE_TIME = regex_compile(PATTERN_TIME)
RE_YEAR = regex_compile(r"\s+(\d{4})\b")

def text_lines(text):
    "Returns a list of the non-empty lines in `text`."
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if len(line) > 0]
    return lines

def truncate(text, max_len):
    "Returns `text` truncated to `max_len` characters."
    text = text.strip()
    if max_len < 0 or len(text) < max_len:
        return text
    n = int(round(0.7 * max_len))
    text0 = text[:n]
    m = max_len - n
    text1 = text[-m:]
    return f"{text0} ... {text1}"

def since(t0):
    "Returns the time elapsed since `t0` in seconds."
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

def match_key(a_dict, key):
    """ Find a key in dictionary `a_dict` that starts with the given key (case-insensitive).
        Returns the matching key if found, None otherwise.
    """
    matches = [k for k in a_dict if k.startswith(key.lower())]
    if not matches:
        print(f"{key}' doesn't match any of {list(a_dict.keys())}", file=sys.stderr)
        return None
    if len(matches) > 1:
        print(f"{key}' matches {matches}. Choose one.", file=sys.stderr)
        return None
    return matches[0]

def round_score(score, num_places=2):
    "Rounds `score` to `num_places` decimal places."
    n = 10**num_places
    return int(round(n * score)) / n

def directory_ticket_numbers(directory):
    "Returns the ticket numbers in the given directory."
    return [int(name.split(".")[0]) for name in os.listdir(directory) if name.endswith(".txt")]


class SummaryReader:
    """
    A class for reading
SECTION_NAMES = ["SUMMARY", "STATUS", "PROBLEMS", "PARTICIPANTS", "EVENTS", "LOGS", "DATES"]

"""
    def __init__(self, section_names):
        self.section_names = section_names

    def summary_to_sections(self, text):
        """
        Convert a summary text into sections based on predefined section names.

        Args:
            text (str): The summary text to be converted into sections.

        Returns:
            dict: A dictionary where the keys are the section names and the values are the
                  corresponding sections.

        """
        lines = text_lines(text)
        section = []
        name = None
        name_section = {}
        for i, line in enumerate(lines):
            if DIVIDER in line:
                if section:
                    name_section[name] = section
                section = []
                name = None
                for g in self.section_names:
                    if g in line:
                        name = g
                        break
                # missing = [g for g in self.section_names if g not in name_section.keys()]
                # assert name, f"missing {missing} of {self.section_names}\n{text}"
            elif name:
                line = line.strip()
                if line:
                    section.append(line)
            missing = [g for g in self.section_names if g not in name_section.keys()]
            if not missing:
                break
        # assert not missing, f"missing {missing} of {self.section_names}\n{text}"
        return {name: "\n".join(section) for name, section in name_section.items()}

    # def summary_to_content(self, text):
    #     "Convert a text summary into content format."
    #     sections =  self._summary_to_sections(text)
    #     summary = sections.get("SUMMARY", "not specified")
    #     problems = sections.get("PROBLEMS", "")
    #     return f"SUMMARY: {summary}\n\nPROBLEMS:\n {problems}"
