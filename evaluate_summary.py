
import os
import time
import datetime
from collections import defaultdict
from config import METADATA_KEYS, DIVIDER
from utils import list_index, load_text, text_lines, RE_DATE, RE_TIME, RE_YEAR
from zendesk_wrapper import comment_paths
from logs_parser import standard_date, extract_dates, extract_log_entries

DAYS_BEFORE = 50 # Number of days before the ticket creation date to consider.
DAYS_AFTER = 10  # Number of days after the ticket update date to consider.

def get_bad_dates(text, valid_strings, min_date, max_date):
    """
    Extracts date strings from `text` and returns a list of date strings
    that are not present in the `valid_strings` list.

    Args:
        text (str): The text to extract date strings from.
        valid_strings (list): A list of valid date strings.
        min_date (datetime.date): The minimum allowed date.
        max_date (datetime.date): The maximum allowed date.

    Returns:
        tuple: A tuple containing two lists:
            - date_strings: A list of unique date strings extracted from the text.
            - bad_dates: A list of date strings that are not present in the valid_strings list.
    """
    date_strings = extract_dates(text)
    date_list = [standard_date(date_str, min_date, max_date) for date_str in date_strings]
    date_strings = sorted({date.strftime("%Y-%m-%d") for date in date_list if date})

    bad_dates = []
    for date_str in date_strings:
        if date_str in valid_strings:
            continue
        if any(date_str in valid_str for valid_str in valid_strings):
            continue
        bad_dates.append(date_str)

    return date_strings, bad_dates

def extract_ticket_logs(ticket_number):
    """
    Extracts ticket logs for a given ticket number.

    Args:
        ticket_number (str): The ticket number to extract logs for.

    Returns:
        list: A list of tuples containing the path of the log file and the extracted log entries.
    """
    paths = comment_paths(ticket_number)
    path_logs = []
    for path in paths:
        text = load_text(path)
        line_matches = extract_log_entries(text)
        if line_matches:
            path_logs.append((path, line_matches))
    return path_logs

def extract_metadata_info(ticket_number, metadata):
    """Reads the metadata for the ticket `ticket_number` and returns
        the metadata as a string, the status, the minimum date, and the maximum date tp consider.
    """
    status = metadata.get("status")
    min_date = metadata["created_at"] - datetime.timedelta(days=DAYS_BEFORE)
    max_date = metadata["updated_at"] + datetime.timedelta(days=DAYS_AFTER)

    metadata_keys = sorted(metadata.keys(), key=lambda k: (list_index(METADATA_KEYS, k), k))
    metadata_parts = [f"{k}: {metadata[k]}" for k in metadata_keys
        if k != "description" and metadata[k] and metadata[k] != "Unknown"]
    metadata_str = f"metadata: [{', '.join(metadata_parts)}]"

    return metadata_str, status, min_date, max_date

def parse_dates_logs(ticket_number, min_date, max_date):
    """
    Parses the logs of a given ticket number and extracts valid dates and log lines.

    Args:
        ticket_number (int): The ticket number to extract logs from.
        min_date (str): The minimum date to consider as valid.
        max_date (str): The maximum date to consider as valid.

    Returns:
        A tuple containing:
            - valid_dates (set): A set of valid dates in the format 'YYYY-MM-DD'.
            - date_summary (str): A summary of the valid dates in the format 'DATE: "date1", "date2", ...'.
            - log_summary (str): A summary of the log lines.
    """
    valid_dates = set()
    date_lines = []
    log_lines = []
    date_dict = defaultdict(list)

    path_logs = extract_ticket_logs(ticket_number)
    if path_logs:
        for m, (path, line_matches) in enumerate(path_logs):
            for line_match in line_matches:
                (i, date_str, time_str, matches, line) = line_match
                if date_str and not matches:
                    date = standard_date(date_str, min_date, max_date)
                    if not date:
                        continue
                    std = date.strftime("%Y-%m-%d")
                    valid_dates.add(std)
                    assert isinstance(std, str), f"std={std}"
                    date_dict[std].append(date_str)

                if matches:
                    log_lines.append(line)
        for std in sorted(date_dict.keys()):
            strings = sorted(set(date_dict[std]))
            strings = [f'"{s}"' for s in strings]
            date_lines.append(f"{std}: {', '.join(strings)}")
            # date_lines = sorted(set(date_lines))

    date_title = f"DATES: {DIVIDER} {len(date_lines)} dates "
    date_summary = "\n".join([date_title] + date_lines)
    log_title = f"LOGS: {DIVIDER} {len(log_lines)} logs "
    log_summary = "\n".join([log_title] + log_lines)

    return valid_dates, date_summary, log_summary

def summarise_ticket(summariser, ticket_number, metadata):
    """
    Summarizes the ticket `ticket_number` by generating answers to a set of predefined questions.

    Returns: A dict of text containing the answers to each of the questions based on the
            comments in the ticket, and an error string.
    """
    input_files = comment_paths(ticket_number)
    if not input_files:
        return None, "[No comments for ticket]"
    status = metadata["status"]
    if not status:
        return None, "[Unknown status]"

    full_answer = summariser.summarise_ticket(ticket_number, input_files, status)
    if not full_answer:
        return None, "[Response not generated]"

    metadata_str, status, min_date, max_date = extract_metadata_info(ticket_number, metadata)
    valid_dates, date_summary, log_summary = parse_dates_logs(ticket_number, min_date, max_date)
    found_dates, bad_dates = get_bad_dates(full_answer, valid_dates, min_date, max_date)

    hallucinated = None
    if bad_dates:
        hallucinated = sorted(bad_dates)
        hallucinated = [f'"{s}"' for s in hallucinated]
        hallucinated = ", ".join(hallucinated)
        hallucinated = "\n".join([
            f"HALLUCINATED DATES: {DIVIDER}",
            f"{len(bad_dates)} hallucinated of {len(found_dates)} dates. {len(valid_dates)} regex dates.",
            hallucinated,
        ])

    return {
        "metadata": metadata_str,
        "full_answer": full_answer,
        "log_summary": log_summary,
        "hallucinated": hallucinated,
        "date_summary": date_summary,
    }, None

SUMMARY_KEYS = ["metadata", "full_answer", "log_summary", "hallucinated", "date_summary"]

def summary_text(summary_dict):
    "Generate a summary text by concatenating the values from `summary_dict`."
    parts = [summary_dict[k] for k in SUMMARY_KEYS if summary_dict.get(k)]
    return "\n\n".join(parts)
