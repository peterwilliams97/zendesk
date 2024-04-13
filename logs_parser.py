""" This module contains functions for parsing log files.
"""
import re
import sys
from dateutil.parser import parse, DEFAULTPARSER
from utils import (text_lines, regex_compile, disjunction,
                   PATTERN_TIME, PATTERN_DATE, RE_TIME, RE_DATE, RE_YEAR)

# The levels of log messages.
LEVELS = [
    "DEBUG", "INFO", "WARN", "ERROR", "FATAL",
    "TRACE", "CRITICAL", "WARNING", "SEVERE",
    "STATUS", "NOTICE", "ALERT", "EMERGENCY",
     "Information", "Warning", "Error"
]
PATTERN_LEVELS = disjunction(LEVELS)

# Patterns for log entries.
LOG_PATTERNS = [
    # r'%s[\:\-\s]{1,4}%s.*%s' % (PATTERN_DATE, PATTERN_TIME, PATTERN_LEVELS),
    r'%s(?:-|:|\s{1,4})%s.*%s' % (PATTERN_DATE, PATTERN_TIME, PATTERN_LEVELS),

    # r'%s:%s.*%s' % (PATTERN_DATE, PATTERN_TIME, PATTERN_LEVELS),
    r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}.*?%s' % PATTERN_LEVELS,

    # 2023-07-20 15:30:45.123456-07:00 12345 (Information) The description for Event ID 100 from source Application cannot be found. Either the component that raises this event is not installed on your local computer or the installation is corrupted. You can install or repair the component on the local computer.
    r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{6}-\d{2}:\d{2} \d+ \(%s\)' % PATTERN_LEVELS,

    #  INFO 2024-03-12-09:04:47 --- PaperCut Mobility Print
    # DEBUG 2024-03-11-15:18:36 Client info o
    r'%s\s+\d{4}-\d{2}-\d{2}[\-\:\s\|]+\d{2}:\d{2}:\d{2}' % PATTERN_LEVELS,

    # "2024-03-05T08:56:04.867583002Z  INFO Performing task
    r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.*%s' % PATTERN_LEVELS,

    # INFO | jvm 1 | 2024/02/06 08:58:35 | at org.snmp4j.asn1.BER.decodeInteger(BER.java:691)')
    # STATUS | wrapper | 2024/03/13 14:21:10
    r'%s\s+\|.+\|\s+\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s*\|' % PATTERN_LEVELS,

    # May 19 17:05:02 INFO : Monitoring 1 printers.')
    r'\w+\s+\d{2}\s+\d{2}:\d{2}:\d{2}\s+%s\s*\:\s+' % PATTERN_LEVELS,

    # 2024/02/23 12:26:54 mobility-print.exe: STDOUT|ERROR loadOsPrintersWrapper:
    # 2023/08/14 08:43:09 pc-print-deploy-client-vdi.exe: STDOUT|    TRACE
    r'\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+.+:\s*[A-Z]+\s*\|\s*(?:DEV|ERR|INFO|DEBUG|TRACE)',

    # 2024/02/12 09:22:17 print-deploy-client: STDOUT|Extract filters
    r'\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}:\d{2}\s+.+:\s*(?:STDOUT|STDERR)\s*\|',
]

RE_INVIDUALS = [regex_compile(pattern) for pattern in LOG_PATTERNS]
pattern_log = disjunction(LOG_PATTERNS)

# C-VGD5X2
RE_CRN = regex_compile(r"([A-Z]{2,3}-[A-Z0-9]{6})")

def standard_date(date_str, min_date, max_date):
    """
    Parses a date string and returns a standardized date within the specified range.

    date_str (str): The date string to be parsed.
    min_date (datetime): The minimum allowed date.
    max_date (datetime): The maximum allowed date.

    Returns:
        datetime: The standardized date within the specified range, or None if the date string
                  cannot be parsed.
    """
    delta = max_date - min_date
    mid_date = min_date + delta / 2
    DEFAULTPARSER._year = mid_date.year

    try:
        date = DEFAULTPARSER.parse(date_str)
    except Exception as e:
        return None

    date = date.replace(tzinfo=None)
    min_date = min_date.replace(tzinfo=None)
    max_date = max_date.replace(tzinfo=None)

    try:
        # Try to fix common 02-03-04 date format confusion.
        if not (min_date <= date <= max_date):
            day, year = date.day, date.year
            ylo = year % 100
            if year // 100 == 20:
                try:
                    new_date = date.replace(year=day + 2000, day=ylo)
                    if min_date <= new_date <= max_date:
                        date = new_date
                except Exception as e:
                    pass
    except TypeError as e:
        print(f"min_date: {type(min_date)} {min_date}", file=sys.stderr)
        print(f"max_date: {type(max_date)} {max_date}", file=sys.stderr)
        print(f"    date: {type(date)} {date}", file=sys.stderr)
        raise

    if not (min_date <= date <= max_date):
        return  None
    return date

def extract_full_dates(line, safe=False):
    "Extracts full dates from a given line of text."
    date_strings = []
    for m in RE_DATE.finditer(line):
        suffix = line[m.end():]
        y = RE_YEAR.search(suffix)
        if not y:
            date_str = m.group(0)
        else:
            date_str = line[m.start():m.end() + y.end()]
            date_str = date_str.strip()
        date_strings.append(date_str)
    return date_strings

def extract_log_entries(text):
    """
    Extracts log entries from `text`.

    Returns: A list of tuples representing the extracted log entries. Each tuple contains the following elements:
        - The line number of the log entry in the original text.
        - The date string extracted from the log entry.
        - The time string extracted from the log entry.
        - A list of indices indicating the matched regular expressions in the log entry.
        - The original log entry line.
    """
    lines = text_lines(text)

    line_matches = []
    for i, line in enumerate(lines):
        date_str = None
        date_strings = extract_full_dates(line)
        if date_strings:
            date_str = date_strings[0]
        m = RE_DATE.search(line)
        time_str = None
        m = RE_TIME.search(line)
        if m:
            time_str = m.group(0)

        matches = []
        if date_str and len(line) >= 30 and any(l in line for l in LEVELS):
            for j, regex in enumerate(RE_INVIDUALS):
                if regex.search(line):
                    matches.append(j)
        if date_str or time_str or matches:
            line_match = (i, date_str, time_str, matches, line)
            line_matches.append(line_match)

    return line_matches

def extract_dates(text):
    "Returns date strings extracted from `text` by extractLogEntries."
    line_matches = extract_log_entries(text)
    lines = [(date_str, line) for i, date_str, time_str, matches, line in line_matches if date_str]
    date_strings = []
    for date_str, line in lines:
        m = RE_DATE.search(line)
        suffix = line[m.end():]
        y = RE_YEAR.search(suffix)
        if y:
            date_str = line[m.start():m.end() + y.end()]
            date_str = date_str.strip()

        date_strings.append(date_str)
    return date_strings
