"""
    This module contains configuration settings for the Zendesk ticket summarisation.
    You will need to repalce the values in this file with the values for your own Zendesk tickets.
"""
import os

# The directory containing the downloaded ticket data.
DATA_ROOT = "data"

# The directory containing the downloaded ticket comments data.
COMMENTS_DIR = os.path.join(DATA_ROOT, "comments")

# The directory containing the downloaded ticket non-comment data.
TICKET_DIR = os.path.join(DATA_ROOT, "tickets")

# The directory containing the metadata extracted from downloaded tickets.
METADATA_DIR = os.path.join(DATA_ROOT, "metadata")

TICKET_INDEX = os.path.join(DATA_ROOT, "ticket_index.csv")
TICKET_ALIASES = os.path.join(DATA_ROOT, "ticket_aliases.json")

# The directory where the summaries will be stored.
SUMMARY_ROOT = "summaries"

# The name of the company that the tickets are for.
# TODO: Update this with your company names.
COMPANY = "PaperCut"

# The PaperCut tickets we are currently working with.
# TODO: Update this list with the your own ticket numbers.
TEST_TICKET_NUMBERS = [1259693, 1260221, 1280919, 1196141, 1116722, 1216136]

#
# The following are the keys that are used to extract metadata from the Zendesk tickets.
#

# Standard Zendesk metadata keys.
METADATA_KEYS = [
    "created_at", "updated_at", "status",
    "product_name", "product_version",
    "priority",
    "problem_type",
    "recipient", "customer", "region",
    "subject",
    "description",
]

# Mapping of Zendesk field IDs to field names.
# This will look something like:
# FIELD_KEY_NAMES = {
#     "12345": "product name",
#     "23456": "product version",
#     "34567": "region",
#     "22333": "problem type",
#     "89912": "customer",
# }
# TODO: Add the field IDs and names from your own Zendesk tickets.
FIELD_KEY_NAMES = {}

DIVIDER = "-------------------------------------------------------------*"
