"""
    This module contains configuration settings for the Zendesk ticket summarisation.
    You will need to repalce the values in this file with the values for your own Zendesk tickets.
"""
import os

FILE_ROOT = os.path.expanduser("~/zendesk_data")

# The directory containing the downloaded ticket data.
DATA_ROOT = os.path.join(FILE_ROOT, "data")

# The directory containing the downloaded ticket data.
TICKET_BATCHES_DIR = os.path.join(DATA_ROOT, "ticket_batches")

# The directory containing the downloaded ticket comments data.
COMMENTS_DIR = os.path.join(DATA_ROOT, "comments")

TICKET_INDEX_PATH = os.path.join(DATA_ROOT, "ticket_index.csv")
TICKET_ALIASES_PATH = os.path.join(DATA_ROOT, "ticket_aliases.json")
TAGS_JSON_PATH = os.path.join(DATA_ROOT, "tags.json")
TAGS_CSV_PATH = os.path.join(DATA_ROOT, "tags.csv")

# The directory where the summaries will be stored.
SUMMARY_ROOT = os.path.join(FILE_ROOT, "summaries")

CLASSIFICATION_DIR = os.path.join(FILE_ROOT, "class_traits")

# The directory where the reranker models are stored.
MODEL_ROOT = os.path.join(FILE_ROOT, "reranker.models")

# The directory where the reranker results are stored.
SIMILARITIES_ROOT = os.path.join(FILE_ROOT, "similarities")

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
    "priority",
    "problem_type",
    "product_name", "product_version",
    "recipient", "customer", "region",
    "subject",
    # "description",
]
CUSTOM_FIELDS_KEY = "custom_fields"

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

# The random seed used for the LLM models.
RANDOM_SEED = 19170829
