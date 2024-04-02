"""
    This module is used to fetch ticket comments from Zendesk.
    https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
"""
import datetime
from config import TICKET_INDEX_PATH, TICKET_ALIASES_PATH
from utils import saveJson
from zendesk_wrapper import loadIndex, updateIndex

# MIN_DATE = datetime.datetime(2024, 1, 1)
# MAX_DATE = datetime.datetime(2024, 1, 31)
MIN_DATE = None
MAX_DATE = None

def main():
    """ Entry point of the program.
        Downloads all the tickets and comments from Zendesk .
    """
    df = loadIndex(TICKET_INDEX_PATH)
    print(f"Loaded {len(df)} tickets from {TICKET_INDEX_PATH}")
    print("Updating the index...")

    df, reversed_aliases = updateIndex(df, min_date=MIN_DATE, max_date=MAX_DATE)

    print(f"Writing {len(df)} tickets to {TICKET_INDEX_PATH}")
    df.to_csv(TICKET_INDEX_PATH)
    print(f"Writing {len(reversed_aliases)} aliases to {TICKET_ALIASES_PATH}")
    saveJson(TICKET_ALIASES_PATH, reversed_aliases)

    print("Index created!")

if __name__ == "__main__":
    main()
