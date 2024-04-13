"""
    This module is used to fetch ticket comments from Zendesk.
    https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
"""
import datetime
from config import TICKET_INDEX_PATH, TICKET_ALIASES_PATH
from utils import save_json
from zendesk_wrapper import load_index, update_index

MIN_DATE = None
MAX_DATE = None

# Uncomment these lines to set a date range for the tickets.
# MIN_DATE = datetime.datetime(2024, 1, 1)
# MAX_DATE = datetime.datetime(2024, 1, 5)

def main():
    """ Entry point of the program.
        Downloads all the tickets and comments from Zendesk .
    """
    print("Updating the index...")
    df = load_index()
    print("Updating the index...")
    update_index(df, min_date=MIN_DATE, max_date=MAX_DATE)
    print("Index created!")

if __name__ == "__main__":
    main()
