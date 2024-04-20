"""
    This module is used to fetch ticket comments from Zendesk.
    https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
"""
import datetime
from argparse import ArgumentParser
from config import TICKET_INDEX_PATH, TICKET_ALIASES_PATH
from utils import save_json
from zendesk_wrapper import load_create_index, update_index

MIN_DATE = None
MAX_DATE = None

# Uncomment these lines to set a date range for the tickets.
# MIN_DATE = datetime.datetime(2024, 1, 1)
# MAX_DATE = datetime.datetime(2024, 1, 5)

def main():
    """ Entry point of the program.
        Downloads all the tickets and comments from Zendesk .
    """
    parser = ArgumentParser(description=("Download ticket from Zendesk."))
    parser.add_argument("--no-fetch", action="store_true", help="Don't fetch the tickets.")
    parser.add_argument("--clean", action="store_true", help="Delete current downloaded tickets.")
    args = parser.parse_args()

    print("Updating the index...")
    df = load_create_index(add_custom_fields=True)

    print("Updating the index...")
    update_index(df,
                min_date=MIN_DATE, max_date=MAX_DATE,
                do_fetch=not args.no_fetch,
                clean=args.clean)

    print("Index created!")

if __name__ == "__main__":
    main()
