"""
    This module is used to fetch ticket comments from Zendesk.
    https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
"""
import datetime
from zendesk_wrapper import fetchTicketNumbers, downloadTickets, makeIndex

def main():
    """ Entry point of the program.
        Downloads all the tickets and comments from Zendesk .
    """
    MAX_BATCHES = 0 # No limit
    START_DATE = datetime.datetime(2024, 3, 20)

    ticket_numbers = fetchTicketNumbers(MAX_BATCHES, START_DATE)
    print(f"Found {len(ticket_numbers)} tickets. {ticket_numbers[:3]}...{ticket_numbers[-3:]}")

    downloadTickets(ticket_numbers, overwrite=False)
    print(f"Downloaded {len(ticket_numbers)} tickets")

    makeIndex()
    print("Index created")

if __name__ == "__main__":
    main()
