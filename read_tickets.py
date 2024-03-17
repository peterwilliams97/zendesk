"""
    This module is used to fetch ticket comments from Zendesk.
    https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
"""
import requests
import json
import os
from collections import defaultdict

USER = os.environ.get("ZENDESK_USER")
TOKEN = os.environ.get("ZENDESK_TOKEN")
SUBDOMAIN = os.environ.get("ZENDESK_SUBDOMAIN")

ZENDESK_API = f"https://{SUBDOMAIN}.zendesk.com/api/v2/"

def makeUrl(path):
    "Constructs and returns a URL by appending `path` to `ZENDESK_API.`"
    return f"{ZENDESK_API}/{path}"

def urlGet(url):
    """Sends a GET request to `url` with authentication and headers.
        Returns: The JSON response parsed as a dictionary.
        Raises exceptions ff the GET request fails or the response is not valid JSON.
    """
    auth = (f"{USER}/token", TOKEN)
    headers = {"Content-Type": "application/json"}
    response = requests.request("GET", url, auth=auth, headers=headers)
    return json.loads(response.text)

def zdGet(path):
    "Calls the Zendesk API with the specified path and returns the response parsed as a dictionary."

    url = makeUrl(path)
    return urlGet(url)

def fetchTicketFields():
    return zdGet("ticket_fields")

def fetchAuthor(author_id):
    return zdGet(f"users/{author_id}")

def fetchTicketComments(ticketNumber):
    """Fetches the comments for Zendesk ticket number `ticketNumber`.
        Returns: A list of comments for the specified ticket number.
    """
    result = zdGet(f"tickets/{ticketNumber}/comments?include=&include_inline_images=false")
    assert "error" not in result, f"^^^ {ticketNumber} {result}"

    comments = []
    for _ in range(100):
        comments.extend(result["comments"])
        url = result.get("next_page")
        if not url:
            break
        result = urlGet(url)

    return comments

def saveText(path, text):
    "Save the given text to a file at the specified path."
    with open(path, "w") as f:
        f.write(text)

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

def saveComments(ticketNumber):
    """Download the comments in Zendesk ticket `ticketNumber` and save them as one text file per
        comment in directory `DATA_DIR/ticketNumber`.
    """
    comments = fetchTicketComments(ticketNumber)
    print(f"^^^ {ticketNumber} {len(comments)}")
    ticketDir = os.path.join(DATA_DIR, ticketNumber)
    os.makedirs(ticketDir, exist_ok=True)
    for i, comment in enumerate(comments):
        body = comment["body"]
        path = os.path.join(ticketDir, f"comment_{i:03d}.txt")
        saveText(path, body)


#
# Test case.
#
if __name__ == "__main__":
    TICKET_NUMBERS = ["1259693", "1260221", "1280919", "1196141", "1116722", "1216136"]
    assert len(TICKET_NUMBERS) == len(set(TICKET_NUMBERS)), "duplicate ticket numbers "
    for ticketNumber in TICKET_NUMBERS:
        saveComments(ticketNumber)
