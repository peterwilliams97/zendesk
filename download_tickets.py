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

assert USER, "missing ZENDESK_USER"
assert TOKEN, "missing ZENDESK_TOKEN"
assert SUBDOMAIN, "missing ZENDESK_SUBDOMAIN"

ZENDESK_API = f"https://{SUBDOMAIN}.zendesk.com/api/v2/"

def makeUrl(path):
    "Constructs and returns a URL by appending `path` to `ZENDESK_API.`"
    return f"{ZENDESK_API}/{path}"

def urlGet(url):
    """Sends a GET request to `url` with authentication and headers.
        Returns: The JSON response parsed as a dictionary.
        Raises exceptions if the GET request fails or the response is not valid JSON.
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

def fetchTicket(ticketNumber):
    """Fetches Zendesk ticket number `ticketNumber`.
        Returns: A dictionary representing the ticket.
        https://developer.zendesk.com/api-reference/ticketing/tickets/tickets/#show-ticket
    """
    result = zdGet(f"tickets/{ticketNumber}")
    assert "error" not in result, f"^^^ {ticketNumber} {result}"
    return result

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

def saveJson(path, a_dict):
    "Save the given dictionary as JSON to a file at the specified path."
    with open(path, "w") as f:
        json.dump(a_dict, f, indent=4)


DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

TICKET_DIR = "tickets"
os.makedirs(TICKET_DIR, exist_ok=True)

def saveTicket(ticketNumber):
    """Download the comments in Zendesk ticket `ticketNumber` and save them as one text file per
        comment in directory `DATA_DIR/ticketNumber`.
    """
    ticketWrapper = fetchTicket(ticketNumber)
    ticket = ticketWrapper["ticket"]
    print(f"^^^ {ticketNumber} {list(ticket.keys())}")
    ticketPath = os.path.join(TICKET_DIR, f"{ticketNumber}.json")
    saveJson(ticketPath, ticket)

    # fields = [f"{field['id']}:{field['title']}" for field in ticket["fields"]]
    tags = ticket["tags"]
    fields = ticket["fields"]
    fields = [field for field in fields if field["value"]]
    fieldsStr = [f"{field['id']}:{field['value']}" for field in fields]
    print(f" -fields: {fieldsStr}")
    print(f" -tags: {ticket['tags']}")
    return fields, tags

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
    allFields = defaultdict(int)
    allTags = defaultdict(int)
    fieldValues = defaultdict(set)
    for ticketNumber in TICKET_NUMBERS:
        fields, tags = saveTicket(ticketNumber)
        for field in fields:
            allFields[field["id"]] += 1
            fieldValues[field["id"]].add(field["value"])
        for tag in tags:
            allTags[tag] += 1
        # saveComments(ticketNumber)
    print(f"{len(allFields)} fields ==============================================================")
    for field in sorted(allFields.keys(), key=lambda k: allFields[k], reverse=True):
        print(f"{field:15}: {allFields[field]} {sorted(fieldValues[field])}")
    print(f"{len(allTags)} tags ==============================================================")
    for tag in sorted(allTags.keys(), key=lambda k: allTags[k], reverse=True):
        print(f"{tag:15}: {allTags[tag]}")
