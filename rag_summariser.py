"""
    A summariser that produces validated structured summaries of Zendesk tickets.

    The summariser uses a Pydantic data model to validate the structured summaries.
    It takes a list of ticket comments, a status, and a ticket number, and produces a structured
    summary of the ticket comments.

    The structured summary includes a summary, status, problems, participants, and events.

    The following is a summary of a Zendesk ticket for COMPANY = "Ben's Dog Walking Service".

SUMMARY: -------------------------------------------------------------*
Pekingese "Tricki Woo" was not being walked in Ben's dog walking service.

STATUS: -------------------------------------------------------------*
Current status: Resolved. Issue with Tricki Woo's leash was identified and replaced, resulting in successful dog walking.

PROBLEMS: -------------------------------------------------------------*
1. The dog walker is unable to walk Tricki Woo due to a broken leash.
2. Ben did not stock spare leashes for such emergencies.

PARTICIPANTS: -------------------------------------------------------------*
1. Mrs Pumphrey: Tricki Woo's owner.
2. William Hodgekin: Ben's Dog Walking Service.

EVENTS: -------------------------------------------------------------*
1. 2021-06-28: Client reported issues with dog not being walked.
2. 2021-06-29: Ben's Dog Walking Service diagnosed the issue as a problem with the dog leash and provided instructions for replacing it.
3. 2021-06-30: Client confirmed successful dog walking after replacing the leash.

"""

import os
import sys
import time
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.types import BaseModel
from typing import List
from config import SUMMARY_ROOT, COMPANY, DIVIDER
from utils import since

assert COMPANY, "Set COMPANY in config.py before running this script"
PYDANTIC_SUB_ROOT = os.path.join(SUMMARY_ROOT, "pydantic")

class TicketSummaryModel(BaseModel):
    """Pydantic data model for a Zendesk ticket summary.
    """
    Summary: str
    Status: str
    Problems: List[str]
    Participants: List[str]
    Events: List[str]

# Markers for parts of PYDANTIC_PROMPT that we will replace with the status of specific tickets.
STATUS_KEY = "[STATUS]"
STATUS_QUESTION = "[STATUS_QUESTION]"

# The prompt for telling the LLM to summarise a ticket and output a JSON response.

PYDANTIC_PROMPT = f"""The following text is a series of messages from a {COMPANY} support ticket.
Please answer the following questions based on the messages in the ticket.
Do not invent any information that is not in the messages.
Format your answers as JSON. Remember to escape special characters such as quotes.

The summary should be a single sentence that captures the main issue in the ticket.

The status is {STATUS_KEY}. {STATUS_QUESTION}

List all the problems raised in the ticket.
Problems are issues that need to be resolved, such as a bug, a feature request.
Each problem should be a single sentence describing the problem.

List all the participants in the ticket.
Give the name and organisation of each participant or 'Unknown' if the organisation is not mentioned.

List all the events and the date they occurred.
List only the key events, such as problems being reported, solutions being proposed, and resolutions
being reached.

Example response:
{{
    "Summary": "Short summary.",
    "Status": "Current status. Explanation",
    "Problems: [
        "Problem 1.",
        "Problem 2."
    ],
    "Participants": [
        "Name 1: Organisation 1",
        "Name 2: Organisation 2"
    ],
    "Events": [
        "2014-07-21: Event 1",
        "2019-11-03: Event 2"
    ]
}}
"""

def status_prompt(status):
    "Returns a prompt to explain the status of a ticket with the given status."
    status_lwr = status.lower()
    if status_lwr in {"open", "new"}:
         question = "an explanation of why this ticket is still open"
    elif status_lwr in {"closed", "solved", "resolved"}:
        question = "an explanation of how customer's problem was solved"
    elif status_lwr in {"pending", "hold"}:
        question = "the work the needs to be done to address the customer's problem."
    else:
        assert False, f"Unknown status: {status}"

    return (f"Include {question} in the Status section. If you can't, say nothing. " +
             "Status should be succinct and one line.")

def summarisation_prompt(status):
    "Returns a prompt to requests a full summary of a ticket with the given status."
    status_question = status_prompt(status)
    prompt = PYDANTIC_PROMPT.replace(STATUS_KEY, status.title())
    prompt = PYDANTIC_PROMPT.replace(STATUS_QUESTION, status_question)
    return prompt

def pydantic_response_text(response, status):
    "Converts JSON `response` to formatted text and adds the known `status` to the Status section."
    sections = []
    for key, value in response.dict().items():
        lines = [f"{key.upper()}: {DIVIDER}"]
        if key == "Status":
            lines.append(f"Current status: {status}. {value}")
        elif isinstance(value, str):
            lines.append(value)
        elif isinstance(value, list):
            lines.append("\n".join([f"{i+1}. {item}" for i, item in enumerate(value)]))
        else:
            assert False, f"Unexpected type {type(value)} for {key}"
        sections.append("\n".join(lines))
    return "\n\n".join(sections)

class PydanticSummariser():
    """
    A summariser that produces and validates structured summaries using a Pydantic data model.

    Args:
        llm (str): The language model to use.
        model (str): The model to use for summarization.

    Attributes:
        summary_dir (str): The directory to store the summaries.
        summariser (TreeSummarize): The summarization model.
        summariser_raw (TreeSummarize): The summarization model without the Pydantic data model.

    Methods:
        summary_path(ticket_number): Returns the path to the summary file for a given ticket number.
        summarise_ticket(ticket_number, input_files, status): Summarizes the comments for a ticket.
    """

    def __init__(self, llm, model, verbose=False):
        self.summary_dir = os.path.join(PYDANTIC_SUB_ROOT, model).replace(":", "_")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.summariser = TreeSummarize(llm=llm, output_cls=TicketSummaryModel, verbose=verbose)
        self.summariser_raw = TreeSummarize(llm=llm, verbose=verbose)

    def summary_path(self, ticket_number):
        "Returns the path to the summary file the ticket with number `ticket_number`."
        return os.path.join(self.summary_dir, f"{ticket_number}.txt")

    def summarise_ticket(self, ticket_number, input_files, status):
        """
        Summarizes the comments for a ticket.

        Args:
            ticket_number (int): The ticket number.
            input_files (list): The list of input files.
            status (str): The status of the ticket.

        Returns:
            str: The full answer summarizing the comments.
        """
        print("  summariseTicket: -----------------------")

        t0 = time.time()
        reader = SimpleDirectoryReader(input_files=input_files)
        docs = reader.load_data()

        texts = [doc.text for doc in docs]
        print(f"   Loaded {len(texts)} comments in {since(t0):.1f} seconds")
        assert texts, f"No comments for ticket {ticket_number}"

        t0 = time.time()
        ticket_summary = self._summarise_validate(ticket_number, texts, status)
        print(f"   Summarised {len(texts)} comments in {since(t0):.1f} seconds")

        return ticket_summary

    def _summarise_validate(self, ticket_number, texts, status):
        """
        Summarises the given texts and validates the JSON response with Pydantic.

        Args:
            ticket_number (int): The ticket number associated with the texts.
            texts (list[str]): The list of texts to be summarised.
            status (str): The status of the ticket.

        Returns:
            str: The summarised response text if successful, None otherwise.
        """
        prompt = summarisation_prompt(status)
        try:
            response = self.summariser.get_response(prompt, texts)
        except Exception as e:
            response_raw = self.summariser_raw.get_response(prompt, texts)
            print(f"  Pydantic ValidationError: ticket {ticket_number}\n" +
                  f"  RAW RESPONSE   {'-' * 80}\n" +
                  f"  {response_raw}\n" +
                  f"  PYDANTIC ERROR {'-' * 80}\n" +
                  f"  {e}",
                  file=sys.stderr)
            return None
        return pydantic_response_text(response, status)
