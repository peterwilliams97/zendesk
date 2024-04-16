""" Use Llama Index TreeSummarize to summarize the comments in Zendesk support tickets.

    The summaries are saved in a subdirectory of `SUMMARY_ROOT` named after the model used for
    summarization.
"""
import os
import sys
import time
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import TreeSummarize
from config import SUMMARY_ROOT, COMPANY, DIVIDER
from utils import since

assert COMPANY, "Set COMPANY in config.py before running this script"

class BaseSummariser:
    """
    Base class for summarizers.

    Args:
        llm (str): The language model to use.
        model (str): The model to use for summarization.

    Attributes:
        summary_dir (str): The directory to store the summaries.
        summariser (TreeSummarize): The summarization model.

    Methods:
        summary_path(ticket_number): Returns the path to the summary file for a given ticket number.
        summarise_ticket(ticket_number, input_files, status): Summarizes the comments for a ticket.
        _sub_dir(): Returns the subdirectory for storing the summaries.
        _summarise(texts, status): Answers questions based on the given texts and status.
    """

    def __init__(self, llm, model, output_cls=None, verbose=False):
        sub_dir = self._sub_dir()
        self.summary_dir = os.path.join(sub_dir, model).replace(":", "_")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.summariser = TreeSummarize(llm=llm, output_cls=output_cls, verbose=verbose)

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
        full_answer = self._summarise(ticket_number, texts, status)
        print(f"   Summarised {len(texts)} comments in {since(t0):.1f} seconds")
        return full_answer

    def _sub_dir(self):
        "Returns the subdirectory for storing the summaries."
        assert False, "Subclass must implement _sub_dir."

    def _summarise(self, ticket_number, texts, status):
        "Returns a summary of the comments in `texts` taking account of the ticket status."
        assert False, "Subclass must implement _summarise."
        return None

PLAIN_SUB_ROOT = os.path.join(SUMMARY_ROOT, "plain")

PLAIN_PROMPT = f"""The following text is a series of messages from a {COMPANY} support ticket.
Summarise the whole conversation, including a list of participants and who they work for,
the problem or problems, the key events and dates they occurred.
and the current status of the ticket."""

def make_plain_prompt(status):
    "Returns a prompt for the status of a ticket with the known status `status`."
    if status == "open":
        question = "Why is this ticket still open?"
    elif status in {"closed", "solved", "resolved"}:
        question = "What was the resolution to the customer's problem?"
    elif status in {"pending", "hold"}:
        question = "What work needs to be done to address the customer's problem?"
    else:
        question = "Summarise the current status of the ticket"
    return f"{PLAIN_PROMPT}\n{question}"

class PlainSummariser(BaseSummariser):
    "A summariser that uses a single prompt to ask for all parts of a ticket summary at once."

    def __init__(self, llm, model):
        super().__init__(llm, model)

    def _sub_dir(self):
        return PLAIN_SUB_ROOT

    def _summarise(self, ticket_number, texts, status):
        "Returns a summary the comments in `texts` taking account of the ticket status."
        prompt = make_plain_prompt(status)
        t0 = time.time()
        answer = self.summariser.get_response(prompt, texts)
        print(f"Summarised in {since(t0):.1f} seconds")
        return answer

STRUCTURED_SUB_ROOT = os.path.join(SUMMARY_ROOT, "structured")

STATUS_UNKNOWN = """What is the current status of the ticket?
Is it open, closed, or pending?
If it is closed, what was the resolution?
If it is pending, what is the next action?
If it is open, what is the current problem?
"""

STATUS_EPILOG = """Do not include any other information in this answer.
Your answer should be one sentence for status, one sentence for the resolution or next action, and
one sentence to explain how you determined the status.
"""

def status_known(status):
    "Returns a prompt for the status of a ticket with the known status `status`."
    if status == "open":
         question = """Why is this ticket still open?"""
    elif status in {"closed", "solved", "resolved"}:
        question = "What was the resolution to the customer's problem?"
    elif status in {"pending", "hold"}:
        question = "What work needs to be done to address the customer's problem?"
    else:
        assert False, f"Unknown status: {status}"

    return f"""The current status of the ticket is {status}.
{question}
Please answer succinctly.
Do not include any other information in this answer.
Do not include the status in the answer.
Do not list all the problems raised in the ticket.
"""

BASE_PROMPT = f"""Please answer the following questions based on the messages in the ticket.
Do not invent any information that is not in the messages."""

# Do not include text from this prompt in your response."""

QUESTION_DETAIL = [
    ("Summary",
    """Give a brief overview of this support ticket.
The summary should be a single sentence that captures the main issue in the ticket.
Avoid minor details."""
),

    ("Problems",
    f"""List all the problems raised in the ticket.
Include all the main issues.
Use a numbered list.
Problems are issues that need to be resolved, such as a bug, a feature request.
Each problem should be a single sentence describing the problem."""

# Don't add a prologue or epilogue to the list.
# Problems are issues that need to be resolved, such as a bug, a feature request.
# Questions about how to use the product are not problems.
# Responses to problems are not problems.
# Each problem should be a single sentence describing the problem.
# Do not repeat the same problem.
# When there is no problem, don't write a line."""

# If there are multiple problems, order them by importance, most important first."""
),

  ("Status", STATUS_UNKNOWN + STATUS_EPILOG),

  ("Participants",
    f"""List the the names of the participants (people) in the messsages and who
they work for.
Use a numbered list.
Don't add a prologue or epilogue to the list.
Use the format: 'Name: Organization.'
'Name' is the name of the participant and 'Organization' is the organization they work for.
'Name' should be the name of the participant as it appears in the messages.
'Name' must appear in at least one message.
"""
),

    ("Events",
    """List the key events and the date they occurred.
An event is something that happens, such as a problem being reported, a solution being proposed, or
a resolution being reached.
Don't include contacts, responses, or other non-events.
Use a numbered list.
Don't add a prologue or epilogue to the list.
Questions about how to use the product are not events.
Responses to problems are not events.
Log lines are not events.
When there is no event, don't write a line.
Use the format: 'Date: Event.' where 'Date' is the date the event occurred and 'Event' is a brief
description of the event.
Format the date as 'YYYY-MM-DD' where 'YYYY' is the year, 'MM' is the month, and 'DD' is the day.
Order the list by date, earliest first."""

# Use the format: 'Date: Event. (Quote)' where 'Date' is the date the event occurred and 'Event' is a brief
# description of the event. 'Quote' is the literal text in the input document containing the date
# the event occurred.
),

]

def make_question_prompt(text):
    "Returns a prompt created by appending `text` to the base prompt `BASE_PROMPT`."
    return f"{BASE_PROMPT}\n{text}"

QUESTIONS = [question for question, _ in QUESTION_DETAIL]
QUESTION_PROMPT = {short: make_question_prompt(detail) for (short, detail) in QUESTION_DETAIL}

def make_structured_answer(question, answer, extra):
    "Returns `question` and `answer` formatted into a structured answer string."
    question = f"{question.upper()}:"
    if extra:
        answer = f"{extra}\n{answer}"
    text = f"{question:13} {DIVIDER}\n{answer}"
    assert "COMPANY" not in answer, f"Answer contains COMPANY: {text}"
    return text

class StructuredSummariser(BaseSummariser):
    "A summariser that uses different prompts to ask for each part of a ticket summary."
    def __init__(self, llm, model, verbose=False):
        super().__init__(llm, model, verbose=verbose)

    def _sub_dir(self):
        return STRUCTURED_SUB_ROOT

    def _summarise(self, ticket_number, texts, status):
        "Returns a summary of the comments in `texts` taking account of the ticket status."
        questionAnswer = {}

        t0 = time.time()
        for i, question in enumerate(QUESTIONS):
            print(f"  question {i:2}: {question:12}", end=" ", flush=True)
            t0 = time.time()

            prompt = QUESTION_PROMPT[question]
            if status and question == "Status":
                prompt = make_question_prompt(status_known(status))

            answer = self.summariser.get_response(prompt, texts)

            questionAnswer[question] = answer.strip()
            print(f"{since(t0):5.1f} seconds to answer", flush=True)

        answers = []
        for question in QUESTIONS:
            extra = f"Status={status.title()}" if (status and question == "Status") else None
            answer = make_structured_answer(question, questionAnswer[question], extra)
            answers.append(answer)

        return "\n\n".join(answers)

COMPOSITE_SUB_ROOT = os.path.join(SUMMARY_ROOT, "composite")

def make_query(question, body):
    "Returns `question` and `answer` formatted into a structured answer string."
    title = f"{question.title()}:"
    return f"{title}: {body} <<<"

COMPOSITE_PROMPT = f"""{BASE_PROMPT}
For each of the following ***TITLE >>> QUESTION <<< questions, answer in the following format:
TITLE: ========================
Answer to the question.
--------------
TITLE is not one of the titles in the list of questions.
"""

class CompositeSummariser(BaseSummariser):
    "A summariser that uses a single composite built from `QUESTION_DETAIL`."
    def __init__(self, llm, model, verbose=False):
        super().__init__(llm, model, verbose=verbose)

    def _sub_dir(self):
        return COMPOSITE_SUB_ROOT

    def _summarise(self, ticket_number, texts, status):
        "Returns a summary of the comments in `texts` taking account of the ticket status."

        prompt_parts = [COMPOSITE_PROMPT]
        for question in QUESTIONS:
            if status and question == "Status":
                body = status_known(status)
            else:
                body = QUESTION_PROMPT[question]
            prompt = make_query(question, body)
            prompt_parts.append(prompt)

        prompt = "\n".join(prompt_parts)
        t0 = time.time()
        answer = self.summariser.get_response(prompt, texts)
        print(f"Summarised in {since(t0):.1f} seconds")
        return answer

PYDANTIC_SUB_ROOT = os.path.join(SUMMARY_ROOT, "pydantic")

from llama_index.core.types import BaseModel
from typing import List

class TicketSummaryModel(BaseModel):
    """Pydantic data model for a Zendesk ticket summary.
    """
    Summary: str
    Status: str
    Problems: List[str]
    # Participants: List[str]
    # Events: List[str]

STATUS_KEY = "[STATUS]"
STATUS_QUESTION = "[STATUS_QUESTION]"
PYDANTIC_PROMPT = f"""The following text is a series of messages from a {COMPANY} support ticket.
Please answer the following questions based on the messages in the ticket.
Do not invent any information that is not in the messages.
Format your answers as JSON. Remember to escape special characters such as quotes.

The summary should be a single sentence that captures the main issue in the ticket.

The status is {STATUS_KEY}. {STATUS_QUESTION}

List all the problems raised in the ticket.
Problems are issues that need to be resolved, such as a bug, a feature request.
Each problem should be a single sentence describing the problem.

Example response:
{{
    "Summary": "Short summary.",
    "Status": "Current status. Explanation",
    "Problems: [
        "Problem 1.",
        "Problem 2."
    ]
}}
"""

def status_prompt(status):
    "Returns a prompt for the status of a ticket with the known status."
    status_lwr = status.lower()
    if status_lwr in {"open", "new"}:
         question = "an explanation of why this ticket is still open"
    elif status_lwr in {"closed", "solved", "resolved"}:
        question = "an explanation of how customer's problem was solved"
    elif status_lwr in {"pending", "hold"}:
        question = "the work the needs to be done to address the customer's problem."
    else:
        assert False, f"Unknown status: {status}"

    return f"Include {question} in the Status section. If you can't, say nothing. Status should be succint and one line."

def pydantic_prompt(status):
    "Returns a prompt for the status of a ticket with the known status."
    status_question = status_prompt(status)
    prompt = PYDANTIC_PROMPT.replace(STATUS_KEY, status.title())
    prompt = PYDANTIC_PROMPT.replace(STATUS_QUESTION, status_question)
    return prompt

def pydantic_response_text(response, status):
    "Converts JSON `response` to formatted text and adds the known `status` to the Status section"
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

class PydanticSummariser(BaseSummariser):
    "A summariser that produces and validates structured summaries using a Pydantic data model."
    def __init__(self, llm, model, verbose=False):
        super().__init__(llm, model, output_cls=TicketSummaryModel, verbose=verbose)

    def _sub_dir(self):
        return PYDANTIC_SUB_ROOT

    def _summarise(self, ticket_number, texts, status):
        prompt = pydantic_prompt(status)
        try:
            response = self.summariser.get_response(prompt, texts)
        except Exception as e:
            print(f"  Pydantic ValidationError: ticket {ticket_number} {e}", file=sys.stderr)
            return None
        return pydantic_response_text(response, status)

# Dictionary of summariser types.
SUMMARISER_TYPES = {
    "plain": PlainSummariser,
    "structured": StructuredSummariser,
    "composite": CompositeSummariser,
    "pydantic": PydanticSummariser,
}

SUMMARISER_DEFAULT = "pydantic"
