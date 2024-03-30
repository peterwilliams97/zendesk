""" Use Llama Index TreeSummarize to summarize the comments in Zendesk support tickets.

    The summaries are saved in a subdirectory of `SUMMARY_ROOT` named after the model used for
    summarization.

"""
import os
import time
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import TreeSummarize
from config import SUMMARY_ROOT, COMPANY

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
        summaryPath(ticket_number): Returns the path to the summary file for a given ticket number.
        summariseTicket(ticket_number, input_files, status): Summarizes the comments for a ticket.
        _subDir(): Returns the subdirectory for storing the summaries.
        _summarise(texts, status): Answers questions based on the given texts and status.
    """

    def __init__(self, llm, model):
        sub_dir = self._subDir()
        self.summary_dir = os.path.join(sub_dir, model).replace(":", "_")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.summariser = TreeSummarize(llm=llm, verbose=False)

    def summaryPath(self, ticket_number):
        "Returns the path to the summary file the ticket with number `ticket_number`."
        return os.path.join(self.summary_dir, f"{ticket_number}.txt")

    def summariseTicket(self, ticket_number, input_files, status):
        """
        Summarizes the comments for a ticket.

        Args:
            ticket_number (str): The ticket number.
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
        print(f"   Loaded {len(texts)} comments in {time.time() - t0:.1f} seconds")
        assert texts, f"No comments for ticket {ticket_number}"
        t0 = time.time()
        full_answer = self._summarise(texts, status)
        print(f"   Summarised {len(texts)} comments in {time.time() - t0:.1f} seconds")
        return full_answer

    def _subDir(self):
        "Returns the subdirectory for storing the summaries."
        assert False, "Subclass must implement _subDir."

    def _summarise(self, texts, status):
        "Returns a summary of the comments in `texts` taking account of the ticket status."
        assert False, "Subclass must implement _summarise."
        return None

PLAIN_SUB_ROOT = os.path.join(SUMMARY_ROOT, "plain")

PLAIN_PROMPT = f"""The following text is a series of messages from a {COMPANY} support ticket.
Summarise the whole conversation, including a list of participants and who they work for,
the problem or problems, the key events and dates they occurred.
and the current status of the ticket."""

def makePlainPrompt(status):
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

    def _subDir(self):
        return PLAIN_SUB_ROOT

    def _summarise(self, texts, status):
        "Returns a summary the comments in `texts` taking account of the ticket status."
        prompt = makePlainPrompt(status)
        t0 = time.time()
        answer = self.summariser.get_response(prompt, texts)
        print(f"Summarised in {time.time() - t0:.1f} seconds")
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

def statusKnown(status):
    "Returns a prompt for the status of a ticket with the known status `status`."
    if status == "open":
#         question = """What is the current unresolved problem(s) in this open ticket?
# Why is this ticket still open? """
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

BASE_PROMPT = f"""The following text is a series of messages from a {COMPANY} support ticket.
Please answer the following questions based on the messages in the ticket.
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
# 'Company' must appear in at least one message.
# If the participant is not mentioned in any messages, don't invent one.
# If the participant is mentioned in a message but their company is not, write `Company` as "Unknown".
# {COMPANY} is not a participant.
# Alice and Bob are not participants
# """
),

# If a participant is a customer, list them first.
# If a participant works for {COMPANY}, list them last.

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

#     ("Logs", """List all the log lines from the messages.
# Use a numbered list.
# Order the list by date, earliest first.
# Don't add a prologue or epilogue to the list.
# When there is no log line, don't write a line.
# Write the full line of text containing the log line
# Log lines are lines that start with a date and a status such as INFO, WARN, DEBUG or ERROR.
# Example: 2022-01-27 13:31:43,628  WARN NetworkAddressResolver - Failed to resolve hostname  to network address: java.net.UnknownHostException: No such host is known  [spring-async-task-1]
# Example: 2023-06-15 14:05:41,513 DEBUG BaseXMLRPCServlet - XMLRPC(providers-xmlrpc) - start - IP: 127.0.0.1, ST: 2, TT: 8, full-ver: 108.14.0.5068-CCA5330 (bundled with 21.2.11.65657) env-ver: 14 (id:Xu1cJx, POST - /rpc/providers/xmlrpc) [http-52]
# Example:  2024/02/18 21:09:02 pc-print-deploy-client-vdi.exe: STDOUT|	TRACE	deploy/deploy.go:156	Unique session per user	{"fqUsername": "northpoint\\sselke2", "session": {"SessionID":2,"SessionName":"rdp-tcp#0","HostName":"","DomainName":"northpoint",:"cany-rog-td026","IPs":["172.23.94.111"],"FarmName":"","IsRemote":true,"Status":1}}
# Example: ERROR  | wrapper  | 2022/01/27 13:30:58 | JVM appears hung: Timed out waiting for signal from JVM.
# """),
]

def makeQuestionPrompt(text):
    "Returns a prompt created by appending `text` to the base prompt `BASE_PROMPT`."
    return f"{BASE_PROMPT}\n{text}"

QUESTIONS = [question for question, _ in QUESTION_DETAIL]
QUESTION_PROMPT = {short: makeQuestionPrompt(detail) for (short, detail) in QUESTION_DETAIL}

def makeAnswer(question, answer, extra):
    "Returns `question` and `answer` formatted into a structured answer string."
    question = f"{question.upper()}:"
    if extra:
        answer = f"{extra}\n{answer}"
    text = f"{question:13} -------------------------------------------------------------*\n{answer}"
    assert "COMPANY" not in answer, f"Answer contains COMPANY: {text}"
    return text

class StructuredSummariser(BaseSummariser):
    "A summariser that uses different prompt to ask for each part of a ticket summary."
    def __init__(self, llm, model):
        super().__init__(llm, model)

    def _subDir(self):
        return STRUCTURED_SUB_ROOT

    def _summarise(self, texts, status):
        "Returns a summary of the comments in `texts` taking account of the ticket status."
        questionAnswer = {}

        t0 = time.time()
        for i, question in enumerate(QUESTIONS):
            print(f"  question {i:2}: {question:12}", end=" ", flush=True)
            t0 = time.time()

            prompt = QUESTION_PROMPT[question]
            if status and question == "Status":
                prompt = makeQuestionPrompt(statusKnown(status))

            answer = self.summariser.get_response(prompt, texts)

            questionAnswer[question] = answer.strip()
            print(f"{time.time() - t0:5.1f} seconds to answer", flush=True)

        answers = []
        for question in QUESTIONS:
            extra = f"Status={status.title()}" if (status and question == "Status") else None
            answer = makeAnswer(question, questionAnswer[question], extra)
            answers.append(answer)

        return "\n\n".join(answers)
