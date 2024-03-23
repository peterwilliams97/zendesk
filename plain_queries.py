""" A module that generates summaries for Zendesk tickets by answering SUMMARY_PROMPT.
"""
import os
import time
import llama_index
from llama_index.core.settings import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import TreeSummarize
from config import COMPANY, SUMMARY_ROOT
from zendesk_utils import commentPaths

SUMMARY_SUB_ROOT = os.path.join(SUMMARY_ROOT, "plain")

summariser = TreeSummarize(verbose=False)

assert COMPANY, "Set COMPANY in config.py before running this script"
SUMMARY_PROMPT = """The following text is a series of messages from a {COMPANY} support ticket.
Summarise the whole conversation, including a list of participants and who they work for,
the problem or problems, the key events and date they occurred,
and the current status of the ticket. Include any log lines from the messages."""

class PlainSummariser:
    """
    A class that generates plain summaries for Zendesk tickets by answering SUMMARY_PROMPT.

    Attributes:
        summary_dir: The directory where the summaries will be stored.

    Methods:
        __init__(self, llm, model): Initializes the StructuredSummariser object.
        summaryPath(self, ticketNumber): Returns the file path for where we store the summary of a
                Zendesk ticket.
        summariseTicket(self, ticketNumber): Summarizes a Zendesk ticket by generating answers to
            predefined questions.
    """

    def __init__(self, llm, model):
        """
        Initializes this summariser.

        llm: The LLM used for summarisation.
        model: The LLM model name.
        """
        self.summary_dir = os.path.join(SUMMARY_SUB_ROOT, model).replace(":", "_")
        os.makedirs(self.summary_dir, exist_ok=True)
        Settings.embed_model = "local"
        Settings.llm = llm

    def summaryPath(self, ticketNumber):
        "Returns the file path for where we store the summary of Zendesk ticket `ticketNumber`."
        assert self.summary_dir
        return os.path.join(self.summary_dir, f"{ticketNumber}.txt")

    def summariseTicket(self, ticketNumber):
        """
        Summarizes the ticket `ticketNumber` using `SUMMARY_PROMPT`.

        Returns: Text containing the answers to each of the questions in the prompy based on the
                comments in the ticket.
        """
        t0 = time.time()
        input_files = commentPaths(ticketNumber)
        reader = SimpleDirectoryReader(input_files=input_files)
        docs = reader.load_data()
        texts = [doc.text for doc in docs]
        print(f"Loaded {len(texts)} comments in {time.time() - t0:.1f} seconds")
        return summariser.get_response(SUMMARY_PROMPT, texts)
