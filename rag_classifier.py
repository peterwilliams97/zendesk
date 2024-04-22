"""
    A summariser that summaries with features that can be used for ticket classification.
    See find_closest_tickets.py for an example of these summaries are used.

    The summariser uses a Pydantic data model to validate the structured summaries.
    It takes a list of ticket comments, a status, and a ticket number, and produces a structured
    summary of the ticket comments.
"""

import os
import sys
import time
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.types import BaseModel, Enum
from typing import List
from config import COMPANY, DIVIDER, CLASSIFICATION_DIR
from utils import since

assert COMPANY, "Set COMPANY in config.py before running this script"
PYDANTIC_SUB_ROOT = CLASSIFICATION_DIR

TICKET_CLASSES = {
    "Accounting": "Accounts, payments, quotes etc",
    "Licencing": "Licencing issues",
    "Technical": "Technical queries from existing customers",
    "Defect": "Bug reports",
    "Pre-sales": "Pre-sales queries",
    "Cancellation": "Cancellation requests",
    "Feature": "Feature requests",
    "Spam": "Spam / Unsolicited enquiries",
}

class ClassEnum(str, Enum):
    Unknown = "Unknown"
    Accounting = "Accounting"
    Licencing = "Licencing"
    Technical = "Technical"
    Defect = "Defect"
    Presales = "Pre-sales"
    Cancellation = "Cancellation"
    Feature = "Feature"
    Spam = "Spam"

DEFECT_TYPES = {
    "Feature": "A request for a new feature or enhancement.",
    "Performance": "A request to improve the performance of the software.",
    "Usability": "A request to improve the usability of the software.",
    "Documentation": "A request to improve the documentation.",
    "Other": "Another type of issue that is not listed above.",
    "Reporting":  "A defect in the reporting functionality.",
    "User Management": "A defect in the user management functionality.",
    "Security": "A defect in the security functionality.",
    "Web interface": "Description of Other Product",
    "Drivers": "Description of Other Product",
    "Find me printing": "Description of Other Product",
    "Print scripting": "Description of Other Product",
    "Release station": "Description of Other Product",
    "Scan to email": "Description of Other Product",
    "Database": "Description of Other Product",
    "Payment gateway": "Description of Other Product",
    "SSO": "Description of Other Product",
    "SSL": "Description of Other Product",
    "Non-admin": "Description of Other Product",
    "Embedded Sharp": "Description of Other Product",
    "Print Deploy Server": "Description of Other Product",
    "Email to Print": "Description of Other Product",
    "Server Performance": "Description of Other Product",
    "Print Deploy Print Queue": "Description of Other Product",
    "Scan OCR": "Description of Other Product",
    "User Group Sync Azure AD": "Description of Other Product",
    "Print Provider": "Description of Other Product",
    "Server Migration": "Description of Other Product",
    "Card Readers": "Description of Other Product",
    "Print Archiving": "Description of Other Product",
    "Can't access PaperCut": "Description of Other Product",
    "Site Server": "Description of Other Product",
    "Embedded Fuji Xerox": "Description of Other Product",
    "Mobility Print Discovery DNS": "Description of Other Product",
    "Print Jobs Disappearing": "Description of Other Product",
    "Embedded Lexmark": "Description of Other Product",
    "Embedded Other": "Description of Other Product",
    "Print Analysis": "Description of Other Product",
    "Mobility Print Cloud": "Description of Other Product",
    "Operating System": "Description of Other Product",
    "Logging": "Description of Other Product",
    "Print Deploy Cloner Tool": "Description of Other Product",
    "User Group Sync Windows AD": "Description of Other Product",
    "Scan to Fax": "Description of Other Product",
    "User Client Deployment": "Description of Other Product",
    "Email to Print Microsoft": "Description of Other Product",
    "Direct Printing": "Description of Other Product",
    "Device Compatibility": "Description of Other Product",
    "Print Deploy Client Deployment": "Description of Other Product",
    "Watermarking": "Description of Other Product",
}

PRODUCTS = {
    "Mobility": "Description of Product 1",
    "Scanning": "Inegrated Scanning",
    "Embedded": "Description of Product 2",
    "Print Deploy": "Description of Product 2",
    "Hive": "Description of Product 3",
    "Pocket": "Description of Other Product",
    "MF": "Description of Other Product",
    "Print": "Description of Other Product",
    "PaperCut": "Description of Other Product",
    "Card Readers": "Description of Other Product",
    "Reporting": "Description of Other Product",
    "User Management": "Description of Other Product",
    "Security": "Description of Other Product",
}

class DefectEnum(str, Enum):
    Unknown = "Unknown"
    Feature = "Feature"
    Performance = "Performance"
    Usability = "Usability"
    Documentation = "Documentation"
    Other = "Other"
    Reporting = "Reporting"
    User_Management = "User Management"
    Security = "Security"
    Web_interface = "Web interface"
    Drivers = "Drivers"
    Find_me_printing = "Find me printing"
    Print_scripting = "Print scripting"
    Release_station = "Release station"
    Scan_to_email = "Scan to email"
    Database = "Database"
    Payment_gateway = "Payment gateway"
    SSO = "SSO"
    SSL = "SSL"
    Non_admin = "Non-admin"
    Embedded_Sharp = "Embedded Sharp"
    Print_Deploy_Server = "Print Deploy Server"
    Email_to_Print = "Email to Print"
    Server_Performance = "Server Performance"
    Print_Deploy_Print_Queue = "Print Deploy Print Queue"
    Scan_OCR = "Scan OCR"
    User_Group_Sync_Azure_AD = "User Group Sync Azure AD"
    Print_Provider = "Print Provider"
    Server_Migration = "Server Migration"
    Card_Readers = "Card Readers"
    Print_Archiving = "Print Archiving"
    Cant_access_PaperCut = "Can't access PaperCut"
    Site_Server = "Site Server"
    Embedded_Fuji_Xerox = "Embedded Fuji Xerox"
    Mobility_Print_Discovery_DNS = "Mobility Print Discovery DNS"
    Print_Jobs_Disappearing = "Print Jobs Disappearing"
    Embedded_Lexmark = "Embedded Lexmark"
    Embedded_Other = "Embedded Other"
    Print_Analysis = "Print Analysis"
    Mobility_Print_Cloud = "Mobility Print Cloud"
    Operating_System = "Operating System"
    Logging = "Logging"
    Print_Deploy_Cloner_Tool = "Print Deploy Cloner Tool"
    User_Group_Sync_Windows_AD = "User Group Sync Windows AD"
    Scan_to_Fax = "Scan to Fax"
    User_Client_Deployment = "User Client Deployment"
    Email_to_Print_Microsoft = "Email to Print Microsoft"
    Direct_Printing = "Direct Printing"
    Device_Compatibility = "Device Compatibility"
    Print_Deploy_Client_Deployment = "Print Deploy Client Deployment"
    Watermarking = "Watermarking"

class TicketTraitsModel(BaseModel):
    """Pydantic data model for a Zendesk ticket summary.
    """
    Summary: str
    Product: str
    Features: str
    Class: ClassEnum
    Defect: DefectEnum
    Description: str
    Characteristics: str
    Problems: List[str]

FORMAT_CLASSES = "\n".join([f"Class={k}  DESCRIPTION='{v}'" for k, v in TICKET_CLASSES.items()])
FORMAT_DEFECTS = "\n".join([f"DEFECT={k}" for k in DEFECT_TYPES.keys()])

# The prompt for telling the LLM to summarise a ticket and output a JSON response.

PYDANTIC_TRAITS_PROMPT = f"""The following text is a series of messages from a {COMPANY} support ticket.
Please answer the following questions based on the messages in the ticket.
Do not invent any information that is not in the messages.
Format your answers as JSON. Remember to escape special characters such as quotes.
Do write anything other than the JSON response.

The summary should be a single sentence that captures the main issue in the ticket.

The ticket class is one of the following:

{FORMAT_CLASSES}

Give the ticket class name or write Unknown you cannot determine the class. Your answer must be a
a ticket class or "Unknown".

The ticket defect type is one of the following:

{FORMAT_DEFECTS}

If the ticket is not a defect, write Unknown. Your answer must be a
a ticket defect type or "Unknown".

List all the product problems raised in the ticket.
Problems are issues that need to be resolved, such as a bug, a feature request.
Each problem should be a single sentence describing the problem.

Give a full description of the ticket. The full description should describe the context of
the ticket, and summarise your response. The full description should be a few sentences long. It
will be used to classify the ticket so the more information the better.

List the {COMPANY} products mentioned in the ticket. If the ticket does not mention any products, write "None".

List the {COMPANY} products features addressed in the ticket. If the ticket does not mention any features, write "None".

Describe the distinguishing characteristics of {COMPANY} product and features addressed in this ticket.
What makes the customer problems like those in other tickets and what makes them different? Focus on the ticket class,
product features, and problems raised in the ticket. Do not mention the customer's name, support
names or dates as these are not relevant to the ticket classification. Do not describe customer
interactions or the resolution of the ticket. Do not describe the customer behaviour. Do not mention
the Class or Defect or any other ticket metadata.
This should be a few sentences long.

Example response:
{{
    "Summary": "Short summary.",
    "Product": "Product namess",
    "Features": "Product features.",
    "Class": "The ticket CLASS name",
    "Defect": "The ticket DEFECT name",
    "Description": "Full description.",
    "Characteristics": "Distinguishing characteristics.",
    "Problems: [
        "Problem 1.",
        "Problem 2."
    ],
}}
"""

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

class PydanticFeatureGenerator():
    """
      A summariser that produces and validates structured summaries using a Pydantic data model.
      These summaries are intended to be used as features for a classifier.

    Args:
        llm (str): The language model to use.
        model (str): The model to use for summarization.

    Attributes:
        summary_dir (str): The directory to store the summaries.
        summariser (TreeSummarize): The summarization model.
        summariser_raw (TreeSummarize): The summarization model without validation.

    Methods:
        summary_path(ticket_number): Returns the path to the summary file for a given ticket number.
        summarise_ticket(ticket_number, input_files, status): Summarizes the comments for a ticket.
    """

    def __init__(self, llm, model,  verbose=False):
        self.summary_dir = os.path.join(PYDANTIC_SUB_ROOT, model).replace(":", "_")
        os.makedirs(self.summary_dir, exist_ok=True)
        self.summariser = TreeSummarize(llm=llm, output_cls=TicketTraitsModel, verbose=verbose)
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
        prompt = PYDANTIC_TRAITS_PROMPT
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
