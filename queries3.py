"""
This script is used to summarize conversations from Zendesk support tickets.
It reads text files containing comments from the ticket and generates a summary
that includes information about the participants, problems raised, key events,
current status of the ticket, and log lines from the messages.

The script uses the `Gemini` model from the `llama_index` package to generate the summary.
The summary is saved in a text file for each ticket.

Usage:
- Modify the `MODEL` variable to specify the desired model for summarization.
- Set the `DATA_DIR` variable to the directory containing the ticket data.
- Run the script to generate summaries for the tickets.

Note: This script requires the `llama_index` package to be installed.
"""
import os
import glob
import llama_index
from llama_index.core import ServiceContext
from llama_index.llms.gemini import Gemini
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import TreeSummarize
from llama_index.core.evaluation import FaithfulnessEvaluator

MODEL = "Gemini"

DATA_DIR = "data"
SUMMARY_ROOT = "structured.summaries"
SUMMARY_DIR = os.path.join(SUMMARY_ROOT, MODEL).replace(":", "_")

os.makedirs(SUMMARY_DIR, exist_ok=True)

def saveText(path, text):
    "Save the given text to a file at the specified path."
    with open(path, "w") as f:
        f.write(text)

def commentPaths(ticketNumber):
    "Returns a sorted list of file paths for the comments in Zendesk ticket `ticketNumber`."
    ticketDir = os.path.join(DATA_DIR, ticketNumber)
    return sorted(glob.glob(os.path.join(ticketDir, "*.txt")))

def summaryPath(ticketNumber):
    "Returns the file path for where we store the summary of Zendesk ticket `ticketNumber`."
    return os.path.join(SUMMARY_DIR, f"{ticketNumber}.txt")

def totalSizeKB(paths):
    "Returns the total size in kilobytes of the files specified by `paths`."
    return sum(os.path.getsize(path) for path in paths) / 1024

def currentTime():
    "Returns the current time in the format 'dd/mm/YYYY HH:MM:SS'."
    from datetime import datetime
    now = datetime.now()
    return now.strftime("%d/%m/%Y %H:%M:%S")

llm = Gemini()
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
summarizer = TreeSummarize(service_context=service_context, verbose=False)

evaluator = FaithfulnessEvaluator(llm=llm)

COMPANY = "PaperCut"
BASE_PROMPT = f"The following text is a series of messages from a {COMPANY} support ticket."

def makePrompt(text):
    return f"{BASE_PROMPT}\n{text}"

QUESTION_DETAIL = [
    ("Summary", "Summarise the whole conversation in one sentence."),

    ("Problems", """List the problems raised in the ticket.
Use a numbered list.
Don't add a prologue or epilogue to the list.
Problems are issues that need to be resolved, such as a bug, a feature request.
Questions about how to use the product are not problems.
Responses to problems are not problems.
Each problem should be a single sentence describing the problem.
If there are no problems, write 'None'.
If there are multiple problems, order them by importance.  """),

  ("Status", """What is the current status of the ticket?
Is it open, closed, or pending?
If it is closed, what was the resolution?
If it is pending, what is the next action?
If it is open, what is the current problem?
Do not include any other information in this answer.
Your answer should be one sentence for status and optionally one sentence for the resolution or next action.
"""),

    ("Participants", """List the participants and who they work for.
Use a numbered list.
Don't add a prologue or epilogue to the list.
Use the format: 'Name: Company.'
List the customer first and {COMPANY} staff last.
"""),

    ("Events", """List the key events and the date they occurred.
Use a numbered list.
Don't add a prologue or epilogue to the list.
If there are no events, write 'None'.
Use the format: 'Date: Event.'
Format the date as 'YYYY-MM-DD'.
Order the list by date, earliest first."""),

    ("Logs", """List all the log lines from the messages.
Use a numbered list.
Order the list by date, earliest first.
Don't add a prologue or epilogue to the list.
If there are no log lines, write 'None'.
Log lines are lines that start with a date and a status such as INFO, WARN, DEBUG or ERROR.
Example: 2022-01-27 13:31:43,628  WARN
Example: 2022-01-26 12:40:18,380 DEBUG ClientManagerImpl
Example: ERROR  | wrapper  | 2022/01/27 13:30:58 | JVM exited unexpectedly. """),
]

QUESTIONS = [question for question, _ in QUESTION_DETAIL]
QUESTION_PROMPT = {short: makePrompt(detail) for (short, detail) in QUESTION_DETAIL}

def makeAnswer(question, answer):
    question = f"{question.upper()}:"
    return f"{question:13} -------------------------------------------------------------*\n{answer}"

def summariseTicket(ticketNumber):
    """Summarizes the ticket `ticketNumber` by generating answers to a set of predefined questions.
         Returns: Structured text containing the answers to each of the questions based on the
         comments in the ticket.
    """
    t0 = time.time()
    input_files = commentPaths(ticketNumber)
    reader = SimpleDirectoryReader(input_files=input_files)
    docs = reader.load_data()
    texts = [doc.text for doc in docs]
    print(f"Loaded {len(texts)} comments in {time.time() - t0:.2f} seconds")
    questionAnswer = {}
    for question in reversed(QUESTIONS):
        t0 = time.time()
        prompt = QUESTION_PROMPT[question]
        answer = summarizer.get_response(prompt, texts)
        questionAnswer[question] = answer.strip()
        print(f"{time.time() - t0:5.2f} seconds to answer {question}")

    return "\n\n".join(makeAnswer(question, questionAnswer[question]) for question in QUESTIONS)

#
# Test case.
#
# ['1259693', '1216136', '1196141', '1260221', '1116722', '1280919']
#    0: 1259693    7 comments   2.888 kb
#    1: 1216136   26 comments  20.715 kb
#    2: 1196141  122 comments  81.527 kb
#    3: 1260221  106 comments 126.619 kb
#    4: 1116722  288 comments 190.168 kb
#    5: 1280919  216 comments 731.220 kb

MAX_SIZE = 100  # Maximum size of ticket comments in kilobytes.

if __name__ == "__main__":
    import time
    print(f"MODEL={MODEL}")
    ticketNumbers = sorted(os.path.basename(path) for path in glob.glob(os.path.join(DATA_DIR, "*")))
    ticketNumbers.sort(key=lambda k: (totalSizeKB(commentPaths(k)), k))
    # ticketNumbers = ticketNumbers[:2]
    ticketNumbers = [k for k in ticketNumbers if totalSizeKB(commentPaths(k)) < MAX_SIZE]
    print(ticketNumbers)
    for i, ticketNumber in enumerate(ticketNumbers):
        paths = commentPaths(ticketNumber)
        print(f"{i:4}: {ticketNumber:8} {len(paths):3} comments {totalSizeKB(paths):7.3f} kb")
    # ticketNumbers = ticketNumbers[:1]

    t00 = time.time()
    summaries = {}
    durations = {}
    commentCounts = {}
    commentSizes = {}
    for i, ticketNumber in enumerate(ticketNumbers):
        commentCount = len(commentPaths(ticketNumber))
        commentSize = totalSizeKB(commentPaths(ticketNumber))

        print(f"{i:2}: ticketNumber={ticketNumber:8} {commentCount:3} comments {commentSize:7.3f} kb {currentTime()}",
            flush=True)

        if os.path.exists(summaryPath(ticketNumber)):
            print(f"       skipping ticket {ticketNumber}", flush=True)
            continue   # Skip tickets that have already been summarised.

        t0 = time.time()
        summary = summariseTicket(ticketNumber)
        duration = time.time() - t0
        description = f"{commentCount} comments {commentSize:7.3f} kb {duration:5.2f} sec summary={len(summary)}"

        print(f"  {description}", flush=True)

        with open(summaryPath(ticketNumber), "w") as f:
            print(f"Summary: ticket {ticketNumber}: {description} -------------------------", file=f)
            print(summary, file=f)

        summaries[ticketNumber] = summary
        durations[ticketNumber] = duration
        commentCounts[ticketNumber] = commentCount
        commentSizes[ticketNumber] = commentSize

    duration = time.time() - t00
    print("====================^^^====================")
    print(f"Duration: {duration:.2f} seconds")
    for i, ticketNumber in enumerate(ticketNumbers):
        if os.path.exists(summaryPath(ticketNumber)):
            continue
        commentCount = commentCounts[ticketNumber]
        commentSize = totalSizeKB(commentPaths(ticketNumber))
        duration = durations[ticketNumber]
        print(f"{i:2}: {ticketNumber:8}: {commentCount:3} comments {commentSize:7.3f} kb {duration:5.2f} seconds")
