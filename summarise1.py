import os
import glob
import llama_index
from llama_index.core import ServiceContext
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.response_synthesizers import TreeSummarize

# MODEL = "mistral"
MODEL = "llama2"
# MODEL = "llama2:text"    # Doesn't follow instructions.
# MODEL = "mistral:instruct"
# MODEL = "llama2:13b"    # Crushes my Mac

DATA_DIR = "data"
SUMMARY_ROOT = "summaries"
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

TIMEOUT_SEC = 600

print(f"Loading {MODEL}")
llm = Ollama(model=MODEL, request_timeout=TIMEOUT_SEC)
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")
summarizer = TreeSummarize(service_context=service_context, verbose=False)

SUMMARY_PROMPT = """The following text is a series of messages from a PaperCut support ticket.
Summarise the whole conversation, including a list of participants and who they work for,
the problem or problems, the key events and date they occurred,
and the current status of the ticket. Include any log lines from the messages."""

def summariseTicket(ticketNumber):
    "Summarizes the Zendesk ticket with the given `ticketNumber` and returns the summary text."
    input_files = commentPaths(ticketNumber)
    reader = SimpleDirectoryReader(input_files=input_files)
    docs = reader.load_data()
    texts = [doc.text for doc in docs]
    return summarizer.get_response(SUMMARY_PROMPT, texts)

#
# Test case.
#
if __name__ == "__main__":
    import time
    print(f"MODEL={MODEL}")
    ticketNumbers = sorted(os.path.basename(path) for path in glob.glob(os.path.join(DATA_DIR, "*")))
    ticketNumbers.sort(key=lambda k: (totalSizeKB(commentPaths(k)), k))
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
            print(f"Skipping ticket {ticketNumber}", flush=True)
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
    # for ticketNumber in ticketNumbers:
    #     summary = ticketSummary[ticketNumber]
    #     duration = durations[ticketNumber]
    #     n = numFiles[ticketNumber]
    #     print(f"Summary: ticket {ticketNumber}: {n} comments {duration:5.2f} sec len={len(summary)} ---------------")
    #     print(summary)
    #     print("-" * 80)
    #     print("")
    print(f"Duration: {duration:.2f} seconds")
    for i, ticketNumber in enumerate(ticketNumbers):
        commentCount = commentCounts[ticketNumber]
        commentSize = totalSizeKB(commentPaths(ticketNumber))
        duration = durations[ticketNumber]
        print(f"{i:2}: {ticketNumber:8}: {commentCount:3} comments {commentSize:7.3f} kb {duration:5.2f} seconds")
