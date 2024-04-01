import os
import re
import glob
from config import DIVIDER
from utils import loadText, textLines
from rag_summariser import STRUCTURED_SUB_ROOT

MODEL = "llama2"
summary_root = os.path.join(STRUCTURED_SUB_ROOT, MODEL)
N = 50

RE_DIVIDER = re.compile(r"^\s*([A-Z\s]+):.*%s" % DIVIDER)
RE_RESULT = re.compile(r"^\s*(\d+)\.")
RE_SIZE = re.compile(r"comments_size:\s*(\d+),", re.MULTILINE)

def resultLines(lines):
    matches = []
    last_n = -1
    for line in lines:
        m = RE_RESULT.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if n < last_n + 1:
            break
        matches.append(line)
        last_n = n
    return matches

def findSize(text):
    if "[No comments for ticket]" in text:
        return 0
    m = RE_SIZE.search(text)
    assert m, f"Failed to match\n\t{text}"
    return int(m.group(1))

def evaluate(text):
    lines = textLines(text)
    size = findSize(text[:500])
    dividers = [idx for idx, line in enumerate(lines) if DIVIDER in line]
    dividers.append(len(lines))
    groups = {}
    for j, idx in enumerate(dividers[:-1]):
        line = lines[idx]
        m = RE_DIVIDER.match(line)
        assert m, f"Failed to match\n\t{line} with\n\t{RE_DIVIDER.pattern}"
        key = m.group(1)
        i0, i1 = idx+1, dividers[j+1]
        matches = resultLines(lines[i0:i1])

        group = (key, idx+1, dividers[j+1])
        groups[key] = matches

    return size, groups

print(f"Summaries for {summary_root}")
assert os.path.exists(summary_root), f"Missing {summary_root}"
summary_paths = glob.glob(os.path.join(summary_root, "*.txt"))
print(f"Found {len(summary_paths)} summaries.")

score_list = []
for summary_path in summary_paths:
    text = loadText(summary_path)
    if "Jane Doe" in text or "John Doe" in text:
        continue
    lines = textLines(text)
    size, groups = evaluate(text)
    if not all([k in groups for k in ["PROBLEMS", "PARTICIPANTS", "EVENTS"]]):
        continue
    score = len(groups["PROBLEMS"]), len(groups["PARTICIPANTS"]), len(groups["EVENTS"])
    score_list.append((size, score, summary_path))

def groupText(group):
    size, score, summary_path = group
    problems, participants, events = score
    total = problems + participants + events
    summary_path = os.path.abspath(summary_path)
    return f"{size:6}  ({problems:2}, {participants:2}, {events:3}) {total:3}  {summary_path}"

def showScore(score_list, n=5):
    for group in score_list[:n]:
        print(groupText(group))

def ticketNumber(group):
    size, score, summary_path = group
    name = os.path.basename(summary_path)
    return int(name.split(".")[0])

top_tickets = {}

score_list.sort(key=lambda x: x[1][0] + x[1][1] + x[1][2], reverse=True)
print("Top 5 summaries: Total score, Problems, Participants, Events")
showScore(score_list)
for group in score_list[:N]:
    top_tickets[ticketNumber(group)] = group

score_list.sort(key=lambda x: 3*x[1][0] + 2*x[1][1] + x[1][2], reverse=True)
print("Top 5 summaries: Total score, 3xProblems, 2xParticipants, Events")
showScore(score_list)
for group in score_list[:N]:
    top_tickets[ticketNumber(group)] = group

score_list.sort(key=lambda x: x[1][0], reverse=True)
print("Top 5 summaries: Total score, Problems,")
showScore(score_list)
for group in score_list[:N]:
    top_tickets[ticketNumber(group)] = group

score_list.sort(key=lambda x: x[1][1], reverse=True)
print("Top 5 summaries: Total score, Participants,")
showScore(score_list)
for group in score_list[:N]:
    top_tickets[ticketNumber(group)] = group

score_list.sort(key=lambda x: x[1][2], reverse=True)
print("Top 5 summaries: Total score, Events,")
showScore(score_list)
for group in score_list[:N]:
    top_tickets[ticketNumber(group)] = group

def sortKey(k):
    group = top_tickets[k]
    size, score, summary_path = group
    n = 3*score[0] + 2*score[1] + score[2]
    return -n / (size+5000), -score[0], -score[1], -score[2]

top_list = sorted(top_tickets.keys(), key=sortKey)
print(f"Top summaries: {len(top_list)} of {5*N}")
for i, k in enumerate(top_list[:N]):
    group = top_tickets[k]
    print(f"{i+1:2}: {groupText(group)}")
