"""

NOT PRODUCTION CODE.

This is an exercise in learning LlamaIndex's internals by comparing my naive sentence splitter to
theirs.

This script that compares my sentence splitter, split_to_sentences() below to  llamaindex's
SentenceSplitter

It takes a text and splits it into sentences based on the specified chunk size and overlap.
It then compares the output of my sentence splitter to llamaindex's SentenceSplitter.
The script takes command-line arguments to specify the chunk size, overlap, and the number of
characters to display for each chunk.
"""
import re
from collections import namedtuple
import spacy
from llama_index.core.utils import get_tokenizer
from llama_index.core.text_splitter import SentenceSplitter
from utils import disjunction


# Span is a tuple of (start, end) indices of a text.
Span = namedtuple("Span", ["start", "end"])
# Split is a namedtuple of (span, level, token_size) where:
#   span: Span of the split
#   level: The level of the split (0=paragraph, 1=sentence, 2=word)
#   token_size: The number of tokens in the split
Split = namedtuple("Split", ["span", "level", "token_size"])

# Regular expressions for preprocessing text.
CHUNKING_REGEX = re.compile(r"(?:[^;。？！]+[;。？！]?|\.\s+)")
NUMBERED_REGEX = re.compile(r"^\s*[\d]+[\.:\s]")
LINE_REGEX = re.compile("[\-]{5,}")
BLOCKQUOTE_REGEX = re.compile("^\s*(?:>|##)")
URL_REGEX = re.compile(r"(?:https?|cifs):\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,}")
EMAIL_REGEX = re.compile("""(?:[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])""")
GREETING = r"(Hi|Hello|Dear|Hey|Good morning|Good afternoon|Good evening|Greetings|Salutations|Howdy|Yo|Hiya|Hi there|Hello there|Hi)"
GREETING_REGEX = re.compile(rf"^\s*{GREETING}?\s*[\w\s]+\s*(?:[,;:]|$)")
THANKS_REGEX = re.compile("(Thank you|Thanks|Cheers|Regards|Best regards|Best|Sincerely|Yours sincerely|Yours faithfully|Yours truly|Yours|Take care|Kind regards|Warm regards|Warmly|With gratitude|With thanks|With appreciation|With best wishes|With all best wishes|With all good wishes|Obrigado|Gracias|Merci|Danke|Dank|Takk|Tack|Kiitos|Bedankt|Dziekuje|Grazie)[\.!]?", re.IGNORECASE)
PRONOUNS = r"\([A-Za-z\s\\/]+\)"
SUPPORTER_REGEX = re.compile(fr"\w+(?:\s+\w+)?\s*(?:{PRONOUNS})?\s*\nPaperCut\s*\n\[URL\]")
EMOJIS = [";-)", ":-)"]
EMOJI_REGEX = re.compile(disjunction(re.escape(e) for e in EMOJIS))
PARA_REGEX = re.compile(r"\n{2,}") # Matches 2 or more newlines
WORD_REGEX = re.compile(r"\s+")

# Load the spacy model
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")

def preprocess_text(text):
    """
    Preprocesses `text` in preparation for ticket classification.

    Try to remove common patterns that are not useful for sentence splitting that contain
    .;,: etc that interfere with sentence splitting.
    Also replace URLs, emails, greetings, support staff names, and thanks with placeholders.

    Returns: The preprocessed text.
    """
    text = text.replace(u"\u200b", "")
    # text = text.replace(";-)", "pcEMOJI")
    if True:
        text = URL_REGEX.sub("pcURL", text)
        text = EMAIL_REGEX.sub("pcEMAIL", text)
        assert "http://www." not in text
        text = GREETING_REGEX.sub("pcGREETING", text)
        text = SUPPORTER_REGEX.sub("pcSUPPORTER", text)
        text = THANKS_REGEX.sub("pcTHANKS", text)
        text = EMOJI_REGEX.sub("pcEMOJI", text)
    if True:
        blockquote_depth = 0
        lines = text.split("\n")
        for i in range(10):
            matches = []
            for line in lines:
                m = BLOCKQUOTE_REGEX.search(line)
                if not m:
                    break
                matches.append(m)
            if len(matches) < len(lines):
                break

            new_lines = [line[m.end():] for line, m in zip(lines, matches)]
            assert i < 5, f"Too many blockquotes: para={text} lines={lines}->{new_lines} matches={[m.end() for m in matches]}"
            lines = new_lines
            blockquote_depth += 1
        if blockquote_depth:
            prefix = "pcBQ" * blockquote_depth
            lines = [f"{prefix} {line}" for line in lines]
            text = "\n".join(lines)

    if True:
        lines = text.split("\n")
        modifed_lines = [line for line in lines]
        for i, line in enumerate(lines):
            line = LINE_REGEX.sub("pcLINE", line)
            m = NUMBERED_REGEX.match(line)
            if m:
                line = "pcLINENUMBER " + line[m.end():].strip() # Remove the number
            modifed_lines[i] = line
        text = "\n".join(modifed_lines)

    text = PARA_REGEX.sub("\n\n", text)

    return text


def compute_spans(text, regex):
    "Compute the spans of all `regex` maches in `text`. Returns a list of (start, end) tuples."
    matches = [m for m in regex.finditer(text)]
    if len(matches) <= 1:
        return [Span]
    spans = []
    for i, m in enumerate(matches):
        start = 0 if i == 0 else matches[i-1].end()
        end = len(text) if i == len(matches) -1 else m.end()
        spans.append(Span(start, end))
    assert spans[-1].end == len(text), (spans[-1], len(text))
    return spans

def compute_sentence_spans(nlp, text):
    "Compute the sentence spans in `text` using `nlp`, a SpaCy NLP model."
    doc = nlp(text)
    sents = list(doc.sents)
    if len(sents) <= 1:
        return [Span(0, len(text))]
    spans = []
    for i, sent in enumerate(sents):
        start = 0 if i == 0 else sents[i-1].end_char
        end = len(text) if i == len(sents) -1 else sent.end_char
        spans.append(Span(start, end))
    return spans

def format_chunk(text, n=0):
    "Formats `text` by truncating it to length `n` and adding ellipsis in the middle if necessary."
    text = repr(text)
    while text.startswith("'"):
        text = text[1:]
    while text.endswith("'"):
        text = text[:-1]
    if n <= 0 or len(text) <= n:
        return text
    start = text[:n//2]
    m = n - len(start)
    end = text[-m:]
    return f"{start} ... {end}"

li_tokenizer = get_tokenizer()
def get_token_size(text: str) -> int:
    "Returns the nimber of tokens in `text`."
    return len(li_tokenizer(text))

SPLIT_PARA, SPLIT_SENT, SPLIT_WORD = 0, 1, 2

def split_to_sentences(text, chunk_size, chunk_overlap, verbose=False):
    """
    Splits the given `text` into sentences based on the specified `chunk_size` and `chunk_overlap`.

    Args:
        text (str): The text to be split into sentences.
        chunk_size (int): The maximum size of each sentence chunk.
        chunk_overlap (int): The overlap between adjacent sentence chunks.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        List[Split]: A list of Split namedtuples, each containing the span, level, and token size of a sentence chunk.
    """
    prn = print if verbose else lambda *args, **kwargs: None

    text_token_size = get_token_size(text)
    prn(f"***split_to_sentences: text={len(text)} token_size={text_token_size} " +
            f" chunk_size={chunk_size} chunk_overlap={chunk_overlap}")

    def token_count(span):
        return get_token_size(text[span.start:span.end])

    def split_paras(span):
        return compute_spans(text, PARA_REGEX)
    def split_sents(span):
        spans = compute_sentence_spans(nlp, text[span.start:span.end])
        return [Span(span.start + s.start, span.start + s.end) for s in spans]
    def split_words(span):
        parent_count = token_count(span)
        spans = compute_spans(text[span.start:span.end], WORD_REGEX)
        counts = [token_count(Span(span.start + s.start, span.start + s.end)) for s in spans]
        baby_count = sum(counts)
        cumulative = 0
        if parent_count != baby_count:
            prn(f"### text=>>>{text[span.start:span.end]}<<<")
            for i, s in enumerate(spans):
                rejoined = "".join(text[span.start + x.start:span.start + x.end] for x in spans[:i+1])
                text_size = get_token_size(rejoined)
                cumulative += counts[i]
                prn(f"### {i:3}: {counts[i]:3} " +
                     f" rejoined={text_size:3} cumulative={cumulative:3} {cumulative-text_size}  " +
                     f"'{format_chunk(text[span.start + s.start:span.start + s.end])}'")
        rejoined = "".join(text[span.start + s.start:span.start + s.end] for s in spans)
        assert rejoined == text[span.start:span.end], (rejoined, text[span.start:span.end])
        assert parent_count <= baby_count, (parent_count, baby_count)
        return [Span(span.start + s.start, span.start + s.end) for s in spans]

    splitters = [split_paras, split_sents] #, split_words]

    def split(top_span, level):
        prn(f"*+*split: {level}: {top_span}")
        splitter = splitters[level]
        span_list = splitter(top_span)
        split_list = []
        for span in span_list:
            token_size = token_count(span)
            if token_size <= chunk_size:
                split_list.append(Split(span, level, token_size))
            else:
                assert level < SPLIT_WORD, (level, span, token_size)
                baby_split_list = split(span, level+1)
                baby_size = sum(s.token_size for s in baby_split_list)
                assert token_size <= baby_size, (token_size, baby_size, [s.token_size for s in baby_split_list])
                split_list.extend(baby_split_list)
        # assert split_list[0].span.start == 0, split_list[0]
        return split_list

    split_list = split(Span(0, len(text)), 0)   # top_span, level
    split_list.append(Split(Span(len(text), len(text)), 0, 0))

    prn(f"***split_to_sentences: split_list={len(split_list)} {split_list[:3]}", flush=True)
    offset = 0
    for i, split in enumerate(split_list):
        snippet = text[split.span.start:split.span.end]
        s, e = split.span
        prn(f"{i:4}: {split.level} {offset:4} {split.token_size:3} {s:4}-{e:4} '{format_chunk(snippet, 120)}'")
        offset += split.token_size

    offset_list = [] # The offset of the start of each split
    offset = 0
    for split in split_list:
        offset_list.append(offset)
        offset += split.token_size
    n_tokens = get_token_size(text)
    assert offset_list[-1] == n_tokens, (offset_list, n_tokens)

    def show_offsets():
        iparts = [f"{i:4}" for i in range(len(offset_list))]
        oparts = [f"{offset:4}" for offset in offset_list]
        prn(f"   offset_list={len(offset_list)}", flush=True)
        prn(f"     |{' |'.join(iparts)} |")
        prn(f"     |{' |'.join(oparts)} |")

    for i, split in enumerate(split_list):
        snippet = text[split.span.start:split.span.end]
        assert get_token_size(snippet) == split.token_size, (split, format_chunk(snippet, 120))

    prn(f"***split_to_sentences: split_list={len(split_list)} {split_list[:5]}", flush=True)
    for i, split in enumerate(split_list):
        offset = offset_list[i]
        snippet = text[split.span.start:split.span.end]
        s, e = split.span
        prn(f"{i:4}: {split.level} {offset:4} {split.token_size:3} {s:4}-{e:4} '{format_chunk(snippet, 120)}'")
    show_offsets()
    prn("-" * 80)

    for i, split in enumerate(split_list):
        offset = offset_list[i]
        d = split.span.end - split.span.start
        # prn(f"{i:4}: {split.level} {offset:4} {split.token_size:3} {d:3} {split.span}")
        assert split.token_size <= chunk_size, (split, split.token_size)
        if i > 0:
            last_split = split_list[i-1]
            last_offset = offset_list[i-1]
            last_split = split_list[i-1]
            snippet = text[last_split.span.start:last_split.span.end]
            last_token_size = get_token_size(snippet)
            assert last_split.span
            assert offset - last_offset == last_split.token_size, offset_list
            assert last_token_size == last_split.token_size, (last_token_size, last_split.token_size)
        if i > 1:
            last_split = split_list[i-2]
            last_offset = offset_list[i-2]
            last_split = split_list[i-2]
            snippet = text[last_split.span.start:split.span.start]
            last_token_size = get_token_size(snippet)
            sum_token_size = last_split.token_size + split_list[i-1].token_size
            assert offset - last_offset == sum_token_size, offset_list
            assert last_token_size <= sum_token_size, (last_token_size, sum_token_size)

    # Build the end list
    # chunk k is composed of split_list[end_list[k]:end_list[k+1]]
    #   (so end_list[k] is the index of the first split in chunk k+1)
    token_size = 0 # The token size of the current chunk
    end_list = []  # The index at which each chunk ends.
    last_k = 0     # The index of the last split in the previous chunk.
    current_k = 0  # The index of the current split

    for k, split in enumerate(split_list[:-1]):
        last_split = split_list[last_k]
        current_split = split_list[current_k]

        max_size = chunk_size if len(end_list) == 0 else chunk_size - chunk_overlap
        accept = token_size + split.token_size > max_size

        snippet = text[last_split.span.start:current_split.span.start]
        snippet_size = get_token_size(snippet)
        marker = "" if not accept else " ACCEPT"
        prn(f"  k={k:2}: max_size={max_size:3} ({last_k:2} {current_k:2} {k:2}) " +
              f"token_size={token_size:3} snippet_size={snippet_size:3} " +
              f"split={split} {marker}")
        assert snippet_size == token_size, (snippet_size, token_size)
        assert token_size <= chunk_size, (token_size, chunk_size)
        if accept:
            assert token_size <= chunk_size, (token_size, chunk_size)
            token_size = 0
            end_list.append(current_k)
            last_k = current_k
        token_size += split.token_size
        current_k = k + 1

    end_list.append(len(split_list)-1)

    prn(f"***split_to_sentences: end_list={len(end_list)} {end_list[:20]}", flush=True)
    s = 0
    total = 0
    for i, end_i in enumerate(end_list):
        split = split_list[end_i]
        e = split.span.start
        snippet = text[s:e]
        snippet_size = get_token_size(snippet)
        snippet_len = len(snippet)
        total += snippet_size
        prn(f"{i:4}: {end_i:3} {snippet_size:3} {total:4} " +
              f"{s:4}-{e:4}={snippet_len:4} '{format_chunk(snippet, n=140)}'")
        assert snippet_size < chunk_size
        s = e

    show_offsets()

    prn(f"***split_to_sentences: FIND STARTs")
    start_list = [] # The start of each chunk
    for i, end_i in enumerate(end_list):
        # Walk back to find the start of the chunk
        end_offset = offset_list[end_i]
        base_offset = max(0, end_offset - chunk_size) # The minimum valid start of the chunk
        start_i = end_i
        start_offset = offset_list[start_i]
        while start_i > 0:
            new_offset = offset_list[start_i - 1]
            if new_offset < base_offset:
                break
            start_i -= 1
            start_offset = offset_list[start_i]

        prn(f"{i:4}: base_offset={base_offset:4}  -=-----------------------------------------------------")
        prn(f"     -- start={start_i:3} start_offset={start_offset:4}")
        prn(f"     --   end={end_i:3}   end_offset={end_offset:4}")
        assert end_offset - start_offset <= chunk_size, (end_offset - start_offset, chunk_size)
        start_list.append(start_i)


    prn(f"***split_to_sentences: start_list={len(start_list):2} {start_list[:20]}")
    prn(f"                         end_list={len(end_list):2} {end_list[:20]}")
    prn(f"                         chunk_size={chunk_size} chunk_overlap={chunk_overlap}")
    last_end = 0
    for i, start_i in enumerate(start_list):
        end_i = end_list[i]
        start = split_list[start_i].span.start
        end = split_list[end_i].span.start
        snippet = text[start:end]
        snippet_len = len(snippet)
        start_offset = offset_list[start_i]
        end_offset = offset_list[end_i]
        overlap = last_end - start_offset
        last_end = end_offset
        expected = end_offset - start_offset
        actual = get_token_size(snippet)
        prn(f"{i:4}: {start_i:2} - {end_i:2} : {start_offset:4}-{end_offset:4} = " +
              f"{expected:3}>={actual:3} overlap={overlap:3} " +
              f"len={snippet_len:4} '{format_chunk(snippet, n=100)}'")
        assert expected >= actual, (expected, actual)
    show_offsets()

    chunk_list = []
    chunk_offset_list = []
    for start_i, end_i in zip(start_list, end_list):
        start_offset = offset_list[start_i]
        end_offset = offset_list[end_i]
        chunk_offset_list.append((start_offset, end_offset))
        start = split_list[start_i].span.start
        end = split_list[end_i].span.end
        chunk = text[start:end].strip()
        chunk_list.append(chunk)
    prn(f"***split_to_sentences: chunk_list={len(chunk_list)}", flush=True)
    for i, chunk in enumerate(chunk_list):
        start_offset, end_offset = chunk_offset_list[i]
        num_tokens = get_token_size(chunk)
        prn(f" @@@@@{i:4}: [{start_offset:4}-{end_offset:4} {num_tokens:3} {len(chunk):4}]::: '{format_chunk(chunk, n=100)}'")
    return chunk_list

text = """
Software developers should read this presentation on the human visual system for several key reasons:

I give this presentation to software developers in the hope of giving them some insights into the design and evolution of the human visual system.

1. Understanding biological visual processing can inspire more effective computer vision algorithms. The human visual system has evolved over millions of years to be highly optimized for tasks like object recognition, edge detection, and scene understanding.

2. There are important design principles and tradeoffs in the visual system, such as the compression of visual information from the retina to the optic nerve, the use of edge detection to efficiently encode visual scenes, and the hierarchical processing in the visual cortex. These insights can inform the design of computer vision systems that need to balance factors like resolution, bandwidth, and computational complexity.

Comprehending the biological constraints and user problems that shaped the evolution of the visual system can inspire novel approaches to computer vision challenged. e.g. the visual system is optimized for tasks essential to an animal's survival and reproduction, rather than trying to emulate a camera. This mindset can lead software developers to rethink the objectives and architectures of their computer vision mødels.

4. The de-tailed expl anations of visual processing stages, from optics to the visual cortex, provide a comprehensive overview of the human visual system that can deepen software developers' understanding of biological visual processing. This knowledge can inform the design of more biologically-inspired computer vision systems.[1]

In summary, this presentation offers software developers valuable insights into the design and evolution of the highly efficient and effective human visual system, which can inspire more innovative and optimized approaches to computer vision challenges.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/12265160/11dcd233-c993-4aa0-ad24-197d4917d535/Hüman Visual System.pdf


Understanding biological visual processing can inspire more efficient and effective computer vision algorithms. The human visual system has evolved over millions of years to be highly optimized for tasks like object recognition, edge detection, and scene understanding. By studying how the visual cortex processes information, software developers can draw parallels to the architecture of convolutional neural networks and other computer vision techniques

Software developers should delve into this presentation to gain a profound understanding of the human visual system, which can significantly enhance their ability to create innovative computer vision applications. By studying the intricacies of human vision, developers can draw inspiration for designing algorithms that mimic the efficiency and adaptability of biological vision systems. The presentation meticulously outlines the evolutionary progression of visual perception, providing insights into how the eye and brain work in concert to process visual information. This knowledge is invaluable for developers aiming to improve or innovate in areas such as facial recognition, object detection, and image segmentation[2][3].

Furthermore, the presentation highlights the parallels between the visual cortex's processing layers and the architecture of Convolutional Neural Networks (CNNs), a cornerstone of modern computer vision technology[1]. Understanding these similarities can guide developers in refining neural network models for more accurate and efficient visual data processing. Additionally, the presentation's exploration of edge detection and compression within the retina can inform strategies for data reduction and feature extraction in software development[1].

By integrating the principles outlined in this presentation, developers can create computer vision software that not only performs better but also provides more intuitive and human-like interactions, thereby improving user experience[3]. This can lead to more engaging and accessible applications across various domains, including security, healthcare, and entertainment. In essence, this presentation serves as a bridge between the fundamental biological processes of human vision and the cutting-edge techniques in computer vision, offering a unique perspective that can inspire and inform the next generation of software development.

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/12265160/11dcd233-c993-4aa0-ad24-197d4917d535/Human Visual System.pdf
[2] https://www.linkedin.com/advice/0/what-most-common-uses-computer-vision-software
[3] https://www.linkedin.com/advice/1/how-can-computer-vision-improve-user-experience-1vxlf
[4] https://www.springboard.com/blog/design/ui-vs-ux-vs-interaction-vs-visual-design/
[5] https://www.linkedin.com/pulse/introduction-user-interface-design-machinemindstechnologies-jl1ff
[6] https://www.interaction-design.org/literature/article/the-relationship-between-visual-design-and-user-experience-design
[7] https://www.invisionapp.com/defined/user-interface-design
[8] https://sdh.global/blog/development/what-computer-vision-is-applications-benefits-and-its-use-in-software-development/
[9] https://www.uxdesigninstitute.com/blog/visual-design-vs-ui-design/
[10] https://builtin.com/design-ux/user-interface-design
[11] https://www.interaction-design.org/literature/topics/ui-design
[12] https://uxdesign.cc/the-importance-of-visual-meaning-in-user-interface-design-f94e29a10903?gi=999051bd14c9
[13] https://en.wikipedia.org/wiki/Human_visual_system_model
[14] https://www.simplilearn.com/computer-vision-article
[15] https://www.leewayhertz.com/computer-vision-development/
[16] https://observablehq.com/blog/why-visualization-helps-developers
[17] https://viso.ai/applications/computer-vision-applications/
[18] https://www.augmentedstartups.com/blog/computer-vision-vs-human-vision-unveiling-the-battle-of-perception
[19] https://www.ibm.com/topics/computer-vision
[20] https://eecs.wsu.edu/~cs445/Lecture_2.pdf
ENDDDDD"""

def print_chunks(chunk_list, num_display):
    "Prints up to `num_display` of all the chunks in `chunk_list."
    print(f"chunks={len(chunk_list)} {[len(c) for c in chunk_list[:20]]}")
    for i, chunk in enumerate(chunk_list):
        num_tokens = get_token_size(chunk)
        print(f"{i:4}: {num_tokens:3} {len(chunk):4}]: '{format_chunk(chunk, n=num_display)}'")

def test_split(test_mine, test_llama, chunk_size, chunk_overlap, num_display, verbose):
    """
    Test the sentence splitter function.

    Args:
        test_mine (bool): Whether to test the custom splitter.
        test_llama (bool): Whether to test the LlamaIndex SentenceSplitter.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.
        num_display (int): The number of chunks to display.
        verbose (bool): Whether to display verbose output.
    """
    print("TEXT: =================================================================================")
    print(text)

    if test_mine:
        print("My splitter: ==========================================================================")
        chunk_list = split_to_sentences(text, chunk_size, chunk_overlap, verbose=verbose)
        print_chunks(chunk_list, num_display)

    if test_llama:
        print("LlamaIndex SentenceSplitter: ==========================================================")
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunk_list = splitter.split_text(text)
        print_chunks(chunk_list, num_display)

def main():
    """
    Compare my sentence splitter to llamaindex.

    Command-line arguments:
    --llama     : Use LlamaIndex only.
    --mine     : Use mine only.
    --verbose   : Enable verbose output.
    --chunk     : Chunk size (default: 200).
    --overlap   : Chunk overlap size (default: 60).
    --display   : Number of chunk chars displayed (default: 160).
    """
    from argparse import ArgumentParser
    parser = ArgumentParser(description=("Compare my sentence splitter to llamaindex."))
    parser.add_argument("--llama",   action="store_true",   help="Test LlamaIndex only.")
    parser.add_argument("--mine",    action="store_true",   help="Test mine only.")
    parser.add_argument("--chunk",   type=int, default=200, help="Chunk size.")
    parser.add_argument("--overlap", type=int, default=60,  help="Chunk overlap size.")
    parser.add_argument("--display", type=int, default=160, help="Number of chunk chars displayed.")
    parser.add_argument("--verbose", action="store_true",   help="Verbose output.")
    args = parser.parse_args()

    test_mine = not args.llama
    test_llama = not args.mine
    test_split(test_mine, test_llama, args.chunk, args.overlap, args.display, args.verbose)

if __name__ == "__main__":
    main()
