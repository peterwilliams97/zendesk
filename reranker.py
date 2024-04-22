"""
    We find the closest tickets to a given ticket using the `Haystack` library.
    The JinaRanker is used to rank the tickets based on their similarity to the given ticket.

    This has some outrageous global variables that are used to avoid putting @component functions
    inside functions and using closures to pass arguments.
"""
import json
import os
import sys
import time
from typing import List
from haystack import Document, component
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.components.embedders.jina import JinaDocumentEmbedder
from haystack import Pipeline
from haystack_integrations.components.embedders.jina import JinaTextEmbedder
from haystack_integrations.components.rankers.jina import JinaRanker
from llama_index.core.utils import get_tokenizer
from llama_index.core.text_splitter import SentenceSplitter
from utils import (load_text, deduplicate, save_json, load_json, since, text_lines, round_score,
                   SummaryReader)
from config import MODEL_ROOT, SIMILARITIES_ROOT, DIVIDER, FILE_ROOT
from zendesk_wrapper import comment_paths, make_empty_index
from rag_classifier import PydanticFeatureGenerator

TOP_K = 10
RECURSIVE_THRESHOLD = 0.8

HAYSTACK_SUB_ROOT = os.path.join(SIMILARITIES_ROOT, "summaries.haystack")
SIMILARITIES_PATH = os.path.join(HAYSTACK_SUB_ROOT, "similarities.json")
CHROMA_MODEL_PATH = os.path.join(MODEL_ROOT, "database")
CHROMA_UPLOADED_PATH = os.path.join(MODEL_ROOT, "uploaded.json")

def make_paths(model_name):
    global HAYSTACK_SUB_ROOT, SIMILARITIES_PATH, CHROMA_MODEL_PATH, CHROMA_UPLOADED_PATH
    suffix = f"{model_name}"
    HAYSTACK_SUB_ROOT = os.path.join(SIMILARITIES_ROOT, suffix, "haystack")
    SIMILARITIES_PATH = os.path.join(HAYSTACK_SUB_ROOT, "similarities.json")
    CHROMA_MODEL_PATH = os.path.join(MODEL_ROOT, suffix, "database")
    CHROMA_UPLOADED_PATH = os.path.join(MODEL_ROOT, suffix, "uploaded.json")

# We use the SentenceSplitter to split the text into chunks that are small enough that Jina won't
# create text framgments that are larger than its internal limit of JINA_TOKEN_LIMIT tokens.
JINA_TOKEN_LIMIT = 8192
MAX_TOKENS = int(round(JINA_TOKEN_LIMIT * 0.6))
splitter = SentenceSplitter(chunk_size=MAX_TOKENS, chunk_overlap=0)

# The tiktoken tokenizer from LlamaIndex.
li_tokenizer = get_tokenizer()

def trim_tokens(text):
    "Trims `text` to MAX_TOKENS tokens."
    chunks = splitter.split_text(text)
    return chunks[0]

SECTION_NAMES = ["PRODUCT", "FEATURES", "CLASS", "DEFECT", "CHARACTERISTICS",
                #  "DESCRIPTION", , "SUMMARY", , "PROBLEMS"
                 ]
reader = SummaryReader(SECTION_NAMES)

def summary_to_content(text):
    "Convert a text summary into content format."
    sections = reader.summary_to_sections(text)
    get = lambda key: sections.get(key, "not specified")
    rows = []
    for name in SECTION_NAMES:
        sep = "\n" if name == "PROBLEMS" else " "
        val = get(name)
        rows.append(f"{name}:{sep}{val}")
    return "\n\n".join(rows)

class ZendeskWrapper:
    def __init__(self, df, summariser):
        """
        Initialize the Reranker object.

        Args:
            df (pandas.DataFrame): The input DataFrame containing ticket data.
            summariser (Summariser): An instance of the Summariser class.

        Attributes:
            summariser (Summariser): The Summariser object used for summarization.
            df (pandas.DataFrame): The filtered DataFrame containing ticket data for which summaries exist.
            columns (list): The list of columns in the filtered DataFrame.

        """
        self.summariser = summariser
        filter_df = make_empty_index(add_custom_fields=True)
        for ticket_number in df.index:
            summary_path = self.summariser.summary_path(ticket_number)
            if os.path.exists(summary_path):
                filter_df.loc[ticket_number] = df.loc[ticket_number]
        print(f"ZendeskWrapper: {len(filter_df)} tickets of {len(df.index)} with summaries")
        self.df = filter_df
        self.columns = [col for col in self.df.columns if not col.startswith("comments_")]

    def make_ticket(self, ticket_number):
        summary_path = self.summariser.summary_path(ticket_number)
        metadata = self.df.loc[ticket_number]
        metadata = [metadata[col] for col in self.columns]
        # input_files = comment_paths(ticket_number)
        # comments = [load_text(path) for path in input_files[:max_commments]]
        assert len(metadata) == len(self.columns), (len(metadata), len(self.columns))
        ticket_dict = {"ticket_number": ticket_number}
        for k,v in zip(self.columns, metadata):
            ticket_dict[k] = v
        # ticket_dict["comments"] = comments
        ticket_dict["created_at"] = ticket_dict["created_at"].isoformat()
        ticket_dict["updated_at"] = ticket_dict["updated_at"].isoformat()
        return ticket_dict

hsqe = None

@component
class LoadTickets:
    @component.output_types(documents=List[Document])
    def run(self, ticket_numbers: List[int]):
        # print("**** LoadTickets")
        tickets = []
        for ticket_number in ticket_numbers:
            # print(f"*** ticket_number={ticket_number}: {ticket_number in hsqe.uploaded}")
            if ticket_number in hsqe.uploaded:
                continue
            ticket_dict = hsqe.make_ticket(ticket_number)
            tickets.append(ticket_dict)
        hsqe.save_uploaded()
        return {"documents": tickets}

@component
class TicketToText:
    @component.output_types(documents=List[Document])
    def run(self, tickets: List[Document]):
        tickets_documents = []
        max_i, max_len, max_tokens = 0, 0, 0
        for i, t in enumerate(tickets):
            ticket_number = t["ticket_number"]
            content = hsqe.ticket_content(ticket_number)
            num_tokens = len(li_tokenizer(content))
            if num_tokens > max_tokens:
                max_i, max_len, max_tokens = i, len(content), num_tokens
            meta ={ k: t[k] for k in ["ticket_number", "subject", "priority",  "status"]}
            doc = Document(content=content, meta=meta)
            tickets_documents.append(doc)
            hsqe.uploaded.add(ticket_number)
        hsqe.save_uploaded()
        return {"documents": tickets_documents}

@component
class RemoveRelated:
    "Removes document with ticket number `query_id` from the list of documents."
    @component.output_types(documents=List[Document])
    def run(self, tickets: List[Document], query_id: int):
        retrieved_tickets = [t for t in tickets if t.meta["ticket_number"] != query_id]
        return {"documents": retrieved_tickets}

def build_indexing_pipeline(document_store):
    """
    Builds an indexing pipeline for processing tickets and indexing them into a document store.

    Args:
        document_store (DocumentStore): The document store to index the tickets into.

    Returns:
        Pipeline: The indexing pipeline.

    """
    pipeline = Pipeline()
    pipeline.add_component("loader", LoadTickets())
    pipeline.add_component("converter", TicketToText())
    pipeline.add_component("embedder", JinaDocumentEmbedder(model="jina-embeddings-v2-base-en"))
    pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))

    pipeline.connect("loader", "converter")
    pipeline.connect("converter", "embedder")
    pipeline.connect("embedder", "writer")
    return pipeline

def build_query_pipeline(document_store):
    """
    Build a query pipeline with Jina components for embedding, retrieval, cleaning, and ranking.

    Args:
        document_store (DocumentStore): The document store used for retrieval.

    Returns:
        Pipeline: The query pipeline with all the components connected.
    """
    retriever = ChromaEmbeddingRetriever(document_store=document_store)
    pipeline = Pipeline()
    pipeline.add_component("query_embedder", JinaTextEmbedder(model="jina-embeddings-v2-base-en"))
    pipeline.add_component("query_retriever", retriever)
    pipeline.add_component("query_cleaner", RemoveRelated())
    pipeline.add_component("query_ranker", JinaRanker())

    pipeline.connect("query_embedder.embedding", "query_retriever.query_embedding")
    pipeline.connect("query_retriever", "query_cleaner")
    pipeline.connect("query_cleaner", "query_ranker")
    return pipeline

class HaystackQueryEngine:
    """
    A class representing a query engine for the Haystack system.

    Args:
        df (pandas.DataFrame): The DataFrame containing the ticket data.

    Attributes:
        document_store (ChromaDocumentStore): The document store used for indexing.
        query_pipeline (Pipeline): The query pipeline used for searching.

    Methods:
        make_ticket: Creates a ticket object based on the ticket number.
        ticket_content: Returns the content string for a given ticket number.
        find_closest_tickets: Finds the closest tickets based on the given ticket number, top_k.
    """
    def __init__(self, df, summariser):
        self.summariser = summariser
        os.makedirs(MODEL_ROOT, exist_ok=True)
        self.document_store = ChromaDocumentStore(persist_path=CHROMA_MODEL_PATH)
        self.query_pipeline = None #  build_query_pipeline(document_store)
        uploaded = load_json(CHROMA_UPLOADED_PATH) if os.path.exists(CHROMA_UPLOADED_PATH) else []
        self.uploaded = set(uploaded)

        self.zd = ZendeskWrapper(df, summariser)

    def save_uploaded(self):
        "Saves the uploaded ticket numbers to a file."
        save_json(CHROMA_UPLOADED_PATH, list(self.uploaded))

    def make_ticket(self, ticket_number):
        "Creates a Zendesk ticket object based on the ticket number."
        return self.zd.make_ticket(ticket_number)

    def ticket_content(self, ticket_number, allow_not_exist=False):
        """
        Returns the content string for ticket number `ticket_number`.
        The content is used by Jina for searching and comparing tickets.
        """
        path = self.summariser.summary_path(ticket_number)
        if allow_not_exist and not os.path.exists(path):
            return "No summary available"
        text = load_text(path)
        return summary_to_content(text)

    def find_closest_tickets(self, ticket_number, top_k):
        """
        Finds the closest `top_k` tickets to the ticket numbered `ticket_number`.

        Args:
            ticket_number (int): The ticket number to find closest tickets for.
            top_k (int): The maximum number of closest tickets to retrieve.

        Returns:
            list: A list of tuples containing the ticket number and score of the closest tickets.
        """
        assert self.query_pipeline is not None, "query_pipeline not set"
        if ticket_number not in self.zd.df.index:
            print(f"find_closest_tickets: ticket {ticket_number} not found", file=sys.stderr)
            return None
        query_content = self.ticket_content(ticket_number)
        try:
            result = self.query_pipeline.run(
                  data={"query_embedder":  {"text": query_content},
                        "query_retriever": {"top_k": max(20, 4 * top_k)},
                        "query_cleaner":   {"query_id": ticket_number},
                        "query_ranker":    {"query": query_content,
                                            "top_k": top_k}
                        }
                    )
        except Exception as e:
            print(f"find_closest_tickets: ticket={ticket_number}\n" +
                  f"\texception={e}\n" +
                  f"\tcontent={repr(query_content[:200])}",
                  file=sys.stderr)
            raise
            return None
        docs = result["query_ranker"]["documents"]
        results = [(doc.meta["ticket_number"], doc.score) for doc in docs]
        results.sort(key=lambda x: (-round_score(x[1]), x[0]))
        return results

def load_hsqe(df, llm, model_name):
    """
    Load the Haystack Query Engine (hsqe) with the given DataFrame, summariser, model name, and summariser name.
    BIG HACK: This is a global variable that is used to avoid reloading the Haystack Query Engine.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data to be loaded into the Haystack Query Engine.
        model_name (str): The name of the model to be used by the Haystack Query Engine.

    Returns:
        HaystackQueryEngine: The loaded Haystack Query Engine instance.

    """
    global hsqe

    summariser = PydanticFeatureGenerator(llm, model_name)

    make_paths(model_name)

    if hsqe:
        return hsqe

    hsqe = HaystackQueryEngine(df, summariser)

    ticket_numbers = hsqe.zd.df.index
    store_count = hsqe.document_store.count_documents()
    print(f"document_store has {store_count} documents")
    print(f"Indexing {len(ticket_numbers)} tickets")

    hsqe.indexing_pipeline = build_indexing_pipeline(hsqe.document_store)
    hsqe.indexing_pipeline.run({"loader": {"ticket_numbers": ticket_numbers}})

    hsqe.query_pipeline = build_query_pipeline(hsqe.document_store)

    return hsqe

class QueryEngine:
    """
    A class that represents a query engine for finding closest tickets based on ticket numbers.
    """

    def __init__(self, df, llm, model_name):
        self.hsqe = load_hsqe(df, llm, model_name)
        os.makedirs(HAYSTACK_SUB_ROOT, exist_ok=True)
        similarities = load_json(SIMILARITIES_PATH) if os.path.exists(SIMILARITIES_PATH) else {}
        self.similarities = {int(k): v for k, v in similarities.items()}

    def ticket_numbers(self):
        "Returns the ticket numbers in the Zendesk data."
        return self.hsqe.zd.df.index

    def ticket_content(self, ticket_number, allow_not_exist=False):
        "Returns the content string for the given ticket number."
        return self.hsqe.ticket_content(ticket_number, allow_not_exist=allow_not_exist)

    def find_closest_tickets(self, ticket_numbers, top_k=TOP_K):
        """
        Finds the closest tickets based on the given ticket numbers.

        Args:
            ticket_numbers (list): A list of ticket numbers to find the closest tickets for.
            top_k (int, optional): The maximum number of closest tickets to return. Defaults to TOP_K.

        Returns:
            list: A list of tuples containing the ticket number and its corresponding closest tickets.
        """
        num_changes = 0
        save_interval = 10
        t0 = time.time()
        for i, ticket_number in enumerate(ticket_numbers):
            n_similar = len(self.similarities.get(ticket_number, {}))
            if n_similar < top_k:
                result = self.hsqe.find_closest_tickets(ticket_number, top_k)
                if result is None:
                    continue
                self.similarities[ticket_number] = result
                num_changes += 1
                if num_changes >= save_interval:
                    print(f"Saving {num_changes:4} changes {len(self.similarities):5} results to " +
                          f"{SIMILARITIES_PATH} {since(t0):4.1f} sec")
                    save_json(SIMILARITIES_PATH, self.similarities)
                    num_changes = 0
                    t0 = time.time()
                    if save_interval < 1_000:
                        save_interval *= 10
        if num_changes > 0:
            save_json(SIMILARITIES_PATH, self.similarities)
        return [(t, self.similarities[t]) for t in ticket_numbers if t in self.similarities]

    def find_closest_tickets_recurse(self, ticket_numbers, top_k=TOP_K, max_results=3*TOP_K,
                                     threshold=RECURSIVE_THRESHOLD):
        top_k = max(1, top_k)
        tickets_found = set()
        tickets_remaining = ticket_numbers[:]
        all_results = []
        max_rounds = max_results // top_k
        for i in range(max_rounds):
            # print(f"Round {i+1} of {max_rounds} all_results={len(all_results)} " +
            #       f"tickets_found={len(tickets_found)} {sorted(tickets_found)[:20]}")
            results = self.find_closest_tickets(tickets_remaining, top_k)
            tickets_remaining = []
            # print(f"  results={len(results)} {results[:10]}")
            for (query_number, result) in results:
                result = [(ticket_number, score) for ticket_number, score in result
                          if ticket_number not in tickets_found ]
                if not result:
                    continue
                all_results.append((query_number, result))
                for ticket_number, _ in result:
                    if ticket_number in tickets_found:
                        continue
                    tickets_found.add(ticket_number)
                    tickets_remaining.append(ticket_number)
                if len(tickets_found) >= max_results:
                    break
            if len(tickets_found) >= max_results:
                    break
        return all_results
