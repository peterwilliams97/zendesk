"""
    We find the closest tickets to a given ticket using the `Haystack` library.
    The JinaRanker is used to rank the tickets based on their similarity to the given ticket.
"""
import json
import os
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
from utils import load_text, deduplicate, save_json, load_json
from config import MODEL_ROOT, SIMILARITIES_ROOT
from zendesk_wrapper import comment_paths, load_index

TOP_K = 10
MAX_DEPTH = 4

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

def ticket_content(ticket):
    "Returns some content text of Zendesk ticket `ticket` for Jina to search and compare."
    comment = "\n--\n".join(ticket["comments"])
    text = f"{ticket['subject']} {comment}"
    return trim_tokens(text)

class ZendeskWrapper:
    """
    A class representing a Zendesk instance.

    Attributes:
        df (pandas.DataFrame): The full DataFrame containing ticket data.
        columns (list): A list of column names in the DataFrame.

    Methods:
        make_ticket: Creates a ticket dictionary for a given ticket number.
    """

    def __init__(self, df):
        self.df = df
        self.columns = [col for col in self.df.columns if not col.startswith("comments_")]

    def make_ticket(self, ticket_number, max_commments=5):
        """
        Returns a ticket dictionary for the given ticket number.

        Args:
            ticket_number (int): The ticket number.
            max_commments (int, optional): The maximum number of comments to include. Defaults to 5.

        Returns:
            dict: A dictionary representing the ticket, including metadata and comments.
        """
        metadata = self.df.loc[ticket_number]
        metadata = [metadata[col] for col in self.columns]
        input_files = comment_paths(ticket_number)
        comments = [load_text(path) for path in input_files[:max_commments]]
        assert len(metadata) == len(self.columns), (len(metadata), len(self.columns))
        ticket_dict = {"ticket_number": ticket_number}
        for k,v in zip(self.columns, metadata):
            ticket_dict[k] = v
        ticket_dict["comments"] = comments
        ticket_dict["created_at"] = ticket_dict["created_at"].isoformat()
        ticket_dict["updated_at"] = ticket_dict["updated_at"].isoformat()
        return ticket_dict

@component
class LoadTickets:
    @component.output_types(documents=List[Document])
    def run(self, query_engine):
        cleaned_tickets = []
        for i, ticket_number in enumerate(query_engine.df.index):
            ticket_dict = query_engine.make_ticket(ticket_number)
            cleaned_tickets.append(ticket_dict)
        return {"documents": cleaned_tickets}

@component
class TicketToText:
    @component.output_types(documents=List[Document])
    def run(self, tickets: List[Document]):
        tickets_documents = []
        max_i, max_len, max_tokens = 0, 0, 0
        for i, t in enumerate(tickets):
            ticket_number = t["ticket_number"]
            content = ticket_content(t)
            num_tokens = len(li_tokenizer(content))
            if num_tokens > max_tokens:
                max_i, max_len, max_tokens = i, len(content), num_tokens
            meta ={ k: t[k] for k in ["ticket_number", "subject", "priority",  "status"]}
            doc = Document(content=content, meta=meta)
            tickets_documents.append(doc)
        print(f"*** max: i={max_i} len={max_len} tokens={max_tokens}")
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
    indexing_pipeline = Pipeline()
    indexing_pipeline.add_component("loader", LoadTickets())
    indexing_pipeline.add_component("converter", TicketToText())
    indexing_pipeline.add_component("embedder", JinaDocumentEmbedder(model="jina-embeddings-v2-base-en"))
    indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP))

    indexing_pipeline.connect("loader", "converter")
    indexing_pipeline.connect("converter", "embedder")
    indexing_pipeline.connect("embedder", "writer")
    return indexing_pipeline

def build_query_pipeline(document_store):
    """
    Build a query pipeline with Jina components for embedding, retrieval, cleaning, and ranking.

    Args:
        document_store (DocumentStore): The document store used for retrieval.

    Returns:
        Pipeline: The query pipeline with all the components connected.
    """
    retriever = ChromaEmbeddingRetriever(document_store=document_store)
    query_pipeline = Pipeline()
    query_pipeline.add_component("query_embedder", JinaTextEmbedder(model="jina-embeddings-v2-base-en"))
    query_pipeline.add_component("query_retriever", retriever)
    query_pipeline.add_component("query_cleaner", RemoveRelated())
    query_pipeline.add_component("query_ranker", JinaRanker())

    query_pipeline.connect("query_embedder.embedding", "query_retriever.query_embedding")
    query_pipeline.connect("query_retriever", "query_cleaner")
    query_pipeline.connect("query_cleaner", "query_ranker")
    return query_pipeline

def round_score(score):
    "Rounds `score` to two decimal places."
    return int(round(100.0 * score)) / 100.0

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

    def __init__(self, df):
        os.makedirs(MODEL_ROOT, exist_ok=True)
        model_path = os.path.join(MODEL_ROOT, "chroma")
        document_store = ChromaDocumentStore(persist_path=model_path)
        self.document_store = document_store
        self.query_pipeline = build_query_pipeline(document_store)

        self.zd = ZendeskWrapper(df)

        if document_store.count_documents() < len(self.zd.df):
            print(f"document_store has {document_store.count_documents()}")
            print(f"Indexing {len(zself.zd.df)} tickets")
            indexing_pipeline = build_indexing_pipeline(document_store)
            indexing_pipeline.run({"loader": {self}})
            assert False, (document_store.count_documents(), len(self.zd.df))

    def make_ticket(self, ticket_number):
        """
        Creates a ticket object based on the ticket number.

        Args:
            ticket_number (int): The ticket number.

        Returns:
            Ticket: The ticket object.
        """
        return self.zd.make_ticket(ticket_number)

    def ticket_content(self, ticket_number):
        """
        Returns the content string for a given ticket number.

        Args:
            ticket_number (int): The ticket number.

        Returns:
            str: The content string for the ticket.
        """
        return ticket_content(self.make_ticket(ticket_number))

    def find_closest_tickets(self, ticket_number, top_k):
        """
        Finds the closest tickets based on the given ticket number, top_k.

        Args:
            ticket_number (int): The ticket number to find closest tickets for.
            top_k (int): The maximum number of closest tickets to retrieve.

        Returns:
            list: A list of tuples containing the ticket number and score of the closest tickets.
        """
        query_content = ticket_content(self.zd.make_ticket(ticket_number))
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
                  f"\texecption={e}\n" +
                  f"\tcontemt={repr(query_content[:20])}")
            return None
        docs = result["query_ranker"]["documents"]
        results = [(doc.meta["ticket_number"], doc.score) for doc in docs]
        # print(f"   @@@ {ticket_number} {top_k} {min_score:.2f} {len(docs)} results")
        results.sort(key=lambda x: (-round_score(x[1]), x[0]))
        return results

HAYSTACK_SUB_ROOT = os.path.join(SIMILARITIES_ROOT, "haystack")

class QueryEngine:
    """
    A class that represents a query engine for finding closest tickets based on ticket numbers.
    """

    def __init__(self, df):
        self.se = HaystackQueryEngine(df)
        os.makedirs(HAYSTACK_SUB_ROOT, exist_ok=True)

    def find_closest_tickets(self, ticket_numbers, top_k=TOP_K):
        """
        Finds the closest tickets based on the given ticket numbers.

        Args:
            ticket_numbers (list): A list of ticket numbers to find the closest tickets for.
            top_k (int, optional): The maximum number of closest tickets to return. Defaults to TOP_K.

        Returns:
            list: A list of tuples containing the ticket number and its corresponding closest tickets.
        """
        similarities_path = os.path.join(HAYSTACK_SUB_ROOT, "similarities.json")
        similarities = load_json(similarities_path) if os.path.exists(similarities_path) else {}
        num_changes = 0
        save_interval = 10
        for ticket_number in ticket_numbers:
            if ticket_number not in similarities:
                result = self.se.find_closest_tickets(ticket_number, top_k)
                if result is None:
                    continue
                similarities[ticket_number] = result
                num_changes += 1
                if num_changes >= save_interval:
                    print(f"Saving {num_changes} changes {len(similarities)} results to {similarities_path}")
                    save_json(similarities_path, similarities)
                    num_changes = 0
                    if save_interval < 1_000:
                        save_interval *= 10
        if num_changes > 0:
            save_json(similarities_path, similarities)
        return [(t, similarities[t]) for t in ticket_numbers]
