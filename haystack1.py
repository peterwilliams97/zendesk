from haystack_integrations.components.generators.ollama import OllamaGenerator

from haystack import Pipeline, Document
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.document_stores.in_memory import InMemoryDocumentStore

template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?
"""

docstore = InMemoryDocumentStore()
docstore.write_documents([Document(content="I really like summer"),
                          Document(content="My favorite sport is soccer"),
                          Document(content="I don't like reading sci-fi books"),
                          Document(content="I don't like crowded places"),])
# MODEL = "zephyr"
MODEL = "mistral"
generator = OllamaGenerator(model=MODEL,
                            url = "http://localhost:11434/api/generate",
                            generation_kwargs={
                              "num_predict": 100,
                              "temperature": 0.9,
                              })

pipe = Pipeline()
pipe.add_component("retriever", InMemoryBM25Retriever(document_store=docstore))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", generator)
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

baseQueries = [
    "What do I enjoy? Answer in a single sentence. Answer based on what you know.",
    "What do I dislike?",
    "How do I feel about playing football in summer in Florence?",
]
qualifiers = [
    "Answer in a single sentence.",
   "Answer based on what you know."
]
fullQualifier = " ".join(qualifiers)

queries = []
for q in baseQueries:
    queries.append(q)
    for a in qualifiers + [fullQualifier]:
        queries.append(f"{q} {a}")


for i, query in enumerate(queries):
    result = pipe.run({"prompt_builder": {"query": query},
                        "retriever":     {"query": query}})
    answer = result["llm"]["replies"][0]
    print(f"{i:2}: {query} --------------------")
    print(answer)



# {'llm': {'replies': ['Based on the provided context, it seems that you enjoy
# soccer and summer. Unfortunately, there is no direct information given about
# what else you enjoy...'],
# 'meta': [{'model': 'zephyr', ...]}}
