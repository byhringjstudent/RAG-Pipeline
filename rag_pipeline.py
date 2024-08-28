# Building RAG Pipelines

import os
from haystack import Pipeline, Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.utils import Secret

# Initialize the document store and write documents
document_store = InMemoryDocumentStore()
document_store.write_documents([
    Document(content="My name is Jean and I live in Paris."),
    Document(content="My name is Mark and I live in Berlin"),
    Document(content="My name is Giorgio and I live in Rome")
])

# Define the prompt template
prompt_template = """
Given these documents, answer the question.
Documents:
{% for doc in documents %}
    {{ doc.content }}
{% endfor %}
Question: {{question}}
Answer:
"""

# Step 2:
retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_template)
llm = OpenAIGenerator(api_key=Secret.from_token("sk-5Q1qChKzUTvwy5knFM9mzoIfBrS3BNnewXcu_6k0LYT3BlbkFJx5YQchskmRhgo6rnzCgGCPOXHAP0iOhYHkJu8zrPYA"))

# Step 3:
rag_pipeline = Pipeline()

# Step 4:
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

# Step 5:
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Questions to ask
questions = ['Who lives in Paris?', 'Who lives in Berlin?', 'Who lives in Rome?']

# Iterate over each question and get the answers
for question in questions:
    results = rag_pipeline.run(
        {
            "retriever": {"query": question},
            "prompt_builder": {"question": question},
        }
    )

    # Extract and print the answer for each question
    answers = results.get('llm', {}).get('replies', [])
    if answers:
        separated_answers = answers[0].split("\n")
        for answer in separated_answers:
            print(f"{question}: {answer}")
    else:
        print(f"{question}: No answer found")
