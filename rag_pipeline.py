# Building RAG Pipelines

# Step 1: Import necessary libraries
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
    Document(content="An AI algorithm is sometimes called a model."),
    Document(content="Neural networks, Support Vector Machines, Bayesian Networks, and Hidden Markov Models."),
    Document(content="Data Classification, Regression analysis, Clustering, Time Series."),
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

# Step 2: Initialize components
retriever = InMemoryBM25Retriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt_template)

# Pass the API key directly as a string
llm = OpenAIGenerator(api_key=Secret.from_token("sk-ul1rugczxSUHKsePaTc-8VORwg5JkJjGgavniptGsrT3BlbkFJXq03HZ7446TRkpUEL_J41rWkk1YEDB407vHKgKPKEA"))


# Step 3: Create the RAG pipeline
rag_pipeline = Pipeline()

# Step 4: Add components to the pipeline using add_component and connect
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("prompt_builder", prompt_builder)
rag_pipeline.add_component("llm", llm)

# Connect components
rag_pipeline.connect("retriever", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "llm")

# Step 5: Define Questions and Run Pipeline
questions = [
    "An AI algorithm is sometimes called this:",
    "Among the most common AI algorithms are these:",
    "Knowing how to model a real-world program to a machine-learning algorithm is a critical task! The four different ways of using AI algorithms to solve problems are:",
]

# Iterate over each question and get the answers
for question in questions:
    results = rag_pipeline.run({
        "retriever": {"query": question},
        "prompt_builder": {"question": question},
    })

    # Extract and print the answer for each question with proper formatting
    print(f"Q: {question}")
    
    if (answers := results.get('llm', {}).get('replies', [])):
        separated_answers = answers[0].split("\n")
        for answer in separated_answers:
            print(f"A: {answer}")
    else:
        print("A: No answer found")

    # Print a separator line between Q&A pairs
    print("\n" + "-" * 40 + "\n")
