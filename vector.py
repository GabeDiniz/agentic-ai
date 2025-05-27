# Logic for embedding our documents and storing them in a vector database. This vectorizes our documents.

import os

import pandas as pd
from langchain_chroma import Chroma  # Vector database
from langchain_core.documents import Document  # Creates documents
from langchain_ollama import OllamaEmbeddings

# Load in csv file
df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # Embedding model

db_location = './chroma_db'  # Location of the vector database
add_documents = not os.path.exists(db_location)  # Check if the database already exists

if add_documents:
    documents = []
    ids = []

    # Go row by row through the csv file
    # and create a Document object for each review and store them in the database
    for i, row in df.iterrows():
        document = Document(
            page_content=row['Title'] + row['Review'],  # The content of the review
            metadata={"rating": row['Rating'], "date": row['Date']},  # Metadata for the document (additional information - but not queried on)
            id=str(i)  # Unique ID for the document
        )
        ids.append(str(i))  # Store the ID
        documents.append(document)

# Add to the vector database
vector_store = Chroma(
    collection_name="restaurant_reviews",  # Name of the collection in the vector database
    persist_directory=db_location,  # Store persistently (rather than in memory only) so you don't need to re-embed every time you run the code
    embedding_function=embeddings,  # The embedding model to use
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)  # Add the documents to the vector database

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}  # Number of documents to retrieve (i.e., finds 5 relevant reviews for a question)
)
