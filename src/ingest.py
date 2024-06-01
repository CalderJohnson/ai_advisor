"""Data ingestion pipeline to create vector database based on the input corpus"""
import json
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode

from db import vector_store, conn, db_name

def load_block():
    """Generator that yields one chunk of text at a time from the input corpus"""
    with open("./data/corpus.txt", "r") as f:
        block = ""
        for line in f:
            if line.strip() == "":
                if block:
                    yield block
                    block = ""
            else:
                block += line
        if block:
            yield block

with open("./paths.json", "r") as f:
    paths = json.load(f)
    name = "BAAI/bge-small-en"
    if "bge-small-en" in paths:
        name = paths["bge-small-en"]

    # Embedding model
    embed_model = HuggingFaceEmbedding(model_name=name)

with conn.cursor() as c:
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")

# Create embeddings for each block of text and store in the database
nodes = []
for block in load_block():
    text_node = TextNode(text=block)
    node_embedding = embed_model.get_text_embedding(
        text_node.get_content(metadata_mode="all")
    )
    text_node.embedding = node_embedding
    nodes.append(text_node)

# Store text nodes
vector_store.add(nodes)
