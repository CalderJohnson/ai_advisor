"""LLM module for generating responses"""
import json

from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.query_engine import RetrieverQueryEngine

from loader import loader

MODEL_URL = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
MODEL_PATH = None

with open("./paths.json", "r") as f:
    paths = json.load(f)
    if "llama-2-7b-chat" in paths:
        MODEL_PATH = paths["llama-2-7b-chat"]

llm = LlamaCPP(
    model_url=MODEL_URL,
    model_path=MODEL_PATH,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 0}, # Use CPU for inference
    verbose=True,
)
query_engine = RetrieverQueryEngine.from_args(loader, llm=llm)

def generate_response(query: str):
    """Generate response using the LLM"""
    response = query_engine.query(query)

    # Return the generated response and the context
    return str(response), response.source_nodes[0].get_content()
