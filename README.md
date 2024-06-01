A second attempt at creating a custom AI assistant for computer science students at The University of Windsor.

Instead of a pure LLM finetune, RAG is being used to improve the quality of responses by including relevant information directly into the prompt for the LLM.

The user query is mapped to relevant documents in the corpus by using semantic search. Afterwards, the documents are included into the prompt to the AI academic advisor.
This allows the LLM to directly access relevant information and apply it to answer the user's query.

Uses LlamaIndex tooling to support RAG and Llama2 open source LLM to generate responses.