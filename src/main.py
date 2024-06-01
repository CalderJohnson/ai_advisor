"""CLI for interacting with the language model"""
from llm import generate_response

def main():
    query = ""
    while query != "exit":
        query = input("\nEnter your query: ")
        response, context = generate_response(query)
        print("\nResponse:", response)
        print("\nContext retrieved:", context)

if __name__ == "__main__":
    main()
