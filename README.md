# LangChain Perplexity Examples

## Example apps
* 01-perplexity-single-shot.py     - Prompt the user for a single round chat
* 02-perplexity-chain.py           - Chain user input during a chat until the user exits
* 03-perplexity-multi-character.py - An automated chat between two ai agents
* 04-perplexity-rag-rules.py       - Chat with RAG backed ruled expert

## Dev setup
```
source .venv/bin/activate .venv
cp .env.example .env # and add key to .env
pip install -r requirements.txt
```

## Setup history
* python -m venv .venv
* pip freeze > requirements.txt
