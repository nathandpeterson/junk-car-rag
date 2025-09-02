# Lemon rules inspector (Work in Progress):

This is a rag-based system that scrapes information from ["24 hours of Lemons"](https://24hoursoflemons.com/), creates embeddings, and calls out to openai.

## System Requirements:

- `uv`
- `chromedriver`
- `ollama` (not required if you use retrieval-only or openai)

## Instructions

After cloning down the repo, follow these steps:

```
# set up virtual environment
uv venv
source .venv/bin/activate

# download requirements
uv sync
```

If you want to run with OpenAI, add a `.env` file and add `OPENAI_API_KEY`.

If you want to run with Ollama, download the model from [here](https://ollama.com/library/llama3.2).

Once you've dowloaded ollama, get the latest model, e.g.:
`ollama run llama3.2`

## Scripts

1. To scrape latest rules, use `extractors/scraper.py https://24hoursoflemons.com/prices-rules/`, which will create a file called `lemons_rules.json` in the `data/` folder
2. To parse word chunks, use `web_rule_parser.py`, which will generate `parsed_rules.json`
3. To create embeddings, use `rule_chunker.py`, which will generate `rule_embeddings.npy`
4. To test workflow, run `rag_system.py`. This runs a set of test queries.

## CLI

# Simple question with default Ollama model

`python rag_system.py -q "What's the budget limit?"`

# Use a specific Ollama model

`python rag_system.py -m llama3.2 -q "Can I upgrade my transmission?"`

# Use OpenAI GPT-4 (requires OPENAI_API_KEY)

`python rag_system.py -p openai -m gpt-4 -q "What are roll cage requirements?"`

# Interactive chat mode

`python rag_system.py --interactive`

## Coming soon:

- exposing `rag_system.py` as an endpoint
- making it multimodal so that it can also ingest and display images
