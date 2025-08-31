# Lemon rules inspector (Work in Progress):

This is a rag-based system that scrapes information from ["24 hours of Lemons"](https://24hoursoflemons.com/), creates embeddings, and calls out to openai.

## System Requirements:

- `uv`
- `chromedriver`

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

## Scripts

1. To scrape latest rules, use `scraper.py`, which will create a file called `lemons_rules.json`
2. To parse word chunks, use `web_rule_parser.py`, which will generate `parsed_rules.json`
3. To create embeddings, use `rule_chunker.py`, which will generate `rule_embeddings.npy`
4. To test workflow, run `rag_system.py`. This runs a set of test queries.

## Coming soon:

- testing with ollama locally
- exposing `rag_system.py` as an endpoint
- making it multimodal so that it can also ingest and display images
