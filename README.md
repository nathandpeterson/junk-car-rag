# Lemon rules inspector (Work in Progress):

This is a rag-based system that scrapes information from ["24 hours of Lemons"](https://24hoursoflemons.com/), creates embeddings, and calls out

1. To scrape latest rules, use `scraper.py`, which will create a file called `lemons_rules.json`
2. To create embeddings, use `web_rule_parser.py`, which will generate `rule_embeddings.npy`
3. To test workflow, run `rag_system.py`. This runs a set of test queries.

## System Requirements:

- `uv`
- `chromedriver`

## Coming soon:

- testing with ollama locally
- exposing `rag_system.py` as an endpoint
- making it multimodal so that it can also ingest and display images
