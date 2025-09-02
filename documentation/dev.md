Create virtual env with `uv venv`
Started with `source .venv/bin/activate`
Added fitz with `uv add fitz`

Futzed around trying to parse the 'how-to-not-fail-lemons-tech-inspection.pdf' with fitz.
Got it working but realized that this doc was non-heirarchical, so I was using wrong strategy.
This code is in 'structure_analyzer.py'. Not the right strategy for this file.

I ran into some errors around pdf ingestion that took up some time. First I started with a downloaded version of the rules pdf. Then, I decided to scrape it from the web instead of relying on having the file locally. I setup chromedriver to emulate uncollapsing each of the sections but it turned out that all of the text could be

Used `scraper.py` to scrape the content and extract the rules into json with a regex for detecting hierarchy. claude had a typo and the script fell back to using requests instead of BeautifulSoup - but we ended up extracting the content into `lemons_rules.json`, which I'm committing.

Now took the scraped json and claude created a script called `web_rule_parser.py` to make it into a
hierarchal document. web_rule_parser took lemon_rules.json (which only had the 6 top-level rules)
and parsed them into `parsed_rules.json`, which contains all the sub-rules with the hierarchy in place to be used as chunks.

My current flow is:
scrape --> structured json

Next, I asked claude to help with chunking and vectorizing. I used `rule_chunker.py` to take the
structured json, use each rule as a "chunk" for retrieval, and vectorize the results, the output of the above step was `rule_embeddings.npy`. Getting close so I didn't look very closely at the code. Did a sanity check with the output and things looked okay.

I had to set up an openAI account and fund it to get the chatbot style responses.

I also added logfire to rag_system.py, which instruments the api calls and shows token usage.
![logfire image](./assets/logfire.png)

The next session, I worked on adding the ability to run the same queries against a local version
of ollama running on my machine. I installed ollama, played around with some test queries, and
then refactored `rag_system.py` so that it was easy to swap between models by passing command
line arguments.

There's a branch where I ran multiple models side-by-side to see how they performed against one
another here: https://github.com/nathandpeterson/junk-car-rag/tree/test-multiple-models. For this
system, ollama has produced results pretty close to openai in quality. Given that we're retrieving
relevant context, it makes sense that results are okay but I'm curious what will happen after
we add the diagrams.

Next, I chatted with claude about the overall architecture and it generated `architecture.md`. This is a nice clear diagram of the flow. It would take some refactor to organize my code around this more modular approach, but that's what I would do next before adding the scrape/parse/vectorize flow for the other documents.
