<div align="center">

<img src="assets/hero-banner.svg" alt="AI Travel Agent" width="100%"/>

# AI Travel Agent

**Ask for a flight in plain language. Get real fares — never a guessed price.**

A flight search and comparison assistant powered by a **local LLM with real tool calling** —
not a chatbot that makes up flight data. The LLM only decides *which tool to call* and *how to
phrase the answer*; every airline, price, time, and airport code comes from a live API call.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/Agent-LangGraph-1C3C3C)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Tests](https://github.com/AbdullohRR/travel-agent/actions/workflows/tests.yml/badge.svg)](https://github.com/AbdullohRR/travel-agent/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## Demo

<div align="center">
<img src="assets/demo.gif" alt="AI Travel Agent live demo — searching and comparing real flights" width="720"/>
</div>

*Real conversation, recorded against the actual running app — not a mockup.*

### Screenshot: proof it's a real tool call, not a hallucinated answer

<div align="center">
<img src="assets/tool-log-screenshot.png" alt="Tool Logs tab showing the real flight_search_tool parameters and raw Travelpayouts API response" width="720"/>
</div>

The **Tool Logs** tab shows the exact parameters the LLM passed to
`flight_search_tool` and the raw JSON the Travelpayouts API returned —
the same data the chat answer above was built from.

## Project Overview

The agent understands natural-language requests (including Uzbek) like:

> "Toshkentdan Istanbulga 15-avgust kuni uchadigan reyslarni top."
> ("Find flights from Tashkent to Istanbul on August 15th.")

...decides to call `flight_search_tool`, hits the real Travelpayouts/Aviasales flight price API,
and formats the response. Follow-ups like *"only show Uzbekistan Airways"* or *"pick the cheapest"*
are answered by filtering/ranking the **already-cached** results — no repeated API call unless
the route, date, or passenger count actually changes.

> **API choice note:** the original design targeted the Amadeus Self-Service API, but Amadeus'
> developer signup was inaccessible during development, so the project uses
> [Travelpayouts](https://www.travelpayouts.com)'s free Aviasales flight-data API instead (an
> explicitly allowed alternative). Its price-data endpoint returns the *cheapest cached fare* per
> route/date (economy, 1 adult) rather than a full live multi-airline GDS search — see the
> caveat in Features below. Swapping in Amadeus later only requires rewriting
> `app/tools/flight_search.py` and `airport_search.py`; the agent/graph/UI layer is unchanged.

## Features

- **Real flight search** via the Travelpayouts/Aviasales price-data API (`/v1/prices/cheap`) —
  a single static API token, no OAuth handshake, issued instantly on signup (no approval wait).
  *Caveat:* this free endpoint returns the cheapest cached economy fare per route/date rather
  than a full live inventory search, so a single query often returns only one or a few fares; if
  the user asks for a passenger count or cabin class it can't actually price, the agent says so
  explicitly instead of pretending the number accounts for it.
- **Flight comparison** by price, duration, departure/arrival time, or stop count, with a
  reasoned recommendation.
- **Airport/city resolution** (`TAS` → Tashkent, "Istanbul airports" → IST/SAW) via
  Travelpayouts' autocomplete API — needs no API token at all.
- **Currency conversion** (bonus) via Frankfurter (no key needed) or exchangerate-api.com.
- **Multi-turn memory**: cached search results + active filters persist across turns in the
  same conversation, so "cheapest", "only X airline", "earlier flights" work without re-searching.
- **Slot-filling**: if a required field (e.g. the travel date) is missing and can't be inferred
  from context, the agent asks a clarifying question instead of guessing or failing.
- **Graceful error handling** for API failures, rate limits, no-results, and invalid/past dates.
- **Streamlit UI** with a chat interface, structured flight results, a comparison table, and a
  Tool Logs panel that shows exactly which tool ran, with what parameters, and the raw result —
  proof that the data is real, not hallucinated.

## Architecture

```
travel-agent/
  app/
    agent/
      graph.py     # LangGraph graph: agent <-> tools loop, checkpointed for memory
      nodes.py      # agent_node (intent parsing + response formatting), tools_node (execution)
      state.py       # AgentState: messages, cached search results, active filters, tool logs
    tools/
      flight_search.py    # Travelpayouts/Aviasales price-data search
      compare_flights.py  # rank/filter cached flights (never re-hits the API)
      airport_search.py   # Travelpayouts autocomplete (no token needed)
      currency.py           # live exchange rate conversion (bonus)
      travelpayouts_client.py  # shared HTTP client (token header + error translation)
    memory/          # LangGraph checkpointer (per-conversation thread state)
    ui/
      streamlit_app.py
    config.py
  tests/
  main.py
```

### LangGraph flow

```
        ┌────────┐   no tool_calls    ┌─────┐
 START →│ agent  │────────────────────→│ END │
        └───┬────┘                    └─────┘
            │ has tool_calls
            ▼
        ┌────────┐
        │ tools  │
        └───┬────┘
            │ loops back
            ▼
        ┌────────┐
        │ agent  │  (interprets tool results into a natural-language answer)
        └────────┘
```

- **`agent` node**: calls the Ollama model (bound to all 4 tools) with a system prompt describing
  the current cache state (route/date/pax already searched, active filters). On the first pass it
  decides which tool to call and with what arguments; on the loop-back (after tool results come
  in as `ToolMessage`s) it formats the final reply — the same node does both jobs.
- **`tools` node**: executes whichever tool(s) the model requested.
  - `flight_search_tool`: compares the requested route/date/passengers/cabin against the last
    cached search. If identical, it re-filters the cached raw results (no API call). If different,
    it calls the Travelpayouts API and refreshes the cache.
  - `compare_flights_tool`: never calls an API. It reads `last_search_results` from graph state,
    applies the model's airline/flight-number filter, ranks by the requested metric, and
    **persists the narrowed set back into state** — so a later "pick the cheapest" operates on
    the already-filtered list, not the original full search.
  - `airport_search_tool` / `currency_conversion_tool`: stateless, always call their real API.
  - Any validation error (e.g. a missing date) is caught and turned into an error `ToolMessage`
    instead of crashing — the next `agent` turn reads that and asks the user a clarifying question.
- **Memory**: a `MemorySaver` checkpointer (see `app/memory`) persists `AgentState` per
  `thread_id`, which is how multi-turn refinement works without the UI re-sending full history.

## Installation

```bash
cd travel-agent
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # then fill in your Travelpayouts token
```

You need one LLM backend (local Ollama needs no key) and a free Travelpayouts
token for live fares. Both, plus optional Streamlit Cloud hosting, are covered
in [Setup](docs/SETUP.md).

## Run Instructions

```bash
python main.py
# or equivalently:
streamlit run app/ui/streamlit_app.py
```

Then open the URL Streamlit prints (default `http://localhost:8501`).

### Run tests

```bash
pytest
```

All tool logic and agent-node caching/filtering logic is covered with mocked HTTP responses
(via `responses`) — no live API keys or a running Ollama instance are needed to run the test
suite.

## Future Improvements

- Persist conversations to disk/Postgres instead of in-memory checkpointing, so history survives
  a restart.
- Flight status / delay lookups.
- PDF itinerary export.
- Travel checklist generator.
- Docker Compose setup bundling the app + Ollama.
- Streaming token-by-token responses in the chat UI.

## Documentation

| | |
|---|---|
| [Setup](docs/SETUP.md) | Travelpayouts API token, airport lookup, currency conversion, Ollama vs Gemini, and Streamlit Cloud hosting |

## License

[MIT](LICENSE)
