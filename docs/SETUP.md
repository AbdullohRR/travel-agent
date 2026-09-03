# API keys, LLM providers and hosting

What each external service is for and how to configure it: the Travelpayouts flight-data
API, airport lookup, currency conversion, the two LLM paths (local Ollama or cloud Gemini),
and hosting a public demo on Streamlit Community Cloud.

[← back to the README](../README.md)

---

## API Usage & Setup

### Travelpayouts / Aviasales Flight Data API (flight prices)

1. Sign up free at <https://www.travelpayouts.com> (email + basic info — no approval process,
   no credit card, no website required).
2. Log in, go to your **Profile**, and copy the **API token** shown there — it's active
   immediately.
3. Put it in `.env`:
   ```
   TRAVELPAYOUTS_API_TOKEN=your_token_here
   ```

The `/v1/prices/cheap` endpoint returns the cheapest *cached* fare per route/date (economy,
1 adult), refreshed periodically — not a live multi-airline GDS search. If a query returns no
results, try a well-known route (major city pairs) a few weeks out; very obscure routes or
same-day dates may have no cached fare yet.

### Airport/city lookup

Uses Travelpayouts' autocomplete endpoint (`autocomplete.travelpayouts.com/places2`), which
needs **no API token at all** — nothing to configure.

### Currency conversion (bonus)

Uses [Frankfurter](https://frankfurter.dev) by default (no key needed, ECB rates — doesn't cover
UZS). To convert to/from currencies Frankfurter lacks (like UZS), get a free key at
<https://www.exchangerate-api.com> and set `EXCHANGE_RATE_API_KEY` in `.env`.

## LLM Setup — two options

### Option A — Local, no API key (Ollama)

1. Install Ollama: <https://ollama.com/download>
2. Pull a tool-calling-capable model:
   ```
   ollama pull qwen2.5:7b-instruct-q4_K_M   # recommended if your machine can handle it
   ollama pull llama3.2:3b                    # lighter fallback
   ```
3. Make sure the server is running (`ollama serve`, or it starts automatically on most installs).
4. Set the model in `.env` (or pick it live from the Streamlit sidebar):
   ```
   OLLAMA_MODEL=qwen2.5:7b-instruct-q4_K_M
   ```

### Option B — Cloud (Gemini), your own key

No Ollama needed. Pick **"Gemini (cloud)"** in the sidebar's LLM Provider dropdown and paste a
free key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — used only for
your session, never stored server-side. This is what makes a public deployment (see Deploy below)
usable with zero setup on the deployer's end: tool-calling works identically to Ollama since both
go through the same `.bind_tools()` LangChain interface (Gemini via its
[OpenAI-compatible endpoint](https://ai.google.dev/gemini-api/docs/openai), reusing
`langchain-openai`'s `ChatOpenAI` rather than adding a Gemini-specific package).

The provider/key are threaded through LangGraph's per-invocation `RunnableConfig`
(`configurable.llm_provider`/`api_key`), not global config mutation — so on a shared public
deployment, two visitors who each pick Gemini and paste different keys never get their
requests mixed up.

## ☁️ Deploy your own (Streamlit Community Cloud, free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub, click **New app**.
3. Point it at this repo, branch `main`, main file path `app/ui/streamlit_app.py`.
4. In **Advanced settings → Secrets**, add:
   ```
   LLM_PROVIDER = "gemini"
   TRAVELPAYOUTS_API_TOKEN = "your_token_here"
   ```
   `LLM_PROVIDER` makes Gemini the default a visitor sees immediately (no
   Ollama server exists on Streamlit Cloud, so without this they'd hit an
   "Ollama unreachable" error before finding the provider dropdown).
   `TRAVELPAYOUTS_API_TOKEN` is yours to configure since flight data needs
   it regardless of LLM provider (free signup, see API Usage above).
5. Deploy. Visitors immediately see the Gemini API key field and paste
   their own free key — no LLM secret to configure on your end.
