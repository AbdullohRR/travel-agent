"""Tests for the LangGraph node logic that implements multi-turn memory and slot-filling:
cache-hit vs. cache-miss flight search, compare_flights_tool's filter/narrow/persist behavior,
and graceful handling of missing/invalid tool arguments (the slot-filling trigger). These test
tools_node()/route_after_agent() directly against constructed state — no live Ollama call — so
they're fast and deterministic.
"""
from datetime import date, timedelta

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.nodes import _merge_unique_flights, route_after_agent, tools_node
from app.agent.state import new_state

PRICES_URL = "https://api.travelpayouts.com/v1/prices/cheap"

# flight_search rejects a departure date in the past, so the date used to drive
# a search has to move with the calendar. A literal here passes on the day it is
# written and starts failing silently once it goes by -- which is exactly what
# happened to the previous "2026-08-15".
SEARCH_DATE = (date.today() + timedelta(days=30)).isoformat()

CACHED_FLIGHTS = [
    {
        "airline": "Uzbekistan Airways",
        "flight_number": "HY701",
        "departure_airport": "TAS",
        "arrival_airport": "IST",
        "departure_time": "2026-08-15T07:20:00",
        "arrival_time": "2026-08-15T09:30:00",
        "duration": "4h 10m",
        "stops": 0,
        "cabin_class": "economy",
        "price": 180.0,
        "currency": "USD",
    },
    {
        "airline": "Turkish Airlines",
        "flight_number": "TK372",
        "departure_airport": "TAS",
        "arrival_airport": "IST",
        "departure_time": "2026-08-15T22:10:00",
        "arrival_time": "2026-08-16T04:55:00",
        "duration": "6h 45m",
        "stops": 1,
        "cabin_class": "economy",
        "price": 150.0,
        "currency": "USD",
    },
]


def _tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def _offers_payload():
    return {
        "success": True,
        "data": {
            "IST": {
                "0": {
                    "price": 180.0,
                    "airline": "HY",
                    "flight_number": "701",
                    "origin_airport": "TAS",
                    "destination_airport": "IST",
                    "departure_at": "2026-08-15T07:20:00+00:00",
                    "transfers": 0,
                    "duration": 250,
                },
                "1": {
                    "price": 150.0,
                    "airline": "TK",
                    "flight_number": "372",
                    "origin_airport": "TAS",
                    "destination_airport": "IST",
                    "departure_at": "2026-08-15T22:10:00+00:00",
                    "transfers": 1,
                    "duration": 405,
                },
            }
        },
    }


def test_route_after_agent_with_tool_calls_goes_to_tools():
    state = new_state([HumanMessage("hi"), _tool_call_message("flight_search_tool", {})])
    assert route_after_agent(state) == "tools"


def test_route_after_agent_without_tool_calls_ends():
    state = new_state([HumanMessage("hi"), AIMessage(content="hello there")])
    assert route_after_agent(state) == "__end__"


def test_flight_search_fresh_call_populates_cache_and_hits_api(
    responses, travelpayouts_test_client, mocker
):
    responses.add(responses.GET, PRICES_URL, json=_offers_payload(), status=200)
    mocker.patch("app.tools.flight_search.get_travelpayouts_client", return_value=travelpayouts_test_client)

    state = new_state(
        [_tool_call_message("flight_search_tool", {"origin": "TAS", "destination": "IST", "date": SEARCH_DATE})]
    )
    updates = tools_node(state)

    assert updates["last_search_params"]["origin"] == "TAS"
    assert len(updates["last_search_raw"]) == 2
    assert len(updates["last_search_results"]) == 2
    assert updates["tool_logs"][0]["served_from_cache"] is False
    assert updates["recent_searches"][0]["destination"] == "IST"


def test_flight_search_same_route_reuses_cache_no_api_call(mocker):
    """Same origin/destination/date/pax/cabin as the cached search -> must NOT call the API,
    only re-filter the already-cached raw results (max_stops=0 narrows to the non-stop flight)."""
    guard = mocker.patch(
        "app.tools.flight_search.get_travelpayouts_client",
        side_effect=AssertionError("must not hit the API on a cache-hit refinement"),
    )

    state = new_state([_tool_call_message("flight_search_tool", {
        "origin": "TAS", "destination": "IST", "date": SEARCH_DATE, "max_stops": 0,
    })])
    state["last_search_raw"] = CACHED_FLIGHTS
    state["last_search_params"] = {
        "origin": "TAS", "destination": "IST", "date": SEARCH_DATE,
        "adults": 1, "children": 0, "cabin_class": None,
    }
    state["last_search_results"] = CACHED_FLIGHTS

    updates = tools_node(state)

    guard.assert_not_called()
    assert updates["tool_logs"][0]["served_from_cache"] is True
    assert len(updates["last_search_results"]) == 1
    assert updates["last_search_results"][0]["flight_number"] == "HY701"


def test_flight_search_different_route_forces_new_api_call(
    responses, travelpayouts_test_client, mocker
):
    """Different destination than what's cached -> must hit the API again, not reuse the cache."""
    responses.add(responses.GET, PRICES_URL, json=_offers_payload(), status=200)
    mocker.patch("app.tools.flight_search.get_travelpayouts_client", return_value=travelpayouts_test_client)

    state = new_state(
        [_tool_call_message("flight_search_tool", {"origin": "TAS", "destination": "DXB", "date": SEARCH_DATE})]
    )
    state["last_search_raw"] = CACHED_FLIGHTS
    state["last_search_params"] = {
        "origin": "TAS", "destination": "IST", "date": SEARCH_DATE,
        "adults": 1, "children": 0, "cabin_class": None,
    }
    state["last_search_results"] = CACHED_FLIGHTS

    updates = tools_node(state)

    assert updates["tool_logs"][0]["served_from_cache"] is False
    assert updates["last_search_params"]["destination"] == "DXB"
    # A genuine route change must reset the pool, not accumulate onto the old route's flights.
    assert len(updates["last_search_raw"]) == 2
    assert all(f["arrival_airport"] == "IST" for f in updates["last_search_raw"])


def test_flight_search_same_route_different_date_accumulates_for_comparison(
    responses, travelpayouts_test_client, mocker
):
    """Real bug found via live testing: searching the same route on a second date must ADD to
    the cached pool (so compare_flights_tool can genuinely rank across both real searches),
    not silently replace it and leave compare_flights_tool with only one flight."""
    new_date_payload = {
        "success": True,
        "data": {
            "IST": {
                "price": 82.0,
                "airline": "DP",
                "flight_number": "6821",
                "origin_airport": "TAS",
                "destination_airport": "IST",
                "departure_at": "2026-09-29T08:50:00+05:00",
                "duration_to": 90,
            }
        },
    }
    responses.add(responses.GET, PRICES_URL, json=new_date_payload, status=200)
    mocker.patch("app.tools.flight_search.get_travelpayouts_client", return_value=travelpayouts_test_client)

    state = new_state(
        [_tool_call_message("flight_search_tool", {"origin": "TAS", "destination": "IST", "date": "2026-09-29"})]
    )
    state["last_search_raw"] = CACHED_FLIGHTS
    state["last_search_params"] = {
        "origin": "TAS", "destination": "IST", "date": SEARCH_DATE,
        "adults": 1, "children": 0, "cabin_class": None,
    }
    state["last_search_results"] = CACHED_FLIGHTS

    updates = tools_node(state)

    assert updates["tool_logs"][0]["served_from_cache"] is False
    # Both the original two cached flights AND the newly fetched one must be present.
    flight_numbers = {f["flight_number"] for f in updates["last_search_raw"]}
    assert flight_numbers == {"HY701", "TK372", "DP6821"}
    assert len(updates["last_search_results"]) == 3

    # And compare_flights_tool must now be able to genuinely rank across all three.
    compare_state = new_state([_tool_call_message("compare_flights_tool", {"metric": "price"})])
    compare_state["last_search_results"] = updates["last_search_results"]
    compare_updates = tools_node(compare_state)
    assert compare_updates["last_comparison"]["best_flight"]["flight_number"] == "DP6821"


def test_merge_unique_flights_dedupes_by_flight_number_and_departure_time():
    existing = [CACHED_FLIGHTS[0]]
    # A re-fetch that happens to include the same flight again (same number + departure time)
    # must not create a duplicate row.
    duplicate_of_existing = dict(CACHED_FLIGHTS[0])
    genuinely_new = CACHED_FLIGHTS[1]

    merged = _merge_unique_flights(existing, [duplicate_of_existing, genuinely_new])

    assert len(merged) == 2
    assert {f["flight_number"] for f in merged} == {"HY701", "TK372"}


def test_flight_search_missing_required_field_returns_error_not_exception():
    """No 'date' -> pydantic ValidationError caught and turned into an error ToolMessage, so the
    next agent turn can ask the user a clarifying question instead of the graph crashing."""
    state = new_state([_tool_call_message("flight_search_tool", {"origin": "TAS", "destination": "IST"})])

    updates = tools_node(state)

    assert len(updates["messages"]) == 1
    content = updates["messages"][0].content
    assert "error" in content.lower() or "Invalid" in content


def test_compare_flights_filters_by_airline_and_persists_narrowing():
    state = new_state()
    state["last_search_results"] = CACHED_FLIGHTS
    state["messages"] = [_tool_call_message("compare_flights_tool", {"airlines": ["Turkish"]})]

    updates = tools_node(state)

    assert len(updates["last_search_results"]) == 1
    assert updates["last_search_results"][0]["airline"] == "Turkish Airlines"
    assert updates["active_filters"]["airlines"] == ["Turkish"]
    assert updates["tool_logs"][0]["served_from_cache"] is True


def test_compare_flights_pick_cheapest_ranks_by_price():
    state = new_state()
    state["last_search_results"] = CACHED_FLIGHTS
    state["messages"] = [_tool_call_message("compare_flights_tool", {"metric": "price"})]

    updates = tools_node(state)

    assert updates["last_comparison"]["best_flight"]["flight_number"] == "TK372"


def test_compare_flights_without_prior_search_errors():
    state = new_state([_tool_call_message("compare_flights_tool", {"metric": "price"})])

    updates = tools_node(state)

    content = updates["messages"][0].content
    assert "error" in content.lower()
    assert "last_search_results" not in updates


def test_compare_flights_narrowing_is_sequential_across_turns():
    """Simulates: search -> 'only Uzbekistan Airways' -> 'cheapest' should resolve within the
    already-narrowed set, not the full original cache."""
    state = new_state()
    state["last_search_results"] = CACHED_FLIGHTS
    state["messages"] = [_tool_call_message("compare_flights_tool", {"airlines": ["Uzbekistan"]})]
    updates_1 = tools_node(state)
    assert len(updates_1["last_search_results"]) == 1

    state["last_search_results"] = updates_1["last_search_results"]
    state["active_filters"] = updates_1["active_filters"]
    state["messages"] = [_tool_call_message("compare_flights_tool", {"metric": "price"})]
    updates_2 = tools_node(state)

    assert updates_2["last_comparison"]["best_flight"]["airline"] == "Uzbekistan Airways"


def test_multiple_tool_calls_in_one_turn_all_execute():
    ai_msg = AIMessage(content="", tool_calls=[
        {"name": "compare_flights_tool", "args": {"metric": "price"}, "id": "c1", "type": "tool_call"},
        {"name": "currency_conversion_tool", "args": {"amount": 100, "from_currency": "USD", "to_currency": "USD"},
         "id": "c2", "type": "tool_call"},
    ])
    state = new_state([ai_msg])
    state["last_search_results"] = CACHED_FLIGHTS

    updates = tools_node(state)

    assert len(updates["messages"]) == 2
    assert {m.tool_call_id for m in updates["messages"]} == {"c1", "c2"}
    assert len(updates["tool_logs"]) == 2
