
import os
import json
from typing import Any

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


# ---------- 1. Agent 1: Destination Search Tool ----------

@tool
def search_destinations(
    region: str,
    interests: str,
    budget_inr: float
) -> str:
    """Find destinations based on region, interests and budget."""

    destinations = [
        {
            "name": "Hyderabad",
            "attractions": ["Charminar", "Golconda Fort", "Hussain Sagar"],
            "interests": "history culture food",
            "daily_cost": 1800
        },
        {
            "name": "Jaipur",
            "attractions": ["Amber Fort", "Hawa Mahal", "City Palace"],
            "interests": "history culture architecture photography",
            "daily_cost": 2500
        },
        {
            "name": "Goa",
            "attractions": ["Calangute Beach", "Fort Aguada", "Fontainhas"],
            "interests": "beach nature adventure food",
            "daily_cost": 3000
        },
        {
            "name": "Mysuru",
            "attractions": ["Mysore Palace", "Chamundi Hill", "Brindavan Gardens"],
            "interests": "history culture nature family",
            "daily_cost": 2000
        },
        {
            "name": "Delhi",
            "attractions": ["Red Fort", "India Gate", "National Museum"],
            "interests": "history culture museums food",
            "daily_cost": 2300
        }
    ]

    region = region.lower().strip()
    interest_words = interests.lower().replace(",", " ").split()

    results = []

    for place in destinations:

        region_match = (
            region in ["india", "any", "anywhere"]
            or region in place["name"].lower()
        )

        interest_match = any(
            word in place["interests"].split()
            for word in interest_words
        )

        if region_match or interest_match:
            results.append({
                "destination": place["name"],
                "attractions": place["attractions"],
                "estimated_daily_cost_inr": place["daily_cost"],
                "budget_status": (
                    "May fit budget"
                    if budget_inr >= place["daily_cost"]
                    else "Budget may be tight"
                )
            })

    if not results:
        results = [
            {
                "destination": p["name"],
                "attractions": p["attractions"],
                "estimated_daily_cost_inr": p["daily_cost"],
                "budget_status": (
                    "May fit budget"
                    if budget_inr >= p["daily_cost"]
                    else "Budget may be tight"
                )
            }
            for p in destinations
        ]

    return json.dumps({
        "note": "Demo suggestions, not live travel prices.",
        "destinations": results
    })


# ---------- 2. Agent 2: Itinerary Tool ----------

@tool
def build_itinerary(
    destination: str,
    days: int,
    budget_inr: float,
    travelers: int,
    interests: str
) -> str:
    """Build a day-wise itinerary and calculate estimated costs."""

    if days < 1 or days > 30:
        return "Trip duration must be between 1 and 30 days."

    if travelers < 1 or budget_inr <= 0:
        return "Provide valid travelers and budget."

    attractions = {
        "hyderabad": ["Charminar", "Golconda Fort", "Hussain Sagar"],
        "jaipur": ["Amber Fort", "Hawa Mahal", "City Palace"],
        "goa": ["Calangute Beach", "Fort Aguada", "Fontainhas"],
        "mysuru": ["Mysore Palace", "Chamundi Hill", "Brindavan Gardens"],
        "delhi": ["Red Fort", "India Gate", "National Museum"]
    }

    daily_costs = {
        "hyderabad": 1800,
        "jaipur": 2500,
        "goa": 3000,
        "mysuru": 2000,
        "delhi": 2300
    }

    key = destination.lower()

    places = next(
        (v for k, v in attractions.items() if k in key),
        ["Explore the city", "Visit a local attraction", "Try local food"]
    )

    daily_cost = next(
        (v for k, v in daily_costs.items() if k in key),
        2200
    )

    itinerary = []

    for day in range(days):
        itinerary.append({
            "day": day + 1,
            "activities": [
                places[(day * 2) % len(places)],
                places[(day * 2 + 1) % len(places)]
            ]
        })

    estimated_cost = daily_cost * days * travelers

    return json.dumps({
        "destination": destination,
        "days": days,
        "travelers": travelers,
        "budget_inr": budget_inr,
        "estimated_local_cost_inr": estimated_cost,
        "budget_status": (
            "Within estimated budget"
            if estimated_cost <= budget_inr
            else "Over budget"
        ),
        "itinerary": itinerary,
        "note": (
            "Demo estimate only. Travel, accommodation and live prices "
            "may not be included."
        )
    })


# ---------- 3. Gemini Configuration ----------

GOOGLE_API_KEY = (
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GOOGLE_APIKEY")
)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Please configure GOOGLE_API_KEY in Render Environment."
    )

GOOGLE_MODEL = "gemini-3.8-flash"

llm = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_retries=2
)

print("CURRENT GEMINI MODEL:", GOOGLE_MODEL)


# ---------- 4. Create Two AI Agents ----------

destination_agent = create_agent(
    model=llm,
    tools=[search_destinations],
    system_prompt=(
        "You are Agent 1, the Destination Discovery Agent. "
        "Find destinations based on region, interests and budget. "
        "Always use the search_destinations tool. "
        "Explain why destinations suit the traveler. "
        "Never invent live prices."
    )
)

itinerary_agent = create_agent(
    model=llm,
    tools=[build_itinerary],
    system_prompt=(
        "You are Agent 2, the Itinerary and Budget Agent. "
        "Create a day-wise travel itinerary. "
        "Use the build_itinerary tool when all required details "
        "are available. Required details are destination, days, "
        "budget and travelers. Ask for missing information. "
        "Clearly explain estimated costs. Never claim live prices."
    )
)


# ---------- 5. Input Schema ----------

class AgentInput(BaseModel):
    input: str = Field(
        description=(
            "Enter destination or region, days, budget in INR, "
            "number of travelers and interests."
        )
    )


def format_for_agent(x: Any) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}


# ---------- 6. Extract AI Response ----------

def extract_response(output: Any) -> str:

    if not isinstance(output, dict):
        return str(output)

    messages = output.get("messages", [])

    for message in reversed(messages):

        if getattr(message, "type", "") != "ai":
            continue

        content = getattr(message, "content", "")

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))

                elif isinstance(item, str):
                    text_parts.append(item)

            result = "\n".join(text_parts).strip()

            if result:
                return result

    return "No final AI response was generated. Check Render logs."


# ---------- 7. Create Agent Chains ----------

def make_chain(agent):

    return (
        RunnableLambda(format_for_agent)
        | agent
        | RunnableLambda(extract_response)
    ).with_types(
        input_type=AgentInput,
        output_type=str
    )


destination_chain = make_chain(destination_agent)

itinerary_chain = make_chain(itinerary_agent)


# ---------- 8. Run Both Agents ----------

def run_both_agents(x: Any) -> str:

    user_input = x["input"] if isinstance(x, dict) else x.input

    destination_result = destination_chain.invoke({
        "input": user_input
    })

    itinerary_input = (
        "Create a travel itinerary and check the budget.\n\n"
        "User travel request:\n"
        f"{user_input}\n\n"
        "Destination recommendations from Agent 1:\n"
        f"{destination_result}\n\n"
        "Use the user's travel details. "
        "Ask for missing essential information instead of guessing."
    )

    itinerary_result = itinerary_chain.invoke({
        "input": itinerary_input
    })

    return (
        "=== AGENT 1: DESTINATION DISCOVERY ===\n\n"
        + destination_result
        + "\n\n"
        "=== AGENT 2: ITINERARY AND BUDGET CHECK ===\n\n"
        + itinerary_result
    )


combined_chain = (
    RunnableLambda(format_for_agent)
    | RunnableLambda(
        lambda state: {"input": state["messages"][0][1]}
    )
    | RunnableLambda(run_both_agents)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# ---------- 9. FastAPI Application ----------

app = FastAPI(
    title="Smart Travel Planning Agent API",
    description=(
        "Two AI agents for destination discovery, "
        "itinerary creation and budget checking."
    ),
    version="1.0.0"
)


add_routes(
    app,
    destination_chain,
    path="/destination-agent",
    playground_type="default"
)

add_routes(
    app,
    itinerary_chain,
    path="/itinerary-agent",
    playground_type="default"
)

add_routes(
    app,
    combined_chain,
    path="/agent",
    playground_type="default"
)


@app.get("/")
def home():

    return {
        "message": "Smart Travel Planning Agent API is running",
        "destination_agent": "/destination-agent/playground/",
        "itinerary_agent": "/itinerary-agent/playground/",
        "combined_agent": "/agent/playground/",
        "docs": "/docs"
    }


# ---------- 10. Start Server ----------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
