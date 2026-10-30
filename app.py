
import os
import json

import uvicorn
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. DESTINATION DATA
# --------------------------------------------------

DESTINATIONS = {
    "Hyderabad": {
        "region": "South India",
        "interests": ["history", "food", "culture"],
        "attractions": [
            "Charminar",
            "Golconda Fort",
            "Hussain Sagar Lake",
            "Ramoji Film City"
        ],
        "daily_cost": 1800
    },

    "Jaipur": {
        "region": "North India",
        "interests": ["history", "culture", "architecture"],
        "attractions": [
            "Amber Fort",
            "Hawa Mahal",
            "City Palace",
            "Jantar Mantar"
        ],
        "daily_cost": 2500
    },

    "Goa": {
        "region": "West India",
        "interests": ["beaches", "adventure", "nature"],
        "attractions": [
            "Baga Beach",
            "Calangute Beach",
            "Fort Aguada",
            "Dudhsagar Falls"
        ],
        "daily_cost": 3000
    },

    "Mysuru": {
        "region": "South India",
        "interests": ["history", "culture", "nature"],
        "attractions": [
            "Mysore Palace",
            "Chamundi Hills",
            "Brindavan Gardens",
            "St. Philomena's Church"
        ],
        "daily_cost": 1600
    },

    "Delhi": {
        "region": "North India",
        "interests": ["history", "food", "culture"],
        "attractions": [
            "India Gate",
            "Red Fort",
            "Qutub Minar",
            "Lotus Temple"
        ],
        "daily_cost": 2200
    }
}


# --------------------------------------------------
# 2. DEFINE TOOLS
# --------------------------------------------------

@tool
def search_destinations(
    region: str,
    interests: str,
    budget_inr: float
) -> str:
    """
    Search Indian travel destinations based on region,
    interests and total available budget in INR.
    """

    results = []

    requested_region = region.lower()

    requested_interests = [
        item.strip().lower()
        for item in interests.split(",")
    ]

    for city, details in DESTINATIONS.items():

        region_match = (
            requested_region in ["any", "india", "all"]
            or requested_region in details["region"].lower()
        )

        interest_match = (
            "any" in requested_interests
            or any(
                interest in details["interests"]
                for interest in requested_interests
            )
        )

        # Daily estimate for one person
        budget_match = details["daily_cost"] <= budget_inr

        if region_match and interest_match and budget_match:

            results.append({
                "destination": city,
                "region": details["region"],
                "interests": details["interests"],
                "attractions": details["attractions"],
                "estimated_daily_cost_per_person_inr":
                    details["daily_cost"]
            })

    if not results:
        return json.dumps({
            "message": (
                "No matching destinations found. "
                "Try changing your region, interests or budget."
            )
        })

    return json.dumps(results, indent=2)


@tool
def build_itinerary(
    destination: str,
    days: int,
    budget_inr: float,
    travelers: int,
    interests: str
) -> str:
    """
    Create a day-by-day travel itinerary based on
    destination, duration, budget, travelers and interests.
    """

    if days < 1 or days > 30:
        return "Please enter a trip duration between 1 and 30 days."

    if travelers < 1:
        return "Number of travelers must be at least 1."

    city = destination.strip().title()

    if city not in DESTINATIONS:
        return (
            "Destination not found in the available database. "
            "Choose Hyderabad, Jaipur, Goa, Mysuru or Delhi."
        )

    details = DESTINATIONS[city]

    attractions = details["attractions"]

    daily_cost = details["daily_cost"]

    estimated_cost = daily_cost * days * travelers

    itinerary = []

    for day in range(days):

        first = attractions[(day * 2) % len(attractions)]

        second = attractions[
            (day * 2 + 1) % len(attractions)
        ]

        itinerary.append({
            "day": day + 1,
            "morning": f"Visit {first}",
            "afternoon": f"Explore {second}",
            "evening": "Enjoy local food and explore the city"
        })

    result = {
        "destination": city,
        "duration_days": days,
        "number_of_travelers": travelers,
        "interests": interests,
        "estimated_daily_cost_per_person_inr": daily_cost,
        "estimated_total_cost_inr": estimated_cost,
        "user_budget_inr": budget_inr,
        "budget_status": (
            "Within budget"
            if estimated_cost <= budget_inr
            else "Over budget"
        ),
        "itinerary": itinerary
    }

    return json.dumps(result, indent=2)


tools = [
    search_destinations,
    build_itinerary
]


# --------------------------------------------------
# 3. INITIALIZE GEMMA MODEL
# --------------------------------------------------

GOOGLE_API_KEY = (
    os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("GOOGLE_APIKEY")
)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Please configure GOOGLE_API_KEY in Render Environment."
    )

GOOGLE_MODEL = "gemma-4-31b-it"

llm = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

print("CURRENT MODEL:", GOOGLE_MODEL)


# --------------------------------------------------
# 4. CREATE TWO AGENTS
# --------------------------------------------------

# Agent 1: Destination Search Agent

destination_agent = create_agent(
    model=llm,
    tools=[search_destinations],
    system_prompt=(
        "You are a Destination Search Agent for Indian travel. "
        "Find suitable destinations using the search_destinations tool. "
        "Consider the user's region, interests and budget. "
        "Present the recommended destinations clearly."
    )
)


# Agent 2: Itinerary Planning Agent

itinerary_agent = create_agent(
    model=llm,
    tools=[build_itinerary],
    system_prompt=(
        "You are a Travel Itinerary Planning Agent. "
        "Create a day-by-day travel plan using the build_itinerary tool. "
        "Consider the destination, number of days, budget, travelers "
        "and interests. Clearly explain the estimated cost and budget status. "
        "Do not invent missing travel details. Ask the user if important "
        "information is missing."
    )
)


# --------------------------------------------------
# 5. INPUT AND OUTPUT FORMATTING
# --------------------------------------------------

class AgentInput(BaseModel):

    input: str = Field(
        description="Enter your travel requirements"
    )


def format_for_agent(x) -> dict:

    user_input = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    return {
        "messages": [
            ("user", user_input)
        ]
    }


def extract_text_response(agent_output) -> str:

    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:

        for value in agent_output.values():

            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:

        last_message = messages[-1]

        content = getattr(
            last_message,
            "content",
            last_message
        )

        if isinstance(content, list):

            return "\n".join(
                str(item.get("text", item))
                if isinstance(item, dict)
                else str(item)
                for item in content
            )

        return str(content)

    return str(agent_output)


# --------------------------------------------------
# 6. CREATE AGENT CHAINS
# --------------------------------------------------

destination_chain = (
    RunnableLambda(format_for_agent)
    | destination_agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


itinerary_chain = (
    RunnableLambda(format_for_agent)
    | itinerary_agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# --------------------------------------------------
# 7. COMBINED TRAVEL AGENT
# --------------------------------------------------

def create_complete_travel_plan(x):

    user_input = (
        x["input"]
        if isinstance(x, dict)
        else x.input
    )

    # First agent searches for destinations

    destination_result = destination_chain.invoke({
        "input": user_input
    })

    # Second agent creates the itinerary

    itinerary_input = (
        "Original travel requirements:\n"
        + user_input
        + "\n\nDestination recommendations from Agent 1:\n"
        + destination_result
        + "\n\nUsing the recommendations above, create a suitable "
        "travel itinerary using the build_itinerary tool. "
        "Use only a destination returned by Agent 1. "
        "If the user has not provided the number of days, "
        "budget or travelers, ask for the missing information "
        "instead of inventing it."
    )

    itinerary_result = itinerary_chain.invoke({
        "input": itinerary_input
    })

    return (
        "DESTINATION RECOMMENDATIONS:\n\n"
        + destination_result
        + "\n\nTRAVEL ITINERARY:\n\n"
        + itinerary_result
    )


combined_chain = (
    RunnableLambda(create_complete_travel_plan)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# --------------------------------------------------
# 8. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Smart Travel Planning Agent API",
    description=(
        "AI travel planning application using Gemma 4 "
        "and two agents: Destination Search Agent "
        "and Itinerary Planning Agent."
    ),
    version="1.0.0"
)


# Destination Agent Route

add_routes(
    app,
    destination_chain,
    path="/destination-agent",
    playground_type="default"
)


# Itinerary Agent Route

add_routes(
    app,
    itinerary_chain,
    path="/itinerary-agent",
    playground_type="default"
)


# Combined Agent Route

add_routes(
    app,
    combined_chain,
    path="/agent",
    playground_type="default"
)


# --------------------------------------------------
# 9. HOME PAGE
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to Smart Travel Planning Agent!",
        "model": GOOGLE_MODEL,
        "destination_agent": "/destination-agent/playground/",
        "itinerary_agent": "/itinerary-agent/playground/",
        "combined_agent": "/agent/playground/",
        "api_docs": "/docs"
    }


# --------------------------------------------------
# 10. RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
