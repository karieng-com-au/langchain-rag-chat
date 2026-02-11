import os
from contextlib import AsyncExitStack

import chainlit as cl
import httpx
from dotenv import load_dotenv
from exa_py import Exa
from httpx import ASGITransport

# from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import StructuredTool, tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from main_api import app

load_dotenv()


@tool
def calculate_area_of_triangle(b: float, h: float) -> float:
    """
    Calculate the area of a triangle.

    Args:
        b: the breadth of a given triangle
        h: the height of a given triangle

    Returns:
        The area of a triangle based on the given breadth and height
    """
    return 0.5 * b * h


@tool
async def get_calories(food_name: str) -> dict[str, str]:
    """
    Retrieve a greeting message from the API server.

    Args:
        food_name: name of the food

    Returns:
        An array
    """
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/chroma-calories", params={"food_name": food_name})

    data = response.json()
    return data

@tool
async def get_answer(question: str) -> dict[str, str]:
    """
    Retrieve answer from the database of the most commonly asked nutrition and health questions and answers.

    Args:
        question: the question from user

    Returns:
        The best answer to the question being asked.
    """
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/qa", params={"food_name": question})

    data = response.json()
    return data

@tool
def search_web(query: str) -> str:
    """
    Search the web for current information.

    Args:
        search_term: A query or question to search on the internet

    Returns:
        Return an array of text results that matches the query
    """
    exa = Exa(os.getenv("EXA_API_KEY"))
    results = exa.search(query=query, type="auto", num_results=10, contents={"text": {"max_characters": 20000}})
    return "\n".join([f"{r.title}: {r.url}" for r in results.results])

@tool
def calculate_bmi(weight: float, height: float) -> float:
    """
    Calculate the Body Mass Index of a person using the person's weight and height

    Args:
        weight: the weight of the person in kg
        height: the height of the person in m

    Returns:
        The value of the person's BMI (Body Mass Index)
    """
    return weight/(height**2)

ddg_search = DuckDuckGoSearchResults(num_results=5)

@cl.on_chat_start
async def on_start():
    llm = ChatAnthropic(
        model=os.environ["ANTHROPIC_MODEL"],
        temperature=0,
        max_retries=3,
        streaming=True,
    )

    # Keep MCP session alive for the duration of the chat
    exit_stack = AsyncExitStack()
    read, write, _ = await exit_stack.enter_async_context(
        streamable_http_client(f"{os.environ['EXA_URL']}?exaApiKey={os.environ['EXA_API_KEY']}")
    )
    session = await exit_stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    mcp_tools = await load_mcp_tools(session)
    cl.user_session.set("exit_stack", exit_stack)

    tools = [calculate_bmi, get_calories, get_answer] + mcp_tools
    calorie_agent_with_search_agent = create_agent(
        llm,
        tools,
        system_prompt="""
        You are a nutrition assistant. Give concise, data-driven answers.

        ## Workflow for calorie queries
        1. Look up the food using get_calories first.
        2. If the food is a meal or the exact item isn't found, use Exa web to find the recipe and ingredients.
        3. Then look up each ingredient individually with get_calories (max 10 calls).
        4. Always prefer get_calories data over web search data for calorie values.

        ## Output format for meals
        - List each ingredient with its estimated quantity and calories for one serving.
        - Show the total calories at the end.

        ## Rules
        - Only return get_calories results that match the requested food — ignore unrelated matches.
        - If no data is found after searching, say so rather than guessing.
        """,
        name="calorie_agent_with_search"
    )

    healthy_breakfast_planner_agent = create_agent(
        llm,
        system_prompt="""
        * You are a helpful assistant that helps with healthy breakfast choices.
        * You give concise answers.
        Given the user's preferences prompt, come up with different vegetarian (not vegan) breakfast meals that are healthy and fit for a busy person.
        * Explicitly mention the meal's names in your response along with a short summary of why this is a healthy choice.
        """,
        name="healthy_breakfast_planner"
    )

    breakfast_price_checker_agent = create_agent(
        llm,
        [ddg_search],
        system_prompt="""
        You receive a list of breakfast meals with their ingredients and calorie information.
        Your job is to search for the current approximate price of each ingredient.

        ## Rules
        - Do NOT ask the user for more information. Work with whatever meal data you receive.
        - Use the duckduckgo_results_json tool to search for grocery prices (e.g. "eggs price per dozen 2025").
        - Search for each unique ingredient, not every meal separately.

        ## Output format (use markdown)
        For each meal, list:
        - Meal name
        - Each ingredient with its calories and approximate price
        - Estimated total cost per serving
        """,
        name="breakfast_price_checker"
    )

    async def _calorie_calculator(query: str) -> str:
        result = await calorie_agent_with_search_agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return result["messages"][-1].content

    calorie_agent_with_search_tool = StructuredTool.from_function(
        coroutine=_calorie_calculator,
        name="calorie_calculator",
        description="Calculate the calories of a meal and its ingredients.",
    )

    async def _breakfast_planner(query: str) -> str:
        result = await healthy_breakfast_planner_agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
        return result["messages"][-1].content

    healthy_breakfast_planner_tool = StructuredTool.from_function(
        coroutine=_breakfast_planner,
        name="breakfast_planner",
        description="Plan healthy breakfast options based on user preferences.",
    )

    breakfast_advisor_agent = create_agent(
        llm,
        tools=[healthy_breakfast_planner_tool, calorie_agent_with_search_tool],
        system_prompt="""
        You are a breakfast advisor that creates meal plans with calories.

        Follow this workflow:
        1. Use breakfast_planner to generate healthy breakfast options based on user preferences.
        2. Use calorie_calculator to look up calories for each meal and its ingredients.

        In your final output, list each meal with its name, ingredients, and calories.
        Do NOT mention pricing, costs, or that you cannot look up prices.
        Another agent will handle pricing — just focus on meals and calories.
        """,
        name="breakfast_advisor",
    )

    cl.user_session.set("breakfast_advisor", breakfast_advisor_agent)
    cl.user_session.set("price_checker", breakfast_price_checker_agent)
    await cl.Message(content=f"Welcome to Nutrition").send()


async def stream_agent(agent, input_text, msg):
    """Stream an agent's response to a Chainlit message and return the full text."""
    tool_steps = {}
    full_output = ""

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": input_text}]},
        config={},
        version="v2",
    ):
        if event["event"] == "on_tool_start":
            run_id = event["run_id"]
            step = cl.Step(name=event["name"], type="tool")
            step.input = str(event["data"].get("input", ""))
            await step.send()
            tool_steps[run_id] = step

        elif event["event"] == "on_tool_end":
            run_id = event["run_id"]
            if run_id in tool_steps:
                step = tool_steps.pop(run_id)
                step.output = str(event["data"].get("output", ""))
                await step.update()

        elif event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if isinstance(content, str) and content:
                await msg.stream_token(content)
                full_output += content
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        await msg.stream_token(block["text"])
                        full_output += block["text"]

    await msg.update()
    return full_output


@cl.on_message
async def main(message: cl.Message):
    breakfast_advisor = cl.user_session.get("breakfast_advisor")
    price_checker = cl.user_session.get("price_checker")

    # Step 1: Breakfast advisor plans meals with calories
    # Use ainvoke to get clean final output (streaming captures all sub-agent text)
    msg1 = cl.Message(content="Planning meals and looking up calories...")
    await msg1.send()
    result = await breakfast_advisor.ainvoke(
        {"messages": [{"role": "user", "content": message.content}]}
    )
    raw_content = result["messages"][-1].content
    if isinstance(raw_content, list):
        advisor_output = "".join(
            block["text"] for block in raw_content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    else:
        advisor_output = raw_content
    msg1.content = advisor_output
    await msg1.update()

    # Step 2: Hand off to price checker with explicit instruction
    price_input = (
        "Search for the current grocery prices of each ingredient in these meals "
        "and produce the final output with meal names, ingredients, calories, and prices:\n\n"
        + advisor_output
    )
    msg2 = cl.Message(content="")
    await msg2.send()
    await stream_agent(price_checker, price_input, msg2)


@cl.on_chat_end
async def on_end():
    exit_stack = cl.user_session.get("exit_stack")
    if exit_stack:
        try:
            await exit_stack.aclose()
        except RuntimeError:
            pass


if __name__ == "__main__":
    main()

