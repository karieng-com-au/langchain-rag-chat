import os

import chainlit as cl
import httpx
from dotenv import load_dotenv
from httpx import ASGITransport

# from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool

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

@cl.on_chat_start
async def on_start():
    llm = ChatAnthropic(
        model=os.environ["ANTHROPIC_MODEL"],
        temperature=0,
        max_retries=3,
        streaming=True,
    )
    tools = [calculate_bmi, get_calories, get_answer]
    agent = create_agent(
        llm,
        tools,
        system_prompt="You are a nutrition assistant. Use the tools provided to look up calorie and nutritional information. Give clear, concise answers based on the data returned. If a food item is not found, say so rather than guessing.",
    )
    cl.user_session.set("agent", agent)
    await cl.Message(content=f"Welcome to Karieng").send()


@cl.on_message
async def main(message: cl.Message):

    agent = cl.user_session.get("agent")

    msg = cl.Message(content="")
    await msg.send()

    tool_steps = {}

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": message.content}]},
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
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        await msg.stream_token(block["text"])

    await msg.update()


if __name__ == "__main__":
    main()

