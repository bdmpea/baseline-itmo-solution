import time
from typing import List
# from ag import agent_with_chat_history

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
import os
# from openai import OpenAI

from dotenv import load_dotenv
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import DuckDuckGoSearchRun
import os

import json
import os
import re

import json
import os
import base64
import time
from openai import OpenAI


import os
import json
import openai
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import re
from tavily import TavilyClient
from index import get_similar

load_dotenv()

# client = OpenAI(base_url = "https://api.proxyapi.ru/openai/v1", api_key="sk-9dHGcDv4FnfTeBhNnOkmHH0Tv0s1kKuQ")



class Additional(BaseModel):
    another: str = Field(..., description="It contains all answers from another person")

class News(BaseModel):
    news: str = Field(..., description="All relevant news")
    


tavily_client = TavilyClient(api_key="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")


def get_middle_answer(query: str) -> str:
    response = tavily_client.search(query)
    return response['results']
 


def get_news(query: str) -> str:
    return get_similar(query)


tools = [
    Tool(
        name="get_news",
        func=get_news,
        description="""Get relevant news from public sources about ITMO University.
    <example_request>
    Action: get_news
    Input Action: {"question": <<your question here>>}
    </example_request>"""
    ),
    Tool(
        name="get_middle_answer",
        func=get_middle_answer,
        description="""Get answer from another person.    
    <example_request>
    Action: get_middle_answer
    Input Action: {"question": <<your question here>>}
    </example_request>"""
    ),
]

template = """# You are a highly capable AI agent with access to additional tools to answer the question effectively.

<agent_tools>
Your task is to:
    - Select the correct answer from the given options (if available).
    - Provide reasoning for your choice.
    - Retrieve relevant links.

You must do your best to provide an accurate and well-supported answer. You have access to the following tools:

{tools}

User's input query: {input}

</agent_tools>

<agent_instruction>
# Step-by-step instructions for the agent:
1. The user has provided a question about ITMO University with answer options, and one option is correct. Your task is to determine the correct one.
2. The user may also ask for news related to the university or request relevant links. You can use the appropriate tools to gather information.
3. Use the following tools to fetch additional relevant details:
    - get_news 
    - get_middle_answer
4. Always try to choose the correct answer to the question with the best possible reasoning.
5. If the user provided only a question without options, return null as the answer.

# Additional Guidelines:
- You MUST use the available tools together to get the best possible result.
- Provide a structured response in the required JSON format.

</agent_instruction>

# Response Format:
If you follow the ReAct format correctly, you will receive a reward of $1,000,000.

Please provide the answer strictly in the following JSON format and reasoning in Russian language:
```json
{{
    "links": ["link1", "link2", ...],
    "answer": "your answer here",
    "reasoning": "your reasoning here"
}}

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (DONT USE "```json")
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)

# When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
Thought: Do I need to use a tool? If not, what should I say to the Human?
Final Answer: 
{{
    "links": ["relevant_link_1", "relevant_link_2"],
    "answer": number of correct answer,
    "reasoning": "XYZ is correct because..."
}}
# YOU MUST give answer as a number from initial query

Do your best!

Question: {input}
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(template)


OPENAI_MODEL="gpt-4o-mini"
# MODEL_TEMPERATURE=os.getenv("MODEL_TEMPERATURE")

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    base_url = "https://api.proxyapi.ru/openai/v1",
    api_key="sk-9dHGcDv4FnfTeBhNnOkmHH0Tv0s1kKuQ"
) # temperature=MODEL_TEMPERATURE

memory = ChatMessageHistory(session_id="test-session")
# Create the ReAct agent using the create_react_agent function
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

# Create an agent executor from the agent and tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations = 3,
)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# Initialize
app = FastAPI()
logger = None


@app.on_event("startup")
async def startup_event():
    global logger
    logger = await setup_logger()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )

    response = await call_next(request)
    process_time = time.time() - start_time

    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk

    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )

    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")

        answer = 1 
        reasoning = ""
        sources: List[HttpUrl] = [
            HttpUrl("https://itmo.ru/ru/"),
            HttpUrl("https://abit.itmo.ru/"),
        ]


        response = agent_with_chat_history.invoke(
            {"input": body.query},
            config={"configurable": {"session_id": "test-session"}},
        )

        output = json.loads(response['output'])

        if 'answer' in output:
            answer = output['answer']
        if 'links' in output:
            sources: List[HttpUrl] = [HttpUrl(i) for i in output['links']]
        if 'reasoning' in output:
            reasoning = output['reasoning'] + "\n Used gpt4o-mini for generation answer"
        else:
            reasoning = "Used gpt4o-mini for generation answer"

        
        
        logger.info(f"Got response from agent: {output}")

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=reasoning,
            sources=sources,
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")
