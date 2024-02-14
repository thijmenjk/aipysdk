import json
from asyncio import sleep
from typing import List

import fastapi
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI, AsyncStream
from openai._types import NotGiven
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from pydantic import BaseModel, Field

from aipysdk import async_openai_stream, create_tool_calls_message, create_tool_result_message

app = fastapi.FastAPI()
openai = AsyncOpenAI()


TOOLS: List[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for",
                    }
                },
                "required": ["query"],
            },
            "description": "Search the web for the given query",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_library",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for",
                    }
                },
                "required": ["query"],
            },
            "description": "Search the library for the given query",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "number",
                        "description": "The first number to add",
                    },
                    "b": {
                        "type": "number",
                        "description": "The second number to add",
                    },
                },
                "required": ["a", "b"],
            },
            "description": "Calculate the sum of two numbers",
        },
    },
]


class Request(BaseModel):
    messages: List[ChatCompletionMessageParam] = Field(min_length=1)


@app.post("/api/chat")
async def chat(request: Request):
    messages = request.messages

    async def _chat(messages: List[ChatCompletionMessageParam]) -> AsyncStream[ChatCompletionChunk]:
        print("[OpenAI] chat", messages)
        return await openai.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            stream=True,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto" if TOOLS else NotGiven(),
        )

    async def _execute_tool_calls(
        tool_calls: List[ChoiceDeltaToolCall],
    ):
        messages.append(create_tool_calls_message(tool_calls))

        for tc in tool_calls:
            if tc.function and tc.function.name == "search_web":
                yield {"status": "pending", "message": "Searching the web..."}

                # Simulate a long-running search_web call
                await sleep(2)
                ans = "The meaning of life is 42"

                yield {"status": "done", "result": ans}
                messages.append(create_tool_result_message(tc, ans))
            elif tc.function and tc.function.name == "search_library":
                yield {"status": "pending", "message": "Searching the library..."}

                # Simulate a long-running search_library call
                await sleep(2)
                ans = "The meaning of life is 43"

                yield {"status": "done", "result": ans}
                messages.append(create_tool_result_message(tc, ans))
            elif tc.function and tc.function.name == "calculate_sum":
                if not tc.function.arguments:
                    yield {"status": "error", "message": "No arguments provided"}
                    messages.append(
                        create_tool_result_message(tc, {"status": "error", "message": "No arguments provided"})
                    )
                    continue

                yield {"status": "pending", "message": "Calculating the sum..."}

                # Simulate a long-running calculate_sum call
                await sleep(2)

                args = json.loads(tc.function.arguments)
                ans = args["a"] + args["b"]

                yield {"status": "done", "result": ans}

                messages.append(create_tool_result_message(tc, {"status": "done", "result": ans}))
            else:
                assert False, f"unexpected tool call: {tc}"

        yield await _chat(messages)

    stream = async_openai_stream(
        stream=await _chat(messages),
        execute_tool_calls_cb=_execute_tool_calls,
    )

    return StreamingResponse(
        stream,
        headers={"X-Experimental-Stream-Data": "true"},
    )
