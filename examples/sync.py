#!/usr/bin/env python
import json
from time import sleep
from typing import List

from openai import OpenAI
from openai._streaming import Stream
from openai._types import NotGiven
from openai.types.chat import ChatCompletionChunk, ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall

from aipysdk import create_tool_calls_message, create_tool_result_message, human_readable_format, openai_stream

openai = OpenAI()

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


def chat(messages: List[ChatCompletionMessageParam]) -> Stream[ChatCompletionChunk]:
    print("[OpenAI] chat", messages)
    return openai.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        stream=True,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto" if TOOLS else NotGiven(),
    )


messages: List[ChatCompletionMessageParam] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": "Can you calculate 40+2 using the calculator and tell me how it relates to the meaning of life by searching the web and the library?",
    },
]


def execute_tool_calls(
    tool_calls: List[ChoiceDeltaToolCall],
):
    messages.append(create_tool_calls_message(tool_calls))

    for tc in tool_calls:
        if tc.function and tc.function.name == "search_web":
            yield {"status": "pending", "message": "Searching the web..."}

            # Simulate a long-running search_web call
            sleep(2)
            ans = "The meaning of life is 42"

            yield {"status": "done", "result": ans}
            messages.append(create_tool_result_message(tc, ans))
        elif tc.function and tc.function.name == "search_library":
            yield {"status": "pending", "message": "Searching the library..."}

            # Simulate a long-running search_library call
            sleep(2)
            ans = "The meaning of life is 43"

            yield {"status": "done", "result": ans}
            messages.append(create_tool_result_message(tc, ans))
        elif tc.function and tc.function.name == "calculate_sum":
            if not tc.function.arguments:
                yield {"status": "error", "message": "No arguments provided"}
                messages.append(create_tool_result_message(tc, {"status": "error", "message": "No arguments provided"}))
                continue

            yield {"status": "pending", "message": "Calculating the sum..."}

            # Simulate a long-running calculate_sum call
            sleep(2)

            args = json.loads(tc.function.arguments)
            ans = args["a"] + args["b"]

            yield {"status": "done", "result": ans}

            messages.append(create_tool_result_message(tc, {"status": "done", "result": ans}))
        else:
            assert False, f"unexpected tool call: {tc}"

    yield chat(messages)


if __name__ == "__main__":
    for chunk in openai_stream(
        stream=chat(messages), execute_tool_calls_cb=execute_tool_calls, format=human_readable_format
    ):
        print(chunk, end="", flush=True)
    print()
