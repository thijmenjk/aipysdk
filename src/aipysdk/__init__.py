import enum
import json
from typing import (Any, AsyncIterator, Callable, Iterator, List, Optional,
                    Union)

from openai._streaming import AsyncStream, Stream
from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionChunk,
                               ChatCompletionMessageToolCallParam,
                               ChatCompletionToolMessageParam)
from openai.types.chat.chat_completion_chunk import Choice, ChoiceDeltaToolCall
from pydantic import BaseModel

CallbackReturnType = Iterator[Union[Any, Stream[ChatCompletionChunk]]]
AsyncCallbackReturnType = AsyncIterator[Union[Any, AsyncStream[ChatCompletionChunk]]]


class ResponseChannel(enum.Enum):
    TEXT = 0
    DATA = 2


class ToolCallReceivedCbParams(BaseModel):
    index: int
    id: str
    function_name: str


def create_tool_calls_message(
    tool_calls: List[ChoiceDeltaToolCall],
) -> ChatCompletionAssistantMessageParam:
    tcs: List[ChatCompletionMessageToolCallParam] = [
        {
            "id": t.id,
            "type": "function",
            "function": {
                "name": t.function.name,
                "arguments": json.dumps(t.function.arguments),
            },
        }
        for t in tool_calls
        if t.function and t.id and t.function.name
    ]
    assert tcs, "did not receive the expected parameters for a tool calls message"

    return {"role": "assistant", "content": "", "tool_calls": tcs}


def create_tool_result_message(tc: ChoiceDeltaToolCall, result: Any) -> ChatCompletionToolMessageParam:
    assert tc.id and result is not None, "did not receive the expected parameters for a tool call message"
    return {
        "role": "tool",
        "tool_call_id": tc.id,
        "content": json.dumps(result),
    }


def vercel_ai_sdk_format(channel: ResponseChannel, content: Any) -> str:
    return f"{channel.value}:{json.dumps(content)}\n"


def human_readable_format(channel: ResponseChannel, content: Any) -> str:
    if channel == ResponseChannel.TEXT:
        return content
    else:
        return f"{json.dumps(content)}\n"


def _merge_diff_tool_calls(
    base: ChoiceDeltaToolCall,
    delta: ChoiceDeltaToolCall,
):
    assert (
        base.function and isinstance(base.function.arguments, str) and delta.function and delta.function.arguments
    ), "did not receive the expected parameters for a tool call diff"

    base.function.arguments += delta.function.arguments


async def async_openai_stream(
    stream: AsyncStream[ChatCompletionChunk],
    pick: Callable[[List[Choice]], Choice] = lambda x: x[0],
    receiving_tool_call_cb: Optional[Callable[[ToolCallReceivedCbParams], AsyncCallbackReturnType]] = None,
    execute_tool_calls_cb: Optional[Callable[[List[ChoiceDeltaToolCall]], AsyncCallbackReturnType]] = None,
    format: Callable[[ResponseChannel, Any], str] = vercel_ai_sdk_format,
    max_recursion_depth: int = 5,
    _depth: int = 0,
) -> AsyncIterator[str]:
    async def _process_cb_result(result: AsyncCallbackReturnType) -> AsyncIterator[str]:
        async for elem in result:
            if isinstance(elem, AsyncStream):
                casted: AsyncStream[ChatCompletionChunk] = elem

                async for chunk in async_openai_stream(
                    casted,
                    pick,
                    receiving_tool_call_cb,
                    execute_tool_calls_cb,
                    format,
                    max_recursion_depth,
                    _depth + 1,
                ):
                    yield chunk
            else:
                yield format(ResponseChannel.DATA, elem if isinstance(elem, list) else [elem])

    if _depth >= max_recursion_depth:
        raise RecursionError("maximum recursion depth exceeded")

    tool_calls: List[Optional[ChoiceDeltaToolCall]] = []
    finish_reason = None
    async for chunk in stream:
        choice = pick(chunk.choices)

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        delta = choice.delta
        if not delta:
            break

        if c := delta.content:
            yield format(ResponseChannel.TEXT, c)
        elif delta.tool_calls:
            for dtc in delta.tool_calls:
                idx = dtc.index
                while len(tool_calls) <= idx:
                    tool_calls.append(None)

                if (tc := tool_calls[idx]) is not None:
                    _merge_diff_tool_calls(tc, dtc)
                else:
                    tool_calls[idx] = dtc

                    if receiving_tool_call_cb:
                        assert (
                            dtc.id and dtc.function and dtc.function.name
                        ), "did not receive the expected parameters for initial tool call"

                        async for chunk in _process_cb_result(
                            receiving_tool_call_cb(
                                ToolCallReceivedCbParams(
                                    index=idx,
                                    id=dtc.id,
                                    function_name=dtc.function.name,
                                )
                            )
                        ):
                            yield chunk

    if finish_reason == "tool_calls":
        assert execute_tool_calls_cb, "tool calls were received but no callback was set"

        async for chunk in _process_cb_result(execute_tool_calls_cb([tc for tc in tool_calls if tc])):
            yield chunk


def openai_stream(
    stream: Stream[ChatCompletionChunk],
    pick: Callable[[List[Choice]], Choice] = lambda x: x[0],
    receiving_tool_call_cb: Optional[Callable[[ToolCallReceivedCbParams], CallbackReturnType]] = None,
    execute_tool_calls_cb: Optional[Callable[[List[ChoiceDeltaToolCall]], CallbackReturnType]] = None,
    format: Callable[[ResponseChannel, Any], str] = vercel_ai_sdk_format,
    max_recursion_depth: int = 5,
    _depth: int = 0,
) -> Iterator[str]:
    def _process_cb_result(result: CallbackReturnType) -> Iterator[str]:
        for elem in result:
            if isinstance(elem, Stream):
                casted: Stream[ChatCompletionChunk] = elem

                yield from openai_stream(
                    casted,
                    pick,
                    receiving_tool_call_cb,
                    execute_tool_calls_cb,
                    format,
                    max_recursion_depth,
                    _depth + 1,
                )
            else:
                yield format(ResponseChannel.DATA, elem if isinstance(elem, list) else [elem])

    if _depth >= max_recursion_depth:
        raise RecursionError("maximum recursion depth exceeded")

    tool_calls: List[Optional[ChoiceDeltaToolCall]] = []
    finish_reason = None
    for chunk in stream:
        choice = pick(chunk.choices)

        if choice.finish_reason:
            finish_reason = choice.finish_reason

        delta = choice.delta
        if not delta:
            break

        if c := delta.content:
            yield format(ResponseChannel.TEXT, c)
        elif delta.tool_calls:
            for dtc in delta.tool_calls:
                idx = dtc.index
                while len(tool_calls) <= idx:
                    tool_calls.append(None)

                if (tc := tool_calls[idx]) is not None:
                    _merge_diff_tool_calls(tc, dtc)
                else:
                    tool_calls[idx] = dtc

                    if receiving_tool_call_cb:
                        assert (
                            dtc.id and dtc.function and dtc.function.name
                        ), "did not receive the expected parameters for initial tool call"

                        yield from _process_cb_result(
                            receiving_tool_call_cb(
                                ToolCallReceivedCbParams(
                                    index=idx,
                                    id=dtc.id,
                                    function_name=dtc.function.name,
                                )
                            )
                        )

    if finish_reason == "tool_calls":
        assert execute_tool_calls_cb, "tool calls were received but no callback was set"

        yield from _process_cb_result(execute_tool_calls_cb([tc for tc in tool_calls if tc]))
