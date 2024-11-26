from __future__ import annotations
from typing import Any, Iterable, Dict, List, Optional, Union, Callable, Type, TYPE_CHECKING
import re

from pydantic import BaseModel
import ujson

if TYPE_CHECKING:
    from .types import (
        PromptElement,
        ImageProps,
        OutputHandler,
        ChatCompletionResponseMessage,
        StreamChatCompletionResponse,
    )


def SystemMessage(
    children: Optional[Union[List[PromptElement], PromptElement]] = None,
    name: Optional[str] = None,
    to: Optional[str] = None,
) -> PromptElement:
    return {
        "type": "chat",
        "role": "system",
        "name": name,
        "to": to,
        "children": to_children(children),
    }


def UserMessage(
    children: Optional[Union[List[PromptElement], PromptElement]] = None,
    name: Optional[str] = None,
    to: Optional[str] = None,
) -> PromptElement:
    return {
        "type": "chat",
        "role": "user",
        "name": name,
        "to": to,
        "children": to_children(children),
    }


def AssistantMessage(
    children: Optional[Union[List[PromptElement], PromptElement]] = None,
    function_call: Optional[Dict[str, str]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    to: Optional[str] = None,
) -> PromptElement:
    message: Dict[str, Any] = {
        "type": "chat",
        "role": "assistant",
        "to": to,
        "children": to_children(children),
    }
    if function_call:
        message["function_call"] = function_call
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def ImageComponent(props: ImageProps) -> PromptElement:
    return {
        "type": "image",
        "bytes": props["bytes"],
        "dimensions": props["dimensions"],
        "detail": props["detail"],
    }


def FunctionMessage(
    name: str,
    children: Optional[Union[List[PromptElement], PromptElement]] = None,
    to: Optional[str] = None,
) -> PromptElement:
    return {
        "type": "chat",
        "role": "function",
        "name": name,
        "to": to,
        "children": to_children(children),
    }


def ToolResultMessage(
    name: str,
    children: Optional[Union[List[PromptElement], PromptElement]] = None,
    to: Optional[str] = None,
) -> PromptElement:
    return {
        "type": "chat",
        "role": "tool",
        "name": name,
        "to": to,
        "children": to_children(children),
    }


def to_children(
    children: Optional[Union[List[PromptElement], PromptElement]],
) -> List[PromptElement]:
    if children is None:
        return []
    if isinstance(children, list):
        return children
    return [children]


def populate_on_stream_response_object_from_on_stream(
    capture_props: Dict[str, Any],
) -> Dict[str, Any]:
    def on_stream_response_object(stream: Iterable[StreamChatCompletionResponse]):
        def message_stream():
            for s in stream:
                if not s["choices"]:
                    continue
                if "delta" not in s["choices"][0]:
                    continue
                yield s["choices"][0]["delta"]

        capture_props["on_stream"](message_stream())

    return {
        **capture_props,
        "on_stream_response_object": on_stream_response_object,
    }


def Tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
) -> PromptElement:
    return {
        "type": "toolDefinition",
        "tool": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
    }


def Tools(
    tools: List[Dict[str, Any]],
    on_return: OutputHandler[Iterable[str]],
) -> List[PromptElement]:
    tool_elements = [
        Tool(
            name=tool["name"],
            description=tool["description"],
            parameters=tool["parameters"],
        )
        for tool in tools
    ]

    def handle_stream(stream: Iterable[ChatCompletionResponseMessage]):
        def generate_output():
            tool_calls_map: Dict[int, Dict[str, str]] = {}

            for message in stream:
                if message.get("content") is not None:
                    yield message["content"]

                tool_calls = message.get("tool_calls", [])
                if not isinstance(tool_calls, list):
                    continue

                for tool_call in tool_calls:
                    index = tool_call["index"]
                    tool_id = tool_call["id"]

                    tool_info = tool_calls_map.get(
                        index, {"name": "", "args": "", "tool_call_id": tool_id}
                    )

                    if "name" in tool_call["function"]:
                        tool_info["name"] = tool_call["function"]["name"]

                    if "arguments" in tool_call["function"]:
                        tool_info["args"] += tool_call["function"]["arguments"]

                        tool = next((t for t in tools if t["name"] == tool_info["name"]), None)
                        if tool and "on_format_and_yield" in tool:
                            try:
                                _ = ujson.loads(tool_info["args"])
                                yield tool["on_format_and_yield"](
                                    tool_info["args"],
                                    tool_info["tool_call_id"],
                                    tool_info["name"],
                                    index,
                                )
                            except ujson.JSONDecodeError:
                                pass

                    tool_calls_map[index] = tool_info

            for tool_index, tool_info in tool_calls_map.items():
                if tool_info["name"] and tool_info["args"]:
                    tool = next((t for t in tools if t["name"] == tool_info["name"]), None)
                    if tool and "on_call" in tool:
                        tool["on_call"](
                            tool_info["args"],
                            tool_info["tool_call_id"],
                            tool_info["name"],
                            tool_index,
                        )

        on_return(generate_output())

    capture = {
        "type": "capture",
        "on_stream": handle_stream,
    }

    return [*tool_elements, populate_on_stream_response_object_from_on_stream(capture)]


def valid_function_name(name: str) -> bool:
    """Validate function name - must be 1-64 chars of a-z, A-Z, 0-9, and underscores."""
    return bool(re.match(r"^[a-zA-Z0-9_]{1,64}$", name))


def Function(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    on_call: Optional[Callable[[str], None]] = None,
) -> List[PromptElement]:
    """
    Create a function definition with optional callback for function calls.
    """
    if not valid_function_name(name):
        raise ValueError(
            f"Invalid function name: {name}. Function names must be between 1 and 64 characters "
            "long and may only contain a-z, A-Z, 0-9, and underscores."
        )

    function_def = {
        "type": "functionDefinition",
        "name": name,
        "description": description,
        "parameters": parameters,
    }

    if on_call is None:
        return [function_def]

    def handle_output(output: ChatCompletionResponseMessage) -> None:
        if (
            (function_call := output.get("function_call"))
            and function_call.get("name") == name
            and (args := function_call.get("arguments")) is not None
        ):
            on_call(args)

    capture = {
        "type": "capture",
        "on_output": handle_output,
    }

    return [function_def, capture]


def pydantic_to_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Convert a Pydantic model to JSON Schema."""
    schema = model.model_json_schema()
    # Remove unnecessary fields from the schema
    for field in ["title", "description", "$defs"]:
        schema.pop(field, None)
    return schema


def PydanticFunction(
    name: str,
    description: str,
    parameters: Type[BaseModel],
    on_call: Optional[Callable[[Any], None]] = None,
    on_parse_error: Optional[Callable[[Exception, str], None]] = None,
) -> List[PromptElement]:
    """
    Create a function definition with Pydantic model validation.
    """
    json_schema = pydantic_to_json_schema(parameters)

    if on_call is None:
        return Function(name, description, json_schema)

    def handle_args(raw_args: str) -> None:
        if not raw_args:
            return

        try:
            args_dict = ujson.loads(raw_args)
            parsed_args = parameters.model_validate(args_dict)
            on_call(parsed_args)
        except Exception as error:
            if on_parse_error:
                on_parse_error(error, raw_args)
            else:
                raise

    return Function(
        name=name,
        description=description,
        parameters=json_schema,
        on_call=handle_args,
    )


def PydanticTools(
    tools: List[Dict[str, Any]],
    on_return: OutputHandler[Iterable[str]],
) -> List[PromptElement]:
    """
    Create tool definitions with Pydantic model validation.
    """
    processed_tools = []

    for tool in tools:
        parameters_model = tool["parameters"]
        json_schema = pydantic_to_json_schema(parameters_model)

        def handle_call(
            args: str,
            tool_call_id: str,
            tool_name: str,
            tool_index: int,
        ) -> None:
            try:
                parsed_args = parameters_model.model_validate_json(args)
                if tool.get("on_call"):
                    tool["on_call"](parsed_args, tool_call_id, tool_name, tool_index)
            except Exception as error:
                if tool.get("on_parse_error"):
                    tool["on_parse_error"](error, args)
                else:
                    raise

        def handle_format_and_yield(
            args: str,
            tool_call_id: str,
            tool_name: str,
            tool_index: int,
        ) -> str:
            try:
                parsed_args = parameters_model.model_validate_json(args)
                if tool.get("on_format_and_yield"):
                    return tool["on_format_and_yield"](
                        parsed_args,
                        tool_call_id,
                        tool_name,
                        tool_index,
                    )
                return args
            except Exception as error:
                print(f"Error formatting arguments for tool {tool_name}:", error)
                return args

        processed_tool = {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": json_schema,
        }

        if tool.get("on_call"):
            processed_tool["on_call"] = handle_call
        if tool.get("on_format_and_yield"):
            processed_tool["on_format_and_yield"] = handle_format_and_yield

        processed_tools.append(processed_tool)

    return Tools(processed_tools, on_return)
