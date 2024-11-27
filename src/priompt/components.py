from __future__ import annotations
import functools
from typing import Literal, TypeAlias
import re

from beartype import beartype
from beartype.typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union
import ujson

from . import types
from .lib import flat
from .openai import ChatCompletionResponseMessage, StreamChatCompletionResponse


Priority: TypeAlias = Optional[types.Number]
OnEject: TypeAlias = Optional[types.NodeCallback]
OnInclude: TypeAlias = Optional[types.NodeCallback]


@beartype
def Scope(
    *children: types.PromptElement,
    name: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
    on_eject: OnEject = None,
    on_include: OnInclude = None,
) -> types.PromptElement:
    return {
        "type": "scope",
        "children": flat(children),
        "name": name,
        "absolute_priority": p,
        "relative_priority": prel,
        "on_eject": on_eject,
        "on_include": on_include,
    }


@beartype
def Isolate(
    *children: types.PromptElement,
    token_limit: Union[types.Number],
    name: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
) -> types.PromptElement:
    return {
        "type": "scope",
        "children": [
            {
                "type": "isolate",
                "token_limit": token_limit,
                "cached_render_output": None,
                "children": flat(children),
            }
        ],
        "name": name,
        "absolute_priority": p,
        "relative_priority": prel,
    }


@beartype
def Capture(
    *children: types.PromptElement,
    on_output: types.NodeCallback,
    on_stream: Optional[types.NodeCallback] = None,
    name: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
) -> types.PromptElement:
    if children:
        raise ValueError(f"capture tag must have no children, got {children}")

    return {
        "type": "scope",
        "children": [{"type": "capture", "on_output": on_output, "on_stream": on_stream}],
        "name": name,
        "absolute_priority": p,
        "relative_priority": prel,
    }


@beartype
def Config(
    *children,
    max_response_tokens: Optional[
        Union[types.Number, Literal["tokens_reserved", "tokens_remaining"]]
    ] = None,
    stop: Optional[Union[str, List[str]]] = None,
    name: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
) -> types.PromptElement:
    if children:
        raise ValueError(f"config tag must have no children, got {children}")

    return {
        "type": "scope",
        "children": [
            {
                "type": "config",
                "max_response_tokens": max_response_tokens,
                "stop": stop,
            }
        ],
        "name": name,
        "absolute_priority": p,
        "relative_priority": prel,
    }


@beartype
def First(
    *children: types.PromptElement,
    on_eject: Optional[types.NodeCallback] = None,
    on_include: Optional[types.NodeCallback] = None,
) -> types.PromptElement:
    new_children: List[types.PromptElement] = []
    for child in children:
        if not isinstance(child, dict):
            raise ValueError(f"First tag must have scope children, got {child}")
        if child["type"] != "scope":
            raise ValueError(f"First tag must have scope children, got {child}")
        new_children.append(child)

    return {
        "type": "first",
        "children": new_children,
        "on_eject": on_eject,
        "on_include": on_include,
    }


@beartype
def Empty(
    *children,
    tokens: Union[types.Number, Callable[[], int]],
    name: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
) -> types.PromptElement:
    if children:
        raise ValueError(f"empty tag must have no children, got {children}")

    return {
        "type": "scope",
        "children": [{"type": "empty", "token_count": tokens, "token_function": tokens}],
        "name": name,
        "absolute_priority": p,
        "relative_priority": prel,
    }


@beartype
def BreakToken(
    *children,
    name: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
) -> types.PromptElement:
    if children:
        raise ValueError(f"breaktoken tag must have no children, got {children}")

    return {
        "type": "scope",
        "children": [{"type": "breaktoken"}],
        "name": name,
        "absolute_priority": p,
        "relative_priority": prel,
    }


@beartype
def Hr(
    *children,
    name: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
) -> types.PromptElement:
    if children:
        raise ValueError(f"hr tag must have no children, got {children}")

    return {
        "type": "scope",
        "children": ["\n\n-------\n\n"],
        "name": name,
        "absolute_priority": p,
        "relative_priority": prel,
    }


@beartype
def Br(
    *children,
    name: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
) -> types.PromptElement:
    if children:
        raise ValueError(f"br tag must have no children, got {children}")

    return {
        "type": "scope",
        "children": ["\n"],
        "name": name,
        "absolute_priority": p,
        "relative_priority": prel,
    }


@beartype
def Image(
    image_bytes: bytes,
    detail: Literal["low", "high", "auto"],
    dimensions: types.ImageDimensions,
) -> types.PromptElement:
    return {
        "type": "image",
        "bytes": image_bytes,
        "dimensions": dimensions,
        "detail": detail,
    }


@beartype
def scope_props(props: Dict[str, Any]) -> types.ScopeProps:
    return {
        "name": props.get("name"),
        "absolute_priority": props.get("p"),
        "relative_priority": props.get("prel"),
        "on_eject": props.get("on_eject"),
        "on_include": props.get("on_include"),
    }


def component(fn):
    typed_fn = beartype(fn)

    @functools.wraps(fn)
    def wrapper(
        *children: types.PromptElement,
        **props,
    ) -> types.PromptElement:
        return {
            "type": "scope",
            "children": flat([typed_fn(*children, **props)]),
            **scope_props(props),
        }

    return wrapper


@component
def SystemMessage(
    *children: types.PromptElement,
    name: Optional[str] = None,
    to: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
    on_eject: OnEject = None,
    on_include: OnInclude = None,
) -> types.PromptElement:
    return {
        "type": "chat",
        "role": "system",
        "name": name,
        "to": to,
        "children": flat(children),
    }


@component
def UserMessage(
    *children: types.PromptElement,
    name: Optional[str] = None,
    to: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
    on_eject: OnEject = None,
    on_include: OnInclude = None,
) -> types.PromptElement:
    return {
        "type": "chat",
        "role": "user",
        "name": name,
        "to": to,
        "children": flat(children),
    }


@component
def AssistantMessage(
    *children: types.PromptElement,
    function_call: Optional[Dict[str, str]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    to: Optional[str] = None,
    name: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
    on_eject: OnEject = None,
    on_include: OnInclude = None,
) -> types.PromptElement:
    return {
        "type": "chat",
        "role": "assistant",
        "to": to,
        "children": flat(children),
        "function_call": function_call,
        "tool_calls": tool_calls,
    }


@component
def FunctionMessage(
    *children: types.PromptElement,
    name: str,
    to: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
    on_eject: OnEject = None,
    on_include: OnInclude = None,
) -> types.PromptElement:
    return {
        "type": "chat",
        "role": "function",
        "name": name,
        "to": to,
        "children": flat(children),
    }


@component
def ToolResultMessage(
    *children: types.PromptElement,
    name: str,
    to: Optional[str] = None,
    p: Priority = None,
    prel: Priority = None,
    on_eject: OnEject = None,
    on_include: OnInclude = None,
) -> types.PromptElement:
    return {
        "type": "chat",
        "role": "tool",
        "name": name,
        "to": to,
        "children": flat(children),
    }


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


@component
def Tool(
    name: str,
    description: str,
    parameters: Dict[str, Any],
) -> types.PromptElement:
    return {
        "type": "tool_definition",
        "tool": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        },
    }


@component
def Tools(
    tools: List[Dict[str, Any]],
    on_return: types.OutputHandler[Iterable[str]],
) -> List[types.PromptElement]:
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

                tool_calls = message.get("tool_calls")
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
                        if tool and callable(tool.get("on_format_and_yield")):
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
                    if tool and callable(tool.get("on_call")):
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


def valid_function_name(name: str, regex=re.compile(r"^[a-zA-Z0-9_]{1,64}$")) -> bool:
    """Validate function name - must be 1-64 chars of a-z, A-Z, 0-9, and underscores."""
    return bool(regex.match(name))


@component
def Function(
    name: str,
    description: str,
    parameters: Dict[str, Any],
    on_call: Optional[Callable[[str], None]] = None,
) -> Union[Tuple[types.FunctionDefinition], Tuple[types.FunctionDefinition, types.Capture]]:
    """
    Create a function definition with optional callback for function calls.
    """
    if not valid_function_name(name):
        raise ValueError(
            f"Invalid function name: {name}. Function names must be between 1 and 64 characters "
            "long and may only contain a-z, A-Z, 0-9, and underscores."
        )

    function_def = {
        "type": "function_definition",
        "name": name,
        "description": description,
        "parameters": parameters,
    }

    if on_call is None:
        return (function_def,)

    def handle_output(output: ChatCompletionResponseMessage) -> None:
        function_call = output.get("function_call")
        if not function_call:
            return

        if name != function_call.get("name"):
            return

        args = function_call.get("arguments")
        if not args:
            return

        on_call(args)

    capture = {
        "type": "capture",
        "on_output": handle_output,
    }

    return (function_def, capture)
