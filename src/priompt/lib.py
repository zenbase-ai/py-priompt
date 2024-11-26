from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterable,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)
import time
import math

import ujson

from .tokenizer import num_tokens_for_image

if TYPE_CHECKING:
    from .tokenizer import PriomptTokenizer
    from .types import (
        Node,
        PromptElement,
        ChatPrompt,
        RenderedPrompt,
        RenderOutput,
        RenderOptions,
        ChatPromptMessage,
        PromptString,
        OutputHandler,
        ConfigProps,
        SourceMap,
        ChatCompletionResponseMessage,
        StreamChatCompletionResponse,
        ChatAndFunctionPromptFunction,
        ChatAndToolPromptToolFunction,
    )

# Constants
BASE_PRIORITY = 1e9


@dataclass
class NodeWithPriority:
    """Node with its calculated priority."""

    node: Node
    priority: int
    name: Optional[str] = None


def get_image_mime_type(bytes_data: bytes) -> str:
    """Check image magic numbers to determine mime type."""
    if bytes_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif bytes_data.startswith(b"\x89PNG\x0d\x0a\x1a\x0a"):
        return "image/png"
    elif bytes_data.startswith(b"GIF"):
        return "image/gif"
    elif bytes_data.startswith(b"RIFF") and bytes_data[8:12] == b"WEBP":
        return "image/webp"
    else:
        raise ValueError("Unsupported image type")


def chat_prompt_to_string(prompt: ChatPrompt) -> str:
    """Convert a chat prompt to string format."""
    return "\n".join(
        f"<|im_start|>{msg['role']}<|im_sep|>{msg['content']}<|im_end|>"
        for msg in prompt["messages"]
    )


def function_prompt_to_string(prompt: Dict) -> str:
    """Convert a function prompt to string format."""
    return "\n".join(ujson.dumps(func) for func in prompt["functions"])


OPENAI_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_sep|>",
    "<|im_end|>",
    "<|meta_start|>",
    "<|meta_sep|>",
    "<|meta_end|>",
    "<|endoftext|>",
    "<|endofprompt|>",
    "<|endoffile|>",
    "<|startoftext|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|disc_score|>",
    "<|disc_sep|>",
    "<|disc_thread|>",
    "<|ipynb_marker|>",
    "<|diff_marker|>",
    "<|ghissue|>",
    "<|ghreview|>",
]


def replace_openai_special_tokens(s: str) -> str:
    """Replace OpenAI special tokens with safer versions."""
    for token in OPENAI_SPECIAL_TOKENS:
        s = s.replace(token, token.replace("<|", "<").replace("|>", ">"))
    return s


def is_chat_prompt(prompt: Optional[RenderedPrompt]) -> bool:
    """Check if prompt is a chat prompt."""
    return (
        isinstance(prompt, dict) and not isinstance(prompt, list) and prompt.get("type") == "chat"
    )


def is_plain_prompt(prompt: Optional[RenderedPrompt]) -> bool:
    """Check if prompt is a plain string or list prompt."""
    return isinstance(prompt, (str, list))


def is_prompt_content(prompt: Optional[RenderedPrompt]) -> bool:
    """Check if prompt is a prompt content wrapper."""
    return (
        isinstance(prompt, dict)
        and not isinstance(prompt, list)
        and prompt.get("type") == "prompt_content"
    )


def prompt_has_functions(prompt: Optional[RenderedPrompt]) -> bool:
    """Check if prompt has functions."""
    return isinstance(prompt, dict) and "functions" in prompt


def prompt_has_tools(prompt: Optional[RenderedPrompt]) -> bool:
    """Check if prompt has tools."""
    return isinstance(prompt, dict) and "tools" in prompt


def prompt_string_to_string(prompt_string: PromptString) -> str:
    """Convert a prompt string to a regular string."""
    if isinstance(prompt_string, list):
        return "".join(prompt_string)
    return prompt_string


def sum_prompts(
    a: Optional[RenderedPrompt], b: Optional[RenderedPrompt]
) -> Optional[RenderedPrompt]:
    """Combine two prompts into one."""
    if a is None:
        return b
    if b is None:
        return a

    # Handle chat prompts
    if (
        (is_chat_prompt(a) and is_chat_prompt(b))
        or (is_chat_prompt(a) and prompt_get_text(b) == "")
        or (is_chat_prompt(b) and prompt_get_text(a) == "")
    ):
        functions = (a.get("functions", []) if prompt_has_functions(a) else []) + (
            b.get("functions", []) if prompt_has_functions(b) else []
        )
        tools = (a.get("tools", []) if prompt_has_tools(a) else []) + (
            b.get("tools", []) if prompt_has_tools(b) else []
        )

        prompt: Dict[str, Any] = {
            "type": "chat",
            "messages": (
                (a["messages"] if is_chat_prompt(a) else [])
                + (b["messages"] if is_chat_prompt(b) else [])
            ),
        }

        if functions:
            prompt["functions"] = functions
        if tools:
            prompt["tools"] = tools

        return prompt

    # Handle plain prompts
    if is_plain_prompt(a) and is_plain_prompt(b):
        return sum_prompt_strings(a, b)

    raise ValueError(f"Cannot sum prompts {a} and {b}")


def sum_prompt_strings(a: PromptString, b: PromptString) -> PromptString:
    """Combine two prompt strings."""
    if isinstance(a, list) and len(a) == 0:
        return b
    if isinstance(b, list) and len(b) == 0:
        return a

    if isinstance(a, list) and isinstance(b, list):
        result = [*a[:-1], a[-1] + b[0], *b[1:]]
        return result

    if isinstance(a, list):
        result = a.copy()
        result[-1] += b
        return result

    if isinstance(b, list):
        result = b.copy()
        result[0] = a + result[0]
        return result

    return a + b


class TooManyTokensForBasePriority(Exception):
    """Raised when base prompt token count exceeds limit."""


def prompt_get_text(prompt: Optional[RenderedPrompt]) -> str:
    """Get text content from a prompt."""
    if not is_text_prompt_potentially_with_functions(prompt):
        return ""
    if is_plain_prompt(prompt):
        return prompt_string_to_string(prompt)
    return prompt_string_to_string(prompt["text"])


def is_text_prompt_potentially_with_functions(prompt: Optional[RenderedPrompt]) -> bool:
    """Check if prompt is a text prompt that may contain functions."""
    return (isinstance(prompt, dict) and "text" in prompt) or isinstance(prompt, str)


def render_unsafe(elem: PromptElement, options: RenderOptions) -> RenderOutput:
    """
    Synchronous version of render that uses unsafe token counting.
    Only use this when you know the token counts are accurate enough.
    """
    # Handle isolated nodes
    if isinstance(elem, dict) and elem.get("type") == "isolate":
        if elem.get("cached_render_output"):
            return elem["cached_render_output"]

        isolated_options = {**options}
        isolated_options["token_limit"] = elem["token_limit"]
        isolated_options["count_tokens_fast_unsafe"] = True

        output = render_unsafe(elem["children"], isolated_options)
        elem["cached_render_output"] = output
        return output

    # Initialize output handlers
    output_handlers: List[OutputHandler[ChatCompletionResponseMessage]] = []
    stream_handlers: List[OutputHandler[AsyncIterable[ChatCompletionResponseMessage]]] = []
    stream_response_object_handlers: List[
        OutputHandler[AsyncIterable[StreamChatCompletionResponse]]
    ] = []

    # Initialize config
    config: ConfigProps = {"max_response_tokens": None, "stop": None}

    start_time = time.perf_counter()

    # Get base priority nodes
    base_priority_nodes = get_base_priority_nodes(
        elem,
        options["tokenizer"],
        True,  # Always use fast token counting
    )

    # Count tokens for base priority nodes
    base_token_count = count_tokens_for_nodes(
        base_priority_nodes,
        options["tokenizer"],
        True,  # Always use fast token counting
    )

    if base_token_count > options["token_limit"]:
        raise TooManyTokensForBasePriority(
            f"Base priority nodes use {base_token_count} tokens, "
            f"but limit is {options['token_limit']}"
        )

    # Binary search for optimal priority cutoff
    priority_cutoff = binary_search_priority_cutoff(elem, base_token_count, options)

    # Get final nodes and render prompt
    final_nodes = get_nodes_up_to_priority(
        elem,
        priority_cutoff,
        options["tokenizer"],
        True,  # Always use fast token counting
    )

    rendered_prompt = nodes_to_rendered_prompt(final_nodes)

    # Build source map if requested
    source_map = None
    if options.get("should_build_source_map", False):
        source_map = build_source_map(final_nodes)

    return {
        "prompt": rendered_prompt,
        "token_count": base_token_count,
        "token_limit": options["token_limit"],
        "tokenizer": options["tokenizer"],
        "tokens_reserved": options["token_limit"] - base_token_count,
        "priority_cutoff": priority_cutoff,
        "output_handlers": output_handlers,
        "stream_handlers": stream_handlers,
        "stream_response_object_handlers": stream_response_object_handlers,
        "config": config,
        "duration_ms": int((time.perf_counter() - start_time) * 1000),
        "source_map": source_map,
    }


def render(elem: PromptElement, options: RenderOptions) -> RenderOutput:
    """
    Render a prompt element with priority-based token management.
    This is the main rendering function.
    """
    # Handle isolated nodes separately
    if isinstance(elem, dict) and elem.get("type") == "isolate":
        if elem.get("cached_render_output"):
            return elem["cached_render_output"]

        isolated_options = {**options}
        isolated_options["token_limit"] = elem["token_limit"]

        output = render(elem["children"], isolated_options)
        elem["cached_render_output"] = output
        return output

    return render_binary_search(elem, options)


def render_binary_search(elem: PromptElement, options: RenderOptions) -> RenderOutput:
    """
    Render a prompt element using binary search to find optimal token usage.
    """
    start_time = time.perf_counter()

    # Initialize output handlers
    output_handlers: List[OutputHandler[ChatCompletionResponseMessage]] = []
    stream_handlers: List[OutputHandler[AsyncIterable[ChatCompletionResponseMessage]]] = []
    stream_response_object_handlers: List[
        OutputHandler[AsyncIterable[StreamChatCompletionResponse]]
    ] = []

    # Initialize config
    config: ConfigProps = {"max_response_tokens": None, "stop": None}
    count_tokens_fast = options.get("count_tokens_fast_unsafe", False)

    # Get base priority nodes
    base_priority_nodes = get_base_priority_nodes(elem, options["tokenizer"], count_tokens_fast)

    # Count tokens for base priority nodes
    base_token_count = count_tokens_for_nodes(
        base_priority_nodes, options["tokenizer"], count_tokens_fast
    )

    if base_token_count > options["token_limit"]:
        raise TooManyTokensForBasePriority(
            f"Base priority nodes use {base_token_count} tokens, "
            f"but limit is {options['token_limit']}"
        )

    # Binary search for optimal priority cutoff
    priority_cutoff = binary_search_priority_cutoff(elem, base_token_count, options)

    # Get final nodes and render prompt
    final_nodes = get_nodes_up_to_priority(
        elem, priority_cutoff, options["tokenizer"], count_tokens_fast
    )

    rendered_prompt = nodes_to_rendered_prompt(final_nodes)

    # Build source map if requested
    source_map = None
    if options.get("should_build_source_map", False):
        source_map = build_source_map(final_nodes)

    return {
        "prompt": rendered_prompt,
        "token_count": base_token_count,
        "token_limit": options["token_limit"],
        "tokenizer": options["tokenizer"],
        "tokens_reserved": options["token_limit"] - base_token_count,
        "priority_cutoff": priority_cutoff,
        "output_handlers": output_handlers,
        "stream_handlers": stream_handlers,
        "stream_response_object_handlers": stream_response_object_handlers,
        "config": config,
        "duration_ms": int((time.perf_counter() - start_time) * 1000),
        "source_map": source_map,
    }


def binary_search_priority_cutoff(
    elem: PromptElement,
    base_token_count: int,
    options: RenderOptions,
) -> int:
    """Find optimal priority cutoff using binary search."""
    min_priority = 0
    max_priority = BASE_PRIORITY
    best_priority = BASE_PRIORITY

    count_tokens_fast = options.get("count_tokens_fast_unsafe", False)

    while min_priority <= max_priority:
        mid_priority = (min_priority + max_priority) // 2

        nodes = get_nodes_up_to_priority(
            elem, mid_priority, options["tokenizer"], count_tokens_fast
        )

        token_count = count_tokens_for_nodes(nodes, options["tokenizer"], count_tokens_fast)

        if token_count <= options["token_limit"]:
            best_priority = mid_priority
            max_priority = mid_priority - 1
        else:
            min_priority = mid_priority + 1

    return best_priority


def get_base_priority_nodes(
    elem: PromptElement, tokenizer: PriomptTokenizer, count_tokens_fast: bool = False
) -> List[Node]:
    """Get nodes with base priority."""
    nodes = traverse_nodes(elem)
    return [node.node for node in nodes if node.priority == BASE_PRIORITY]


def traverse_nodes(
    elem: PromptElement, current_priority: int = BASE_PRIORITY, parent_name: Optional[str] = None
) -> List[NodeWithPriority]:
    """Traverse nodes and calculate their priorities."""
    if elem is None or isinstance(elem, (bool, int)):
        return []

    if isinstance(elem, str):
        return [NodeWithPriority(elem, current_priority)]

    if isinstance(elem, list):
        results = []
        for item in elem:
            results.extend(traverse_nodes(item, current_priority, parent_name))
        return results

    if not isinstance(elem, dict):
        return []

    node_type = elem.get("type")
    if not node_type:
        return []

    if node_type == "scope":
        priority = elem.get("absolute_priority", current_priority)
        if elem.get("relative_priority"):
            priority = min(priority, current_priority - elem["relative_priority"])

        name = elem.get("name", parent_name)
        results = []
        for child in elem.get("children", []):
            results.extend(traverse_nodes(child, priority, name))
        return results

    if node_type == "first":
        results = []
        for child in elem.get("children", []):
            child_nodes = traverse_nodes(child, current_priority, parent_name)
            if child_nodes:
                results.extend(child_nodes)
                break
        return results

    if node_type == "isolate":
        # Handle isolated nodes separately
        return [NodeWithPriority(elem, current_priority, parent_name)]

    return [NodeWithPriority(elem, current_priority, parent_name)]


def count_tokens_for_nodes(
    nodes: List[Node], tokenizer: PriomptTokenizer, count_tokens_fast: bool = False
) -> int:
    """Count total tokens for a list of nodes."""
    token_count = tokenizer.estimate_tokens_fast if count_tokens_fast else tokenizer.num_tokens
    total = 0
    for node in nodes:
        if isinstance(node, str):
            total += token_count(node)
        elif isinstance(node, dict):
            if node["type"] == "chat":
                total += count_chat_message_tokens(node, tokenizer, count_tokens_fast)
            elif node["type"] == "image":
                total += num_tokens_for_image(node["dimensions"], node["detail"])
            elif node["type"] == "empty":
                if "token_count" in node:
                    total += node["token_count"]
                elif "token_function" in node:
                    total += node["token_function"](tokenizer.num_tokens)
    return total


def count_chat_message_tokens(
    message: ChatPromptMessage, tokenizer: PriomptTokenizer, count_tokens_fast: bool = False
) -> int:
    """Count tokens in a chat message."""
    token_count = tokenizer.estimate_tokens_fast if count_tokens_fast else tokenizer.num_tokens
    if message["role"] == "function":
        # Add extra tokens for function messages
        name_tokens = token_count(message["name"])
        content_tokens = count_tokens_for_content(message["content"], tokenizer, count_tokens_fast)
        return name_tokens + content_tokens + 2

    if message["role"] == "assistant" and "function_call" in message:
        function_tokens = count_function_call_tokens(
            message["function_call"], tokenizer, count_tokens_fast
        )
        content_tokens = 0
        if message.get("content"):
            content_tokens = count_tokens_for_content(
                message["content"], tokenizer, count_tokens_fast
            )
        return function_tokens + content_tokens

    content_tokens = count_tokens_for_content(
        message.get("content", ""), tokenizer, count_tokens_fast
    )

    # Add image tokens if present
    if message["role"] == "user" and "images" in message:
        for image in message["images"]:
            content_tokens += num_tokens_for_image(
                image["image_url"]["dimensions"], image["image_url"]["detail"]
            )

    return content_tokens


def count_tokens_for_content(
    content: Union[str, List[str], None],
    tokenizer: PriomptTokenizer,
    count_tokens_fast: bool = False,
) -> int:
    """Count tokens in content string or list."""
    if content is None:
        return 0

    token_count = tokenizer.estimate_tokens_fast if count_tokens_fast else tokenizer.num_tokens

    if isinstance(content, list):
        return sum(token_count(item) for item in content)

    return token_count(content)


def count_function_call_tokens(
    function_call: Dict[str, str], tokenizer: PriomptTokenizer, count_tokens_fast: bool = False
) -> int:
    """Count tokens in a function call."""
    token_count = tokenizer.estimate_tokens_fast if count_tokens_fast else tokenizer.num_tokens

    name_tokens = token_count(function_call["name"])
    args_tokens = token_count(function_call["arguments"])

    return name_tokens + args_tokens + 5  # Extra tokens for function call structure


def get_nodes_up_to_priority(
    elem: PromptElement,
    priority_cutoff: int,
    _tokenizer: PriomptTokenizer,
    _count_tokens_fast: bool = False,
) -> List[Node]:
    """Get all nodes up to given priority cutoff."""
    nodes = traverse_nodes(elem)
    return [node.node for node in nodes if node.priority >= priority_cutoff]


def nodes_to_rendered_prompt(nodes: List[Node]) -> RenderedPrompt:
    """Convert nodes to final rendered prompt."""
    result: List[Union[str, Dict]] = []
    current_chat_messages: List[ChatPromptMessage] = []
    functions: List[ChatAndFunctionPromptFunction] = []
    tools: List[ChatAndToolPromptToolFunction] = []

    for node in nodes:
        if isinstance(node, str):
            if current_chat_messages:
                last_message = current_chat_messages[-1]
                if last_message.get("content"):
                    last_message["content"] += node
                else:
                    last_message["content"] = node
            else:
                result.append(node)

        elif isinstance(node, dict):
            if node["type"] == "chat":
                current_chat_messages.append(node)
            elif node["type"] == "functionDefinition":
                functions.append(node)
            elif node["type"] == "toolDefinition":
                tools.append(node["tool"])

    if current_chat_messages:
        chat_prompt: ChatPrompt = {"type": "chat", "messages": current_chat_messages}
        if functions:
            chat_prompt["functions"] = functions
        if tools:
            chat_prompt["tools"] = tools
        return chat_prompt

    if len(result) == 1 and isinstance(result[0], str):
        return result[0]

    return result


def build_source_map(nodes: List[Node]) -> SourceMap:
    """Build source map from nodes."""

    def build_map(node: Node, start: int = 0) -> Tuple[SourceMap, int]:
        if isinstance(node, str):
            end = start + len(node)
            return {"name": "text", "string": node, "start": start, "end": end}, end

        if not isinstance(node, dict):
            return {"name": "unknown", "start": start, "end": start}, start

        name = node.get("name", node.get("type", "unknown"))
        children: List[SourceMap] = []
        current_pos = start

        if "children" in node:
            for child in node["children"]:
                child_map, current_pos = build_map(child, current_pos)
                children.append(child_map)

        return {
            "name": name,
            "children": children if children else None,
            "start": start,
            "end": current_pos,
        }, current_pos

    root_map, _ = build_map({"type": "root", "children": nodes})
    return root_map


def estimate_function_tokens_using_charcount(
    function_definition: Dict, tokenizer: PriomptTokenizer
) -> Tuple[int, int]:
    """Estimate token count range for a function definition using character count."""
    stringified_function = ujson.dumps(
        {
            "name": function_definition["name"],
            "description": function_definition["description"],
            "parameters": function_definition["parameters"],
        },
        indent=2,
    )
    raw = tokenizer.estimate_tokens_using_char_count(stringified_function)
    # Multiply by 1.5 and add 10 for upper bound to be safe until more testing
    return (math.ceil(raw[0] * 0.5), math.ceil(raw[1] * 1.5) + 10)


def estimate_tool_tokens_using_charcount(
    tool_definition: Dict, tokenizer: PriomptTokenizer
) -> Tuple[int, int]:
    """Estimate token count range for a tool definition using character count."""
    stringified_tool = ujson.dumps(
        {
            "name": tool_definition["function"]["name"],
            "description": tool_definition["function"]["description"],
            "parameters": tool_definition["function"]["parameters"],
        },
        indent=2,
    )
    raw = tokenizer.estimate_tokens_using_char_count(stringified_tool)
    # Multiply by 1.5 and add 10 for upper bound to be safe until more testing
    return (math.ceil(raw[0] * 0.5), math.ceil(raw[1] * 1.5) + 10)


def estimate_lower_bound_tokens_for_prompt(
    prompt: Optional[RenderedPrompt], tokenizer: PriomptTokenizer
) -> int:
    """Estimate minimum number of tokens needed for a prompt."""
    if prompt is None:
        return 0

    content_tokens = 0
    if is_chat_prompt(prompt):
        for msg in prompt["messages"]:
            if msg["role"] == "function":
                # Assume no extra tokens for lower bound
                content_tokens += tokenizer.estimate_tokens_using_char_count(
                    msg["name"] + msg["content"]
                )[0]
            elif msg["role"] == "assistant" and "function_call" in msg:
                content_tokens += tokenizer.estimate_tokens_using_char_count(
                    msg["function_call"]["name"]
                    + msg["function_call"]["arguments"]
                    + (msg.get("content", ""))
                )[0]
            else:
                content = msg.get("content", "")
                if content:
                    content_tokens += tokenizer.estimate_tokens_using_char_count(
                        prompt_string_to_string(content)
                    )[0]
    elif is_plain_prompt(prompt):
        content_tokens = tokenizer.estimate_tokens_using_char_count(
            prompt_string_to_string(prompt)
        )[0]
    elif is_prompt_content(prompt):
        content_tokens = tokenizer.estimate_tokens_using_char_count(
            prompt_string_to_string(prompt["content"])
        )[0]
    else:
        content_tokens = tokenizer.estimate_tokens_using_char_count(
            prompt_string_to_string(prompt["text"])
        )[0]

    function_tokens = (
        sum(
            estimate_function_tokens_using_charcount(func, tokenizer)[0]
            for func in prompt.get("functions", [])
        )
        if prompt_has_functions(prompt)
        else 0
    )

    tool_tokens = (
        sum(
            estimate_tool_tokens_using_charcount(tool, tokenizer)[0]
            for tool in prompt.get("tools", [])
        )
        if prompt_has_tools(prompt)
        else 0
    )

    return content_tokens + function_tokens + tool_tokens


def get_prompt_element_node_count(elem: PromptElement) -> int:
    """Count the number of nodes in a prompt element tree."""
    if elem is None or isinstance(elem, (bool, int, str)):
        return 1

    if isinstance(elem, list):
        return sum(get_prompt_element_node_count(p) for p in elem)

    if not isinstance(elem, dict):
        return 1

    node_type = elem.get("type")
    if node_type in (
        "functionDefinition",
        "toolDefinition",
        "breaktoken",
        "capture",
        "config",
        "empty",
        "image",
    ):
        return 1
    elif node_type in ("first", "isolate", "scope", "chat"):
        return 1 + get_prompt_element_node_count(elem["children"])

    return 1


def normalize_source_map(source_map: SourceMap) -> SourceMap:
    """Normalize a source map by flattening single-child nodes."""
    if not source_map.get("children"):
        source_map["children"] = None
        return source_map

    if len(source_map["children"]) == 0:
        source_map["children"] = None
        return source_map

    if len(source_map["children"]) == 1:
        return normalize_source_map(
            {
                "name": f"{source_map['name']}.{source_map['children'][0]['name']}",
                "children": source_map["children"][0].get("children"),
                "start": source_map["start"],
                "end": source_map["end"],
            }
        )

    source_map["children"] = [normalize_source_map(c) for c in source_map["children"]]
    return source_map


def merge_source_maps(
    source_maps: List[Optional[SourceMap]], source_name: str
) -> Optional[SourceMap]:
    """Merge multiple source maps into one."""
    filtered_maps = [s for s in source_maps if s is not None]
    if not filtered_maps:
        return None

    shifted_maps = [filtered_maps[0]]
    max_end = filtered_maps[0]["end"]

    for next_map in filtered_maps[1:]:
        new_base = shifted_maps[-1]["end"]
        next_map["start"] += new_base
        next_map["end"] += new_base
        max_end = max(max_end, next_map["end"])
        shifted_maps.append(next_map)

    return {"name": source_name, "children": shifted_maps, "start": 0, "end": max_end}


def prompt_to_tokens(prompt: RenderedPrompt, tokenizer: PriomptTokenizer) -> List[int]:
    """Convert a prompt to tokens, always leaving the last message 'open'."""
    if is_plain_prompt(prompt):
        if isinstance(prompt, list):
            tokens = []
            for s in prompt:
                tokens.extend(tokenizer.encode_tokens(s))
            return tokens
        return tokenizer.encode_tokens(prompt)

    elif is_chat_prompt(prompt):
        messages = prompt["messages"]
        for msg in messages:
            if msg["role"] == "function":
                raise ValueError("Function messages not supported in promptToTokens yet")
            elif msg["role"] == "assistant" and "function_call" in msg:
                raise ValueError("Function call messages not supported in promptToTokens yet")
            elif msg.get("content") is None:
                raise ValueError("Message content cannot be None")

        return tokenizer.apply_chat_template_tokens(
            [
                {
                    "role": msg["role"],
                    "name": msg.get("name"),
                    "to": msg.get("to"),
                    "content": msg["content"]
                    if not isinstance(msg["content"], list)
                    else "".join(msg["content"]),
                }
                for msg in messages
            ]
        )

    raise ValueError(f"Invalid prompt type: {prompt}")


def count_msg_tokens(message: ChatPromptMessage, tokenizer: PriomptTokenizer) -> int:
    """Count tokens in a chat message."""
    if message["role"] == "function":
        # Add extra 2 tokens for good measure
        return (
            tokenizer.num_tokens(message["name"])
            + num_tokens_prompt_string(message["content"], tokenizer)
            + 2
        )

    elif message["role"] == "assistant" and "function_call" in message:
        function_tokens = count_function_call_message_tokens(message["function_call"], tokenizer)
        content_tokens = (
            num_tokens_prompt_string(message["content"], tokenizer)
            if message.get("content") is not None
            else 0
        )
        return function_tokens + content_tokens

    else:
        num_tokens = num_tokens_prompt_string(message.get("content", ""), tokenizer)
        if message["role"] == "user" and "images" in message:
            for image in message["images"]:
                num_tokens += num_tokens_for_image(
                    image["image_url"]["dimensions"], image["image_url"]["detail"]
                )
        return num_tokens


def count_function_call_message_tokens(
    function_call: Dict[str, str], tokenizer: PriomptTokenizer
) -> int:
    """Count tokens in a function call message."""
    # Add constant factor for function call structure
    return (
        tokenizer.num_tokens(function_call["name"])
        + tokenizer.num_tokens(function_call["arguments"])
        + 5
    )


def count_function_tokens(
    function_definition: ChatAndFunctionPromptFunction, tokenizer: PriomptTokenizer
) -> int:
    """Count tokens in a function definition."""
    stringified_function = ujson.dumps(
        {
            "name": function_definition["name"],
            "description": function_definition["description"],
            "parameters": function_definition["parameters"],
        },
        indent=2,
    )
    # Multiply by 1.5 and add 10 to be safe until more testing
    raw = tokenizer.num_tokens(stringified_function)
    return math.ceil(raw * 1.5) + 10


def count_tool_tokens(
    tool_definition: ChatAndToolPromptToolFunction, tokenizer: PriomptTokenizer
) -> int:
    """Count tokens in a tool definition."""
    stringified_tool = ujson.dumps(
        {
            "name": tool_definition["function"]["name"],
            "description": tool_definition["function"]["description"],
            "parameters": tool_definition["function"]["parameters"],
        },
        indent=2,
    )
    # Multiply by 1.5 and add 10 to be safe until more testing
    raw = tokenizer.num_tokens(stringified_tool)
    return math.ceil(raw * 1.5) + 10


def num_tokens_prompt_string(
    prompt_string: Optional[PromptString], tokenizer: PriomptTokenizer
) -> int:
    """Count tokens in a prompt string."""
    if prompt_string is None:
        return 0
    if isinstance(prompt_string, list):
        return sum(tokenizer.num_tokens(s) for s in prompt_string)
    return tokenizer.num_tokens(prompt_string)


def prompt_to_openai_chat_messages(prompt: RenderedPrompt) -> List[Dict[str, Any]]:
    """Convert a prompt to OpenAI chat messages format."""
    if is_plain_prompt(prompt):
        return [{"role": "user", "content": prompt_string_to_string(prompt)}]

    elif is_chat_prompt(prompt):
        messages = []
        for msg in prompt["messages"]:
            if msg["role"] == "function":
                messages.append(
                    {
                        "role": msg["role"],
                        "name": msg["name"],
                        "content": prompt_string_to_string(msg["content"]),
                    }
                )
            elif msg["role"] == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "name": msg["name"],
                        "tool_call_id": msg["to"],
                        "content": prompt_string_to_string(msg["content"]),
                    }
                )
            elif msg["role"] == "assistant" and "function_call" in msg:
                messages.append(
                    {
                        "role": msg["role"],
                        "content": prompt_string_to_string(msg.get("content", "")),
                        "function_call": msg["function_call"],
                    }
                )
            elif msg["role"] == "assistant" and "tool_calls" in msg:
                messages.append(
                    {
                        "role": msg["role"],
                        "content": prompt_string_to_string(msg.get("content", "")),
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": tool_call["id"],
                                "index": tool_call["index"],
                                "function": {
                                    "name": tool_call["tool"]["function"]["name"],
                                    "arguments": tool_call["tool"]["function"]["arguments"],
                                },
                            }
                            for tool_call in msg["tool_calls"]
                        ],
                    }
                )
            elif msg["role"] == "assistant":
                messages.append(
                    {
                        "role": msg["role"],
                        "content": prompt_string_to_string(msg.get("content", "")),
                    }
                )
            elif msg["role"] == "system":
                messages.append(
                    {
                        "role": msg["role"],
                        "name": msg.get("name"),
                        "content": prompt_string_to_string(msg.get("content", "")),
                    }
                )
            else:
                if msg.get("images"):
                    content = []
                    content.extend(msg["images"])
                    text_content = prompt_string_to_string(msg.get("content", ""))
                    content.append({"type": "text", "text": text_content})
                    messages.append(
                        {"role": msg["role"], "content": content, "name": msg.get("name")}
                    )
                else:
                    messages.append(
                        {
                            "role": msg["role"],
                            "content": prompt_string_to_string(msg.get("content", "")),
                            "name": msg.get("name"),
                        }
                    )
        return messages

    raise ValueError(f"Invalid prompt type: {prompt}")


def prompt_to_openai_chat_request(prompt: RenderedPrompt) -> Dict[str, Any]:
    """Convert a prompt to OpenAI chat completion request format."""
    functions = prompt["functions"] if prompt_has_functions(prompt) else None
    tools = prompt["tools"] if prompt_has_tools(prompt) else None
    messages = prompt_to_openai_chat_messages(prompt)

    return {
        "messages": messages,
        "functions": functions,
        "tools": tools,
        "tool_choice": "auto" if tools and len(tools) > 0 else None,
    }
