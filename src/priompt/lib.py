from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)
import base64
import math
import os
import time


import ujson

from priompt.types import ToolDefinition

from .openai import (
    CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT,
    CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR,
)
from .tokenizer import num_tokens_for_image

if TYPE_CHECKING:
    from .tokenizer import PriomptTokenizer
    from .types import (
        Number,
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
BASE_PRIORITY = int(1e9)


@dataclass
class NodeWithPriority:
    """Node with its calculated priority."""

    node: Node
    priority: int
    name: Optional[str] = None


def get_image_mime_type(bytes_data: bytes) -> str:
    """Check image magic numbers to determine mime type."""
    if bytes_data[0] == 0xFF and bytes_data[1] == 0xD8 and bytes_data[2] == 0xFF:
        return "image/jpeg"
    elif (
        bytes_data[0] == 0x89
        and bytes_data[1] == 0x50
        and bytes_data[2] == 0x4E
        and bytes_data[3] == 0x47
    ):
        return "image/png"
    elif bytes_data[0] == 0x47 and bytes_data[1] == 0x49 and bytes_data[2] == 0x46:
        return "image/gif"
    elif (
        bytes_data[0] == 0x52
        and bytes_data[1] == 0x49
        and bytes_data[2] == 0x46
        and bytes_data[3] == 0x46
    ):
        return "image/webp"
    else:
        raise ValueError("Unsupported image type")


def get_chat_prompt_content(p) -> str:
    if is_plain_prompt(p["prompt"]):
        return p["prompt"]
    if p["prompt"]:
        return p["prompt"].get("text", "")
    return ""


def chat_prompt_to_string(prompt: ChatPrompt) -> str:
    """Convert a chat prompt to string format."""
    return "\n".join(
        f"<|im_start|>{msg['role']}<|im_sep|>{msg['content']}<|im_end|>"
        for msg in prompt["messages"]
    )


def compact(d):
    """Compact a dictionary by removing keys with None values recursively."""
    if isinstance(d, dict):
        return {k: compact(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [compact(v) for v in d if v is not None]
    else:
        return d


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
    return isinstance(prompt, dict) and isinstance(prompt.get("functions"), list)


def prompt_has_tools(prompt: Optional[RenderedPrompt]) -> bool:
    """Check if prompt has tools."""
    return isinstance(prompt, dict) and isinstance(prompt.get("tools"), list)


def prompt_string_to_string(prompt_string: PromptString) -> str:
    """Convert a prompt string to a regular string."""
    if isinstance(prompt_string, (list, tuple)):
        return "".join(prompt_string)
    return prompt_string


def sum_prompts(
    a: Optional[RenderedPrompt],
    b: Optional[RenderedPrompt],
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

    if (prompt_has_tools(a) or prompt_has_tools(b)) and (
        prompt_has_functions(a) or prompt_has_functions(b)
    ):
        raise ValueError(
            f"Cannot sum prompts {a} and {b} since you should only use tools or functions, but not both"
        )

    if (prompt_has_tools(a) or prompt_has_tools(b)) and (
        is_text_prompt_potentially_with_functions(a)
        and is_text_prompt_potentially_with_functions(b)
    ):
        tools = (a.get("tools", []) if prompt_has_tools(a) else []) + (
            b.get("tools", []) if prompt_has_tools(b) else []
        )
        prompt: Dict[str, Any] = {
            "type": "text",
            "text": sum_prompt_strings(
                a if is_plain_prompt(a) else a["text"],
                b if is_plain_prompt(b) else b["text"],
            ),
            "tools": tools,
        }
        return prompt

    if (prompt_has_functions(a) or prompt_has_functions(b)) and (
        is_text_prompt_potentially_with_functions(a)
        and is_text_prompt_potentially_with_functions(b)
    ):
        functions = (a.get("functions", []) if prompt_has_functions(a) else []) + (
            b.get("functions", []) if prompt_has_functions(b) else []
        )
        prompt: Dict[str, Any] = {
            "type": "text",
            "text": sum_prompt_strings(
                a if is_plain_prompt(a) else a["text"],
                b if is_plain_prompt(b) else b["text"],
            ),
            "functions": functions,
        }
        return prompt

    if (prompt_has_tools(a) and is_prompt_content(b)) or (
        prompt_has_tools(b) and is_prompt_content(a)
    ):
        raise ValueError(
            f"Cannot sum prompts {a} and {b} since one has tools and the other has images"
        )

    if (prompt_has_functions(a) and is_prompt_content(b)) or (
        prompt_has_functions(b) and is_prompt_content(a)
    ):
        raise ValueError(
            f"Cannot sum prompts {a} and {b} since one has a function and the other has images"
        )

    if is_plain_prompt(a) and is_prompt_content(b):
        return {
            "type": b["type"],
            "content": sum_prompt_strings(a, b["content"]),
            "images": b["images"],
        }
    elif is_plain_prompt(b) and is_prompt_content(a):
        return {
            "type": a["type"],
            "content": sum_prompt_strings(a["content"], b),
            "images": a["images"],
        }
    elif is_prompt_content(a) and is_prompt_content(b):
        return {
            "type": a["type"],
            "content": sum_prompt_strings(a["content"], b["content"]),
            "images": (a.get("images", []) or []) + (b.get("images", []) or []),
        }

    if is_plain_prompt(a) and is_plain_prompt(b):
        return sum_prompt_strings(a, b)

    raise ValueError(
        f"Cannot sum prompts {a} ({type(a) if is_plain_prompt(a) else a['type']}) and {b} ({type(b) if is_plain_prompt(b) else b['type']})"
    )


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
    return (isinstance(prompt, dict) and prompt.get("text")) or isinstance(prompt, str)


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
    stream_handlers: List[OutputHandler[Iterable[ChatCompletionResponseMessage]]] = []
    stream_response_object_handlers: List[
        OutputHandler[Iterable[StreamChatCompletionResponse]]
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
        "prompt": compact(rendered_prompt),
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
    return render_binary_search(elem, options)


def render_binary_search(elem: PromptElement, options: RenderOptions) -> RenderOutput:
    """
    Render a prompt element using binary search to find optimal token usage.
    """
    start_time = time.perf_counter()
    elem = deepcopy(elem)

    # Validate the unrendered prompt
    validate_unrendered_prompt(elem)
    validating_duration = time.perf_counter() - start_time

    if should_print_verbose_logs():
        print(f"Validating prompt took {validating_duration * 1000} ms")

    _node_count = get_prompt_element_node_count(elem)
    # statsd logging would go here

    # Compute priority levels
    priority_levels = set()
    compute_priority_levels(elem, BASE_PRIORITY, priority_levels)
    priority_levels.add(BASE_PRIORITY)
    sorted_priority_levels = sorted(list(priority_levels))

    # Adjust token limit for fast counting
    count_tokens_fast = options.get("count_tokens_fast_unsafe", False)
    used_token_limit = (
        options["token_limit"] * 0.95 if count_tokens_fast else options["token_limit"]
    )

    # Hydrate isolates if needed
    hydrate_isolates(elem, options["tokenizer"], options.get("should_build_source_map", False))
    hydrate_empty_token_count(elem, options["tokenizer"])

    # Binary search for optimal priority cutoff
    largest_token_count_seen = 0
    exclusive_lower_bound = -1
    inclusive_upper_bound = len(sorted_priority_levels) - 1

    token_counter = count_tokens_approx_fast if count_tokens_fast else count_tokens_exact
    last_message_is_incomplete = options.get("last_message_is_incomplete", False)

    while exclusive_lower_bound < inclusive_upper_bound - 1:
        candidate_level_index = (exclusive_lower_bound + inclusive_upper_bound) // 2
        candidate_level = sorted_priority_levels[candidate_level_index]

        if should_print_verbose_logs():
            print(f"Trying candidate level {candidate_level} with index {candidate_level_index}")

        try:
            prompt = render_with_level_and_early_exit_with_token_estimation(
                elem,
                candidate_level,
                options["tokenizer"],
                options["token_limit"],
            )

            token_count = token_counter(
                options["tokenizer"],
                prompt.get("prompt", ""),
                {"last_message_is_incomplete": last_message_is_incomplete},
            )

            largest_token_count_seen = max(largest_token_count_seen, token_count)

            if token_count + prompt["empty_token_count"] > used_token_limit:
                exclusive_lower_bound = candidate_level_index
            else:
                inclusive_upper_bound = candidate_level_index

        except TokenLimitExceeded:
            exclusive_lower_bound = candidate_level_index

    # Final render with optimal priority level
    source_map_info = None
    if options.get("should_build_source_map", False):
        source_map_info = {"name": "root", "is_last": True}

    final_prompt = render_with_level(
        elem,
        sorted_priority_levels[inclusive_upper_bound],
        options["tokenizer"],
        True,
        source_map_info,
    )

    if final_prompt["source_map"]:
        final_prompt["source_map"] = normalize_source_map(final_prompt["source_map"])

    # Get exact token count for final output
    final_token_count = count_tokens_exact(
        options["tokenizer"],
        final_prompt.get("prompt", ""),
        {"last_message_is_incomplete": last_message_is_incomplete},
    )

    if final_token_count + final_prompt["empty_token_count"] > options["token_limit"]:
        raise TooManyTokensForBasePriority(
            f"Base prompt estimated token count is {final_token_count} with "
            f"{final_prompt['empty_token_count']} tokens reserved, which is higher than "
            f"the limit {options['token_limit']}. This is probably a bug in the prompt — "
            "please add some priority levels to fix this."
        )

    duration_ms = int((time.perf_counter() - start_time) * 1000)
    if should_print_verbose_logs() and duration_ms > 100:
        print(
            f"Priompt WARNING: rendering prompt took {duration_ms} ms, which is longer than "
            "the recommended maximum of 100 ms. Consider reducing the number of scopes you have."
        )

    return {
        "prompt": compact(final_prompt.get("prompt", "")),
        "token_count": final_token_count,
        "tokens_reserved": final_prompt["empty_token_count"],
        "token_limit": options["token_limit"],
        "tokenizer": options["tokenizer"],
        "duration_ms": duration_ms,
        "output_handlers": final_prompt.get("output_handlers", []),
        "stream_handlers": final_prompt.get("stream_handlers", []),
        "stream_response_object_handlers": final_prompt.get("stream_response_object_handlers", []),
        "priority_cutoff": sorted_priority_levels[inclusive_upper_bound],
        "source_map": final_prompt.get("source_map", None),
        "config": final_prompt.get("config", {}),
    }


def normalize_prompt(elem: PromptElement) -> List[Dict]:
    """Normalize a prompt element by merging adjacent strings and normalizing nodes."""
    result = []
    current_string = ""

    def push_current_string():
        nonlocal current_string
        if current_string:
            result.append({"type": "normalized_string", "s": current_string, "cached_count": None})
            current_string = ""

    elem_array = list(elem if isinstance(elem, (list, tuple)) else [elem])

    for node in elem_array:
        if node is None:
            continue

        if isinstance(node, str):
            current_string += node
        elif isinstance(node, (int, float)):
            current_string += str(node)
        elif isinstance(node, dict):
            push_current_string()

            node_type = node.get("type")

            if node_type in ("config", "capture", "isolate", "breaktoken", "image", "empty"):
                new_node = node
            elif node_type in ("tool_definition", "function_definition"):
                new_node = {**node, "cached_count": None}
            elif node_type == "first":
                new_node = {
                    **node,
                    "children": [
                        {**c, "children": normalize_prompt(c["children"])} for c in node["children"]
                    ],
                }
            elif node_type in ("chat", "scope"):
                new_node = {**node, "children": normalize_prompt(node["children"])}
            else:
                raise ValueError("Invalid prompt element")

            result.append(new_node)
        else:
            raise ValueError("Invalid prompt element")

    push_current_string()
    return result


def render_cumulative_sum(
    elem: PromptElement,
    options: RenderOptions,
) -> RenderOutput:
    """A fast, synchronous, somewhat inexact and incomplete way to render a prompt.
    Yields ~50x speedup in many cases and is useful for datajobs."""
    start_time = time.perf_counter() if should_print_verbose_logs() else None
    elem = deepcopy(elem)

    if not options.get("tokenizer"):
        raise ValueError("Must specify tokenizer!")
    tokenizer = options["tokenizer"]
    token_limit = options["token_limit"]

    if should_print_verbose_logs():
        start_time_validating = time.perf_counter()
    validate_unrendered_prompt(elem)
    if should_print_verbose_logs():
        end_time_validating = time.perf_counter()
        print(
            f"Validating prompt took {int((end_time_validating - start_time_validating) * 1000)} ms"
        )

    start_time_computing = time.perf_counter() if should_print_verbose_logs() else None

    # Normalize the node first
    normalized_node = normalize_prompt(elem)

    # Map priority levels to their token counts
    priority_levels_tokens: Dict[float, List[Union[int, str, Dict]]] = {}

    def compute_priority_levels_tokens(
        elem: PromptElement,
        parent_priority: float,
        mapping: Dict[float, List[Union[int, str, Dict]]],
    ) -> None:
        if isinstance(elem, (list, tuple)):
            for child in elem:
                compute_priority_levels_tokens(child, parent_priority, mapping)
            return

        if elem is None or elem is False:
            return

        if isinstance(elem, (str, int, float)):
            if parent_priority not in mapping:
                mapping[parent_priority] = []
            mapping[parent_priority].append(str(elem))
            return

        elem_type = elem.get("type")

        if elem_type == "scope":
            priority = compute_priority(elem, parent_priority)
            for child in elem["children"]:
                compute_priority_levels_tokens(child, priority, mapping)
            return

        if elem_type in ("function_definition", "tool_definition"):
            if parent_priority not in mapping:
                mapping[parent_priority] = []
            mapping[parent_priority].append(elem)
            return

    compute_priority_levels_tokens(normalized_node, BASE_PRIORITY, priority_levels_tokens)

    # Also compute priority levels normally for rendering
    priority_levels = set()
    compute_priority_levels(elem, BASE_PRIORITY, priority_levels)

    # Sort priority levels from highest to lowest
    sorted_priority_levels = sorted(list(priority_levels_tokens.keys()), reverse=True)

    if should_print_verbose_logs() and start_time_computing:
        end_time_computing = time.perf_counter()
        print(
            f"Computing priority levels took {int((end_time_computing - start_time_computing) * 1000)} ms"
        )

    # Traverse in reverse order
    running_token_sum = 0
    best_token_level = BASE_PRIORITY

    for priority_level in sorted_priority_levels:
        countables = priority_levels_tokens[priority_level]
        new_tokens = 0

        for countable in countables:
            if isinstance(countable, (int, str)):
                new_tokens += tokenizer.estimate_num_tokens_fast(str(countable))
            elif isinstance(countable, dict):
                if countable["type"] == "function_definition":
                    new_tokens += tokenizer.count_function_tokens_approx(countable)
                elif countable["type"] == "tool_definition":
                    new_tokens += tokenizer.count_tool_tokens_approx(countable["tool"])

        running_token_sum += new_tokens
        if running_token_sum > token_limit:
            break
        best_token_level = priority_level

    start_exact_count = time.perf_counter() if should_print_verbose_logs() else None

    prompt = render_with_level(elem, best_token_level, tokenizer, True)

    if prompt["prompt"] is None:
        raise ValueError("renderWithLevel returned None")

    if should_print_verbose_logs() and start_exact_count:
        end_exact_count = time.perf_counter()
        print(
            f"Computing exact token count took {int((end_exact_count - start_exact_count) * 1000)} ms"
        )

    duration = None
    if start_time:
        end_time = time.perf_counter()
        duration = int((end_time - start_time) * 1000)
        if duration > 100:
            print(
                f"Priompt WARNING: rendering prompt took {duration} ms, which is longer than the recommended maximum of 100 ms. Consider reducing the number of scopes you have."
            )

    return {
        "prompt": compact(prompt.get("prompt", "")),
        "tokens_reserved": prompt["empty_token_count"],
        "token_limit": token_limit,
        "tokenizer": tokenizer,
        "duration_ms": duration,
        "output_handlers": prompt["output_handlers"],
        "stream_handlers": prompt["stream_handlers"],
        "stream_response_object_handlers": prompt["stream_response_object_handlers"],
        "priority_cutoff": best_token_level,
        "config": prompt["config"],
    }


def render_backwards_linear_search(
    elem: PromptElement,
    options: RenderOptions,
) -> RenderOutput:
    """Render prompt using backwards linear search to find optimal priority cutoff."""
    start_time = time.perf_counter() if should_print_verbose_logs() else None
    elem = deepcopy(elem)

    if should_print_verbose_logs():
        start_time_validating = time.perf_counter()
        validate_unrendered_prompt(elem)
        end_time_validating = time.perf_counter()
        print(
            f"Validating prompt took {int((end_time_validating - start_time_validating) * 1000)} ms"
        )

    if should_print_verbose_logs():
        start_time_normalizing = time.perf_counter()
    normalized_elem = normalize_prompt(elem)
    if should_print_verbose_logs():
        end_time_normalizing = time.perf_counter()
        print(
            f"Normalizing prompt took {int((end_time_normalizing - start_time_normalizing) * 1000)} ms"
        )

    if should_print_verbose_logs():
        start_time_computing_priority_levels = time.perf_counter()

    priority_levels = set()
    compute_priority_levels(normalized_elem, BASE_PRIORITY, priority_levels)
    priority_levels.add(BASE_PRIORITY)
    sorted_priority_levels = sorted(list(priority_levels), reverse=True)

    if should_print_verbose_logs():
        end_time_computing_priority_levels = time.perf_counter()
        print(
            f"Computing priority levels took {int((end_time_computing_priority_levels - start_time_computing_priority_levels) * 1000)} ms"
        )

    if should_print_verbose_logs():
        start_time_rendering = time.perf_counter()

    prev_prompt = None
    prev_level = None
    this_prompt = None

    for level in sorted_priority_levels:
        this_prompt = render_with_level_and_count_tokens(
            normalized_elem, level, options["tokenizer"]
        )
        if is_chat_prompt(this_prompt["prompt"]):
            this_prompt["token_count"] += CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT
        if this_prompt["token_count"] + this_prompt["empty_token_count"] > options["token_limit"]:
            break
        prev_prompt = this_prompt
        prev_level = level

    if should_print_verbose_logs():
        end_time_rendering = time.perf_counter()
        print(f"Rendering prompt took {int((end_time_rendering - start_time_rendering) * 1000)} ms")

    if prev_prompt is None:
        raise TooManyTokensForBasePriority(
            f"Base prompt estimated token count is {this_prompt['token_count'] if this_prompt else None} with "
            f"{this_prompt['empty_token_count'] if this_prompt else None} tokens reserved, which is higher than "
            f"the limit {options['token_limit']}. This is probably a bug in the prompt — "
            "please add some priority levels to fix this."
        )

    if should_print_verbose_logs():
        start_exact_token_count = time.perf_counter()

    if prev_prompt["prompt"] is not None:
        exact_token_count = count_tokens_exact(
            options["tokenizer"],
            prev_prompt["prompt"],
            {"last_message_is_incomplete": options.get("last_message_is_incomplete", False)},
        )
        print(
            f"Discrepancy: (estimated token count) - (actual token count) = "
            f"{prev_prompt['token_count']} - {exact_token_count} = "
            f"{prev_prompt['token_count'] - exact_token_count}"
        )
        prev_prompt["token_count"] = exact_token_count
        if exact_token_count + prev_prompt["empty_token_count"] > options["token_limit"]:
            print(
                f"Actual token count is {exact_token_count} with {prev_prompt['empty_token_count']} "
                f"tokens reserved, which is higher than the limit {options['token_limit']}. "
                "This can possibly happen in rare circumstances, but should never be a problem in practice."
            )

    if should_print_verbose_logs():
        end_exact_token_count = time.perf_counter()
        print(
            f"Computing exact token count took {int((end_exact_token_count - start_exact_token_count) * 1000)} ms"
        )

    duration_ms = int((time.perf_counter() - start_time) * 1000) if start_time else None

    return {
        "prompt": compact(prev_prompt.get("prompt", "")),
        "token_count": prev_prompt["token_count"],
        "tokens_reserved": prev_prompt["empty_token_count"],
        "token_limit": options["token_limit"],
        "tokenizer": options["tokenizer"],
        "output_handlers": prev_prompt.get("output_handlers", []),
        "stream_handlers": prev_prompt.get("stream_handlers", []),
        "stream_response_object_handlers": prev_prompt.get("stream_response_object_handlers", []),
        "duration_ms": duration_ms,
        "priority_cutoff": prev_level if prev_level is not None else BASE_PRIORITY,
        "config": prev_prompt.get("config", {}),
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

    if isinstance(elem, (list, tuple)):
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
                if node.get("token_count") is not None:
                    total += node["token_count"]
                elif callable(node.get("token_function")):
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

    if message["role"] == "assistant" and message.get("function_call"):
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
    if message["role"] == "user" and message.get("images"):
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
            elif node["type"] == "function_definition":
                functions.append(node)
            elif node["type"] == "tool_definition":
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

        for child in node.get("children", []):
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

    def token_estimator(content: str) -> int:
        return tokenizer.estimate_tokens_using_char_count(content)[0]

    content_tokens = 0
    if is_chat_prompt(prompt):
        for msg in prompt["messages"]:
            if msg["role"] == "function":
                # Assume no extra tokens for lower bound
                content_tokens += token_estimator(msg["name"] + msg["content"])
            elif msg["role"] == "assistant" and msg.get("function_call"):
                content_tokens += token_estimator(
                    msg["function_call"]["name"]
                    + msg["function_call"]["arguments"]
                    + (msg.get("content", ""))
                )
            else:
                content = msg.get("content", "")
                if content:
                    content_tokens += token_estimator(prompt_string_to_string(content))
    elif is_plain_prompt(prompt):
        content_tokens = token_estimator(prompt_string_to_string(prompt))
    elif is_prompt_content(prompt):
        content_tokens = token_estimator(prompt_string_to_string(prompt["content"]))
    else:
        content_tokens = token_estimator(prompt_string_to_string(prompt["text"]))

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

    if isinstance(elem, (list, tuple)):
        return sum(get_prompt_element_node_count(p) for p in elem)

    if not isinstance(elem, dict):
        return 1

    node_type = elem.get("type")
    if node_type in (
        "function_definition",
        "tool_definition",
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
                raise ValueError("Function messages not supported in prompt_to_tokens yet")
            elif msg["role"] == "assistant" and msg.get("function_call"):
                raise ValueError("Function call messages not supported in prompt_to_tokens yet")
            elif msg.get("content") is None:
                raise ValueError("Message content cannot be None")

        chat_messages = [
            {
                "role": msg["role"],
                "name": msg.get("name"),
                "to": msg.get("to"),
                "content": prompt_string_to_string(msg["content"]),
            }
            for msg in messages
        ]
        return tokenizer.apply_chat_template_tokens(chat_messages)

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

    elif message["role"] == "assistant" and message.get("function_call"):
        function_tokens = count_function_call_message_tokens(message["function_call"], tokenizer)
        content_tokens = (
            num_tokens_prompt_string(message["content"], tokenizer)
            if message.get("content") is not None
            else 0
        )
        return function_tokens + content_tokens

    else:
        num_tokens = num_tokens_prompt_string(message.get("content", ""), tokenizer)
        if message["role"] == "user" and message.get("images"):
            for image in message["images"]:
                num_tokens += num_tokens_for_image(
                    image["image_url"]["dimensions"], image["image_url"]["detail"]
                )
        return num_tokens


def count_function_call_message_tokens(
    function_call: Dict[str, str], tokenizer: PriomptTokenizer
) -> int:
    """Count tokens in a function call message."""
    token_counter = tokenizer.num_tokens
    # Add constant factor for function call structure
    return token_counter(function_call["name"]) + token_counter(function_call["arguments"]) + 5


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
    token_counter = tokenizer.num_tokens
    if isinstance(prompt_string, list):
        return sum(token_counter(s) for s in prompt_string)
    return token_counter(prompt_string)


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
            elif msg["role"] == "assistant" and msg.get("function_call"):
                messages.append(
                    {
                        "role": msg["role"],
                        "content": prompt_string_to_string(msg.get("content", "")),
                        "function_call": msg["function_call"],
                    }
                )
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
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


def prompt_to_string_VULNERABLE_TO_PROMPT_INJECTION(
    prompt: RenderedPrompt,
    tokenizer: PriomptTokenizer,
) -> str:
    """
    Convert a prompt to string format. WARNING: Vulnerable to prompt injection!
    Do not use unless necessary as context (e.g. scraped docs) could include <|im_end|> strings and mess up the prompt.
    Also does not have <breaktoken> support.
    """
    if is_plain_prompt(prompt):
        # Encode as plain prompt
        if isinstance(prompt, list):
            return "".join(prompt)
        return prompt

    elif is_chat_prompt(prompt):
        parts = []
        for msg in prompt["messages"]:
            if msg["role"] == "function":
                raise ValueError(
                    "BUG!! promptToString got a chat prompt with a function message, which is not supported yet!"
                )
            elif msg["role"] == "assistant" and msg.get("function_call"):
                raise ValueError(
                    "BUG!! promptToString got a chat prompt with a function message, which is not supported yet!"
                )
            else:
                header_tokens = tokenizer.get_header_string_for_message(msg)
                if isinstance(msg.get("content"), list):
                    # Combine tokens to string array to handle images
                    new_content = "".join(str(c) for c in msg["content"] if isinstance(c, str))
                else:
                    new_content = msg.get("content", "")

                parts.append(
                    header_tokens
                    + prompt_to_string_VULNERABLE_TO_PROMPT_INJECTION(new_content, tokenizer)
                )

        return tokenizer.get_eos_token().join(parts)

    raise ValueError("BUG!! promptToString got an invalid prompt")


def flat(items: Iterable[Any], depth: int = 1) -> List[Any]:
    """Flatten a potentially nested list."""
    result = []
    for item in items:
        if isinstance(item, (list, tuple)) and depth > 0:
            result.extend(flat(item, depth - 1))
        else:
            result.append(item)
    return result


def is_dev() -> bool:
    return os.getenv("ENV") == "development"


def should_print_verbose_logs() -> bool:
    return is_dev() or os.getenv("PRIOMPT_VERBOSE_LOGS") == "true"


def validate_no_children_higher_priority_than_parent(
    elem: PromptElement, parent_priority: "Number" = BASE_PRIORITY
) -> None:
    if isinstance(elem, (list, tuple)):
        for child in elem:
            validate_no_children_higher_priority_than_parent(child, parent_priority=BASE_PRIORITY)
        return

    if elem is None or elem is False:
        return

    if isinstance(elem, (str, int, float)):
        return

    if not isinstance(elem, dict):
        return

    elem_type = elem.get("type")
    if elem_type in ("chat", "first"):
        for child in elem["children"]:
            validate_no_children_higher_priority_than_parent(child, parent_priority=BASE_PRIORITY)
        return

    if elem_type == "isolate":
        # Isolate is isolated, so don't pass parent priority
        validate_no_children_higher_priority_than_parent(elem["children"])
        return

    if elem_type in (
        "capture",
        "image",
        "breaktoken",
        "function_definition",
        "tool_definition",
        "empty",
        "config",
    ):
        return

    if elem_type == "scope":
        # Calculate priority for this scope
        priority = elem.get("absolute_priority", BASE_PRIORITY)
        if elem.get("relative_priority") is not None:
            priority = parent_priority + elem["relative_priority"]

        if priority > parent_priority:
            print(
                f"Priompt WARNING: child scope has a higher priority({priority}) than its parent({parent_priority}). "
                "This is discouraged, because the child will only be included if the parent is, and thus the "
                "effective priority of the child is just the parent's priority."
            )

        for child in elem["children"]:
            validate_no_children_higher_priority_than_parent(child, parent_priority=priority)
        return


def validate_not_both_absolute_and_relative_priority(elem: PromptElement) -> None:
    if isinstance(elem, (list, tuple)):
        for child in elem:
            validate_not_both_absolute_and_relative_priority(child)
        return

    if elem is None or elem is False:
        return

    if isinstance(elem, (str, int, float)):
        return

    elem_type = elem.get("type")
    if elem_type in ("chat", "isolate", "first"):
        for child in elem["children"]:
            validate_not_both_absolute_and_relative_priority(child)
        return

    if elem_type in (
        "capture",
        "breaktoken",
        "function_definition",
        "tool_definition",
        "image",
        "config",
        "empty",
    ):
        return

    if elem_type == "scope":
        if elem.get("absolute_priority") is not None and elem.get("relative_priority") is not None:
            print(
                "Priompt WARNING: scope has both absolute and relative priority. "
                "This is discouraged. Ignoring relative priority."
            )
        for child in elem["children"]:
            validate_not_both_absolute_and_relative_priority(child)
        return


def validate_unrendered_prompt(elem: PromptElement) -> None:
    if not is_dev():
        return

    validate_no_children_higher_priority_than_parent(elem)
    validate_not_both_absolute_and_relative_priority(elem)


def compute_priority(elem: PromptElement, parent_priority: "Number") -> "Number":
    priority = elem.get("absolute_priority")
    if priority is None:
        priority = parent_priority + (elem.get("relative_priority") or 0)

    return priority


def compute_priority_levels(
    elem: PromptElement,
    parent_priority: int,
    priority_levels: set["Number"],
) -> None:
    """Compute priority levels for all nodes in the prompt element tree."""
    if isinstance(elem, (list, tuple)):
        for child in elem:
            compute_priority_levels(child, parent_priority, priority_levels)
        return

    if elem is None or elem is False:
        return

    if isinstance(elem, str):
        return

    if isinstance(elem, (int, float)):
        return

    elem_type = elem.get("type")

    if elem_type in ("chat", "first"):
        for child in elem["children"]:
            compute_priority_levels(child, parent_priority, priority_levels)
    elif elem_type == "scope":
        priority = compute_priority(elem, parent_priority)
        priority_levels.add(priority)
        elem["absolute_priority"] = priority
        for child in elem["children"]:
            compute_priority_levels(child, priority, priority_levels)
    elif elem_type in (
        "image",
        "capture",
        "function_definition",
        "tool_definition",
        "breaktoken",
        "config",
        "empty",
        "isolate",  # Nothing happens because we fully re-render
        "normalized_string",
    ):
        return


def hydrate_isolates(elem: PromptElement, tokenizer: Any, should_build_source_map: bool) -> None:
    """Hydrate any isolated elements in the prompt."""
    if elem is None or elem is False:
        return None

    if isinstance(elem, (list, tuple)):
        for e in elem:
            hydrate_isolates(e, tokenizer, should_build_source_map)
        return None

    if isinstance(elem, str):
        return None

    if isinstance(elem, (int, float)):
        return None

    elem_type = elem.get("type")

    if elem_type == "first":
        return hydrate_isolates(elem["children"], tokenizer, should_build_source_map)

    if elem_type in (
        "capture",
        "empty",
        "image",
        "breaktoken",
        "config",
        "function_definition",
        "tool_definition",
    ):
        return None

    if elem_type == "isolate":
        if elem.get("cached_render_output") is None:
            elem["cached_render_output"] = render(
                elem["children"],
                {
                    "tokenizer": tokenizer,
                    "token_limit": elem["token_limit"],
                    "should_build_source_map": should_build_source_map,
                },
            )
        return None

    if elem_type in ("chat", "scope"):
        return hydrate_isolates(elem["children"], tokenizer, should_build_source_map)

    return None


def hydrate_empty_token_count(elem: PromptElement, tokenizer: Any) -> None:
    """Hydrate token counts for empty elements in the prompt."""
    if elem is None or elem is False:
        return None

    if isinstance(elem, (list, tuple)):
        for e in elem:
            hydrate_empty_token_count(e, tokenizer)
        return None

    if isinstance(elem, str):
        return None

    if isinstance(elem, (int, float)):
        return None

    elem_type = elem.get("type")

    if elem_type in ("chat", "scope", "first"):
        return hydrate_empty_token_count(elem["children"], tokenizer)

    if elem_type in (
        "capture",
        "image",
        "isolate",
        "breaktoken",
        "config",
        "tool_definition",
        "function_definition",
    ):
        return None

    if elem_type == "empty":
        if elem.get("token_count") is None:
            if not callable(elem.get("token_function")):
                raise ValueError(
                    "BUG!! empty token function is undefined. THIS SHOULD NEVER HAPPEN. BUG IN PRIOMPT."
                )
            elem["token_count"] = elem["token_function"](lambda s: tokenizer.num_tokens(s))
        return None

    return None


def render_with_level_and_count_tokens(
    elem: PromptElement, level: int, tokenizer: PriomptTokenizer
) -> Dict:
    """Render a prompt element with a priority level cutoff and count tokens."""
    if elem is None or elem is False:
        return {
            "prompt": None,
            "token_count": 0,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": {"max_response_tokens": None, "stop": None},
        }

    if isinstance(elem, (list, tuple)):
        results = [render_with_level_and_count_tokens(e, level, tokenizer) for e in elem]
        combined = {
            "prompt": None,
            "token_count": 0,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": {"max_response_tokens": None, "stop": None},
        }
        for r in results:
            combined["prompt"] = sum_prompts(combined["prompt"], r["prompt"])
            combined["token_count"] += r["token_count"]
            combined["empty_token_count"] += r["empty_token_count"]
            combined["output_handlers"].extend(r["output_handlers"])
            combined["stream_handlers"].extend(r["stream_handlers"])
            combined["stream_response_object_handlers"].extend(r["stream_response_object_handlers"])
            combined["config"] = merge_configs_in_place(combined["config"], r["config"])
        return combined

    if isinstance(elem, str):
        token_count = tokenizer.num_tokens(elem)
        return {
            "prompt": elem,
            "token_count": token_count,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": {"max_response_tokens": None, "stop": None},
        }

    if isinstance(elem, (int, float)):
        text = str(elem)
        token_count = tokenizer.num_tokens(text)
        return {
            "prompt": text,
            "token_count": token_count,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": {"max_response_tokens": None, "stop": None},
        }

    elem_type = elem.get("type")

    if elem_type == "first":
        for child in elem["children"]:
            if child.get("absolute_priority") is None:
                raise ValueError(
                    "BUG!! compute_priority_levels should have set absolute_priority for all children of first"
                )
            if child["absolute_priority"] >= level:
                if callable(elem.get("on_include")):
                    elem["on_include"]()
                return render_with_level_and_count_tokens(child, level, tokenizer)
        return {
            "prompt": None,
            "token_count": 0,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": {"max_response_tokens": None, "stop": None},
        }

    elif elem_type == "capture":
        result = {
            "prompt": None,
            "token_count": 0,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": {"max_response_tokens": None, "stop": None},
        }
        if callable(elem.get("on_output")):
            result["output_handlers"].append(elem["on_output"])
        if callable(elem.get("on_stream")):
            result["stream_handlers"].append(elem["on_stream"])
        if callable(elem.get("on_stream_response_object")):
            result["stream_response_object_handlers"].append(elem["on_stream_response_object"])
        return result

    elif elem_type == "config":
        return {
            "prompt": None,
            "token_count": 0,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": elem,
        }

    elif elem_type == "breaktoken":
        return {
            "prompt": ["", ""],
            "token_count": 0,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": {"max_response_tokens": None, "stop": None},
        }

    elif elem_type == "empty":
        if elem.get("token_count") is None:
            if not callable(elem.get("token_function")):
                raise ValueError(
                    "BUG!! empty token function is undefined. THIS SHOULD NEVER HAPPEN. BUG IN PRIOMPT."
                )
            elem["token_count"] = elem["token_function"](lambda s: tokenizer.num_tokens(s))
        return {
            "prompt": None,
            "token_count": 0,
            "empty_token_count": elem["token_count"],
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": {"max_response_tokens": None, "stop": None},
        }

    elif elem_type == "isolate":
        if not elem.get("cached_render_output"):
            elem["cached_render_output"] = render(
                elem["children"],
                {
                    "tokenizer": tokenizer,
                    "token_limit": elem["token_limit"],
                },
            )
        return {
            "prompt": elem["cached_render_output"]["prompt"],
            "token_count": elem["cached_render_output"]["token_count"],
            "empty_token_count": elem["cached_render_output"]["tokens_reserved"],
            "output_handlers": elem["cached_render_output"]["output_handlers"],
            "stream_handlers": elem["cached_render_output"]["stream_handlers"],
            "stream_response_object_handlers": elem["cached_render_output"][
                "stream_response_object_handlers"
            ],
            "config": {"max_response_tokens": None, "stop": None},
        }

    elif elem_type == "scope":
        if elem.get("absolute_priority") is None:
            raise ValueError(
                "BUG!! compute_priority_levels should have set absolute_priority for all scopes"
            )
        if elem["absolute_priority"] >= level:
            return render_with_level_and_count_tokens(elem["children"], level, tokenizer)
        return {
            "prompt": None,
            "token_count": 0,
            "empty_token_count": 0,
            "output_handlers": [],
            "stream_handlers": [],
            "stream_response_object_handlers": [],
            "config": {"max_response_tokens": None, "stop": None},
        }

    return None


class TokenLimitExceeded(ValueError): ...


def render_with_level_and_early_exit_with_token_estimation(
    elem: PromptElement, level: int, tokenizer: Any, token_limit: int
) -> RenderOutput:
    """
    Render a prompt element with a priority level cutoff, exiting early if token limit exceeded.
    Returns rendered prompt and empty token count.
    """
    # Initialize state
    prompt = None
    empty_token_count = 0

    def render_in_place(elem: PromptElement) -> None:
        nonlocal prompt, empty_token_count

        if elem is None or elem is False:
            return

        if isinstance(elem, (list, tuple)):
            for e in elem:
                render_in_place(e)
            # Check token limit
            lower_bound = estimate_lower_bound_tokens_for_prompt(prompt, tokenizer)
            if lower_bound > token_limit:
                raise TokenLimitExceeded("Token limit exceeded!")
            return

        if isinstance(elem, (str, int, float)):
            prompt = sum_prompts(prompt, str(elem))
            return

        elem_type = elem.get("type")

        if elem_type == "first":
            for child in elem["children"]:
                if child.get("absolute_priority") is None:
                    raise ValueError(
                        "BUG!! compute_priority_levels should have set absolute_priority for all children of first"
                    )
                if child["absolute_priority"] >= level:
                    render_in_place(child)
                    return
            return

        if elem_type in ("capture", "config"):
            return

        if elem_type == "breaktoken":
            prompt = sum_prompts(prompt, ["", ""])
            return

        if elem_type == "empty":
            if elem.get("token_count") is None:
                raise ValueError(
                    "BUG!! empty token count is undefined. THIS SHOULD NEVER HAPPEN. "
                    "BUG IN PRIOMPT. Empty token count should've been hydrated first!"
                )
            empty_token_count += elem["token_count"]
            return

        if elem_type == "function_definition":
            prompt = sum_prompts(
                prompt,
                {
                    "type": "text",
                    "text": "",
                    "functions": [
                        {
                            "name": elem["name"],
                            "description": elem["description"],
                            "parameters": elem["parameters"],
                        }
                    ],
                },
            )
            return

        if elem_type == "tool_definition":
            prompt = sum_prompts(
                prompt,
                {
                    "type": "text",
                    "text": "",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": elem["tool"]["function"]["name"],
                                "description": elem["tool"]["function"]["description"],
                                "parameters": elem["tool"]["function"]["parameters"],
                            },
                        }
                    ],
                },
            )
            return

        if elem_type == "image":
            mime_type = get_image_mime_type(elem["bytes"])
            base64_string = base64.b64encode(elem["bytes"]).decode("utf-8")
            prompt = sum_prompts(
                prompt,
                {
                    "type": "prompt_content",
                    "content": [],
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_string}",
                                "detail": elem["detail"],
                                "dimensions": elem["dimensions"],
                            },
                        }
                    ],
                },
            )
            return

        if elem_type == "isolate":
            if not elem.get("cached_render_output"):
                raise ValueError(
                    "BUG!! Isolates should have been hydrated before calling render_with_level_and_early_exit_with_token_estimation"
                )
            prompt = sum_prompts(prompt, elem["cached_render_output"]["prompt"])
            empty_token_count += elem["cached_render_output"]["tokens_reserved"]
            return

        if elem_type == "scope":
            if elem.get("absolute_priority") is None:
                raise ValueError(
                    "BUG!! compute_priority_levels should have set absolute_priority for all scopes"
                )
            if elem["absolute_priority"] >= level:
                render_in_place(elem["children"])
            return

        if elem_type == "chat":
            # Handle chat messages recursively
            p = render_with_level_and_early_exit_with_token_estimation(
                elem["children"], level, tokenizer, token_limit
            )

            if is_chat_prompt(p["prompt"]):
                raise ValueError(
                    "Incorrect prompt: we have nested chat messages, which is not allowed!"
                )

            message = {}
            if elem["role"] == "user":
                if is_prompt_content(p["prompt"]):
                    message = {
                        "role": elem["role"],
                        "name": elem.get("name"),
                        "to": elem.get("to"),
                        "content": p["prompt"]["content"],
                        "images": p["prompt"]["images"],
                    }
                else:
                    message = {
                        "role": elem["role"],
                        "to": elem.get("to"),
                        "name": elem.get("name"),
                        "content": get_chat_prompt_content(p),
                    }
            elif elem["role"] == "system":
                if is_prompt_content(p["prompt"]):
                    raise ValueError("Did not expect images in system message")
                message = {
                    "role": elem["role"],
                    "to": elem.get("to"),
                    "name": elem.get("name"),
                    "content": get_chat_prompt_content(p),
                }
            elif elem["role"] == "assistant":
                if is_prompt_content(p["prompt"]):
                    raise ValueError("Did not expect images in assistant message")
                if elem.get("function_call"):
                    message = {
                        "role": elem["role"],
                        "to": elem.get("to"),
                        "content": get_chat_prompt_content(p),
                        "function_call": elem["function_call"],
                    }
                elif elem.get("tool_calls"):
                    message = {
                        "role": elem["role"],
                        "to": elem.get("to"),
                        "content": get_chat_prompt_content(p),
                        "tool_calls": elem["tool_calls"],
                    }
                else:
                    message = {
                        "role": elem["role"],
                        "to": elem.get("to"),
                        "content": get_chat_prompt_content(p),
                    }
            elif elem["role"] in ("function", "tool"):
                if is_prompt_content(p["prompt"]):
                    raise ValueError(f"Did not expect images in {elem['role']} message")
                message = {
                    "role": elem["role"],
                    "name": elem.get("name"),
                    "to": elem.get("to"),
                    "content": get_chat_prompt_content(p),
                }
            else:
                raise ValueError(f"BUG!! Invalid role {elem['role']}")

            prompt = sum_prompts(
                prompt,
                {
                    "type": "chat",
                    "messages": [message],
                    "functions": p["prompt"].get("functions")
                    if prompt_has_functions(p["prompt"])
                    else None,
                    "tools": p["prompt"].get("tools") if prompt_has_tools(p["prompt"]) else None,
                },
            )
            empty_token_count += p["empty_token_count"]
            return

    render_in_place(elem)
    return {
        "prompt": prompt,
        "empty_token_count": empty_token_count,
    }


def normalize_source_map(source_map: Any) -> Any:
    """Normalize a source map by flattening single-child nodes and removing empty children arrays."""
    if source_map.get("children") is None:
        return source_map

    if len(source_map["children"]) == 0:
        source_map.pop("children")
        return source_map

    if len(source_map["children"]) == 1:
        child = source_map["children"][0]
        return normalize_source_map(
            {
                "name": f"{source_map['name']}.{child['name']}",
                "children": child.get("children"),
                "start": source_map["start"],
                "end": source_map["end"],
            }
        )
    else:
        return {
            **source_map,
            "children": [normalize_source_map(child) for child in source_map["children"]],
        }


def render_with_level(
    elem: PromptElement,
    level: int,
    tokenizer: PriomptTokenizer,
    call_ejected_callback: bool = False,
    source_info: Optional[Dict] = None,
) -> Dict:
    """Render a prompt element at a given priority level."""
    # Initialize result object
    result = {
        "prompt": None,
        "empty_token_count": 0,
        "output_handlers": [],
        "stream_handlers": [],
        "stream_response_object_handlers": [],
        "config": {"max_response_tokens": None, "stop": None},
    }

    def render_in_place(
        elem: PromptElement, source_info: Optional[Dict] = None
    ) -> Optional[SourceMap]:
        """Render element in place, modifying result object."""
        if elem is None or elem is False:
            return None

        if isinstance(elem, (list, tuple)):
            source_maps = [
                render_in_place(
                    e,
                    {
                        "name": str(i),
                        "is_last": (source_info is None or source_info.get("is_last", True))
                        and i == len(elem) - 1,
                    }
                    if source_info
                    else None,
                )
                for i, e in enumerate(elem)
            ]
            return merge_source_maps(source_maps, source_info["name"]) if source_info else None

        if isinstance(elem, str):
            result["prompt"] = sum_prompts(result["prompt"], elem)
            if not source_info:
                return None
            return {
                "name": source_info["name"],
                "children": None,
                "start": 0,
                "end": len(elem),
                "string": elem,
            }

        if isinstance(elem, (int, float)):
            prompt = str(elem)
            result["prompt"] = sum_prompts(result["prompt"], prompt)
            if not source_info:
                return None
            return {
                "name": source_info["name"],
                "start": 0,
                "end": len(prompt),
                "string": prompt,
            }

        if not isinstance(elem, dict):
            return None

        node_type = elem.get("type")

        if node_type == "first":
            for i, child in enumerate(elem["children"]):
                if child.get("absolute_priority") is None:
                    raise ValueError(
                        "BUG!! compute_priority_levels should have set absolute_priority for all children of first"
                    )
                if child["absolute_priority"] >= level:
                    if callable(elem.get("on_include")):
                        elem["on_include"]()
                    return render_in_place(
                        child,
                        {
                            "name": f"{source_info['name']}.{i}",
                            "is_last": (source_info is None or source_info.get("is_last", True))
                            and i == len(elem["children"]) - 1,
                        }
                        if source_info
                        else None,
                    )
                elif call_ejected_callback:
                    recursively_eject(child)
            return None

        elif node_type == "capture":
            if callable(elem.get("on_output")):
                result["output_handlers"].append(elem["on_output"])
            if callable(elem.get("on_stream")):
                result["stream_handlers"].append(elem["on_stream"])
            if callable(elem.get("on_stream_response_object")):
                result["stream_response_object_handlers"].append(elem["on_stream_response_object"])
            return None

        elif node_type == "config":
            result["config"] = merge_configs_in_place(result["config"], elem)
            return None

        elif node_type == "breaktoken":
            result["prompt"] = sum_prompts(result["prompt"], ["", ""])
            return None

        elif node_type == "empty":
            if elem.get("token_count") is None:
                raise ValueError("BUG!! empty token count is undefined. THIS SHOULD NEVER HAPPEN.")
            result["empty_token_count"] += elem["token_count"]
            return None

        elif node_type == "function_definition":
            result["prompt"] = sum_prompts(
                result["prompt"],
                {
                    "type": "text",
                    "text": "",
                    "functions": [
                        {
                            "name": elem["name"],
                            "description": elem["description"],
                            "parameters": elem["parameters"],
                        }
                    ],
                },
            )
            return None

        elif node_type == "tool_definition":
            result["prompt"] = sum_prompts(
                result["prompt"],
                {
                    "type": "text",
                    "text": "",
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": elem["tool"]["function"]["name"],
                                "description": elem["tool"]["function"]["description"],
                                "parameters": elem["tool"]["function"]["parameters"],
                            },
                        }
                    ],
                },
            )
            return None

        elif node_type == "isolate":
            if not elem.get("cached_render_output"):
                raise ValueError(
                    "BUG!! Isolates should have been hydrated before calling render_with_level"
                )
            cached_output = elem["cached_render_output"]
            result["prompt"] = sum_prompts(result["prompt"], cached_output["prompt"])
            result["empty_token_count"] += cached_output["tokens_reserved"]
            result["output_handlers"].extend(cached_output["output_handlers"])
            result["stream_handlers"].extend(cached_output["stream_handlers"])
            result["stream_response_object_handlers"].extend(
                cached_output["stream_response_object_handlers"]
            )
            return cached_output.get("source_map")

        elif node_type == "scope":
            if elem.get("absolute_priority") is None:
                raise ValueError(
                    "BUG!! compute_priority_levels should have set absolute_priority for all scopes"
                )
            if elem["absolute_priority"] >= level:
                if callable(elem.get("on_include")):
                    elem["on_include"]()

                source_map = render_in_place(
                    elem["children"],
                    {"name": elem.get("name", "scope"), "is_last": source_info["is_last"]}
                    if source_info
                    else None,
                )
                if source_map is None or source_info is None:
                    return None
                return {
                    "name": source_info["name"],
                    "children": [source_map],
                    "start": 0,
                    "end": source_map["end"],
                }
            elif call_ejected_callback:
                recursively_eject(elem)
            return None

        elif node_type == "image":
            base64_encoded_bytes = base64.b64encode(elem["bytes"]).decode("utf-8")
            media_type = get_image_mime_type(elem["bytes"])
            result["prompt"] = sum_prompts(
                result["prompt"],
                {
                    "type": "prompt_content",
                    "content": [],
                    "images": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{base64_encoded_bytes}",
                                "detail": elem["detail"],
                                "dimensions": elem["dimensions"],
                            },
                        }
                    ],
                },
            )
            return None

        elif node_type == "chat":
            p = render_with_level(
                elem["children"],
                level,
                tokenizer,
                call_ejected_callback,
                {
                    "name": f"{elem['role']}-message",
                    # Set is_last based on source_info if available
                    "is_last": source_info["is_last"] if source_info else False,
                }
                if source_info
                else None,
            )

            if is_chat_prompt(p["prompt"]):
                raise ValueError(
                    "Incorrect prompt: we have nested chat messages, which is not allowed!"
                )

            if elem["role"] == "user":
                if is_prompt_content(p["prompt"]):
                    message = {
                        "role": elem["role"],
                        "name": elem.get("name"),
                        "to": elem.get("to"),
                        "content": p["prompt"]["content"],
                        "images": p["prompt"]["images"],
                    }
                else:
                    message = {
                        "role": elem["role"],
                        "name": elem.get("name"),
                        "to": elem.get("to"),
                        "content": get_chat_prompt_content(p),
                    }
            elif elem["role"] == "system":
                if is_prompt_content(p["prompt"]):
                    raise ValueError("Did not expect images in system message")
                message = {
                    "role": elem["role"],
                    "to": elem.get("to"),
                    "name": elem.get("name"),
                    "content": get_chat_prompt_content(p),
                }
            elif elem["role"] == "assistant":
                if is_prompt_content(p["prompt"]):
                    raise ValueError("Did not expect images in assistant message")
                if elem.get("function_call"):
                    message = {
                        "role": elem["role"],
                        "to": elem.get("to"),
                        "content": get_chat_prompt_content(p),
                        "function_call": elem["function_call"],
                    }
                elif elem.get("tool_calls"):
                    message = {
                        "role": elem["role"],
                        "to": elem.get("to"),
                        "content": get_chat_prompt_content(p),
                        "tool_calls": elem["tool_calls"],
                    }
                else:
                    message = {
                        "role": elem["role"],
                        "to": elem.get("to"),
                        "content": get_chat_prompt_content(p),
                    }
            elif elem["role"] == "function":
                if is_prompt_content(p["prompt"]):
                    raise ValueError("Did not expect images in function message")
                message = {
                    "role": elem["role"],
                    "to": elem.get("to"),
                    "name": elem.get("name"),
                    "content": get_chat_prompt_content(p),
                }
            elif elem["role"] == "tool":
                if is_prompt_content(p["prompt"]):
                    raise ValueError("Did not expect images in tool message")
                message = {
                    "role": elem["role"],
                    "name": elem.get("name"),
                    "to": elem.get("to"),
                    "content": get_chat_prompt_content(p),
                }
            else:
                raise ValueError(f"BUG!! Invalid role {elem['role']}")

            source_map = p.get("source_map")

            if source_info and source_map:
                source_map = get_source_map_for_chat(message, tokenizer, source_map, source_info)

            result["prompt"] = sum_prompts(
                result["prompt"],
                {
                    "type": "chat",
                    "messages": [message],
                    "functions": p["prompt"].get("functions")
                    if prompt_has_functions(p["prompt"])
                    else None,
                    "tools": p["prompt"].get("tools") if prompt_has_tools(p["prompt"]) else None,
                },
            )
            result["empty_token_count"] += p["empty_token_count"]
            result["output_handlers"].extend(p["output_handlers"])
            result["stream_handlers"].extend(p["stream_handlers"])
            result["stream_response_object_handlers"].extend(p["stream_response_object_handlers"])
            return source_map

    source_map = render_in_place(elem, source_info)
    result["source_map"] = source_map
    return result


def count_tokens_exact(
    tokenizer: PriomptTokenizer,
    prompt: RenderedPrompt,
    options: Optional[Dict[str, bool]] = None,
) -> int:
    """Count exact number of tokens in a prompt."""
    if options is None:
        options = {}

    tokens = 0
    if is_plain_prompt(prompt):
        tokens += num_tokens_prompt_string(prompt, tokenizer)
    elif is_chat_prompt(prompt):
        msg_tokens = [count_msg_tokens(msg, tokenizer) for msg in prompt["messages"]]
        # docs here: https://platform.openai.com/docs/guides/chat/introduction
        tokens += (
            sum(msg_tokens)
            + CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR * len(prompt["messages"])
            + CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT
        )
        if options.get("last_message_is_incomplete"):
            tokens = tokens - (CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT + 1)
    elif is_prompt_content(prompt):
        # We count the tokens of each text element
        tokens += tokenizer.num_tokens(prompt["content"])
        if prompt.get("images"):
            for image in prompt["images"]:
                tokens += num_tokens_for_image(
                    image["image_url"]["dimensions"], image["image_url"]["detail"]
                )
    else:
        tokens += tokenizer.num_tokens(prompt["text"])

    if prompt_has_functions(prompt):
        # we assume an extra 2 tokens per function
        function_tokens = [
            count_function_tokens(func, tokenizer) + 2 for func in prompt["functions"]
        ]
        tokens += sum(function_tokens)

    if prompt_has_tools(prompt):
        # we assume an extra 2 tokens per tool
        tool_tokens = [count_tool_tokens(tool, tokenizer) + 2 for tool in prompt["tools"]]
        tokens += sum(tool_tokens)

    return tokens


def merge_configs_in_place(a: ConfigProps, b: ConfigProps) -> ConfigProps:
    """Merge config b into config a, modifying a in place."""
    for key in b:
        if a.get(key) is None:
            a[key] = b[key]
    return a


def get_source_map_for_chat(
    message: ChatPromptMessage,
    tokenizer: PriomptTokenizer,
    source_map: SourceMap,
    source_info: Dict,
) -> SourceMap:
    """Generate source map for a chat message."""
    if message["role"] == "function":
        # Not implemented for functions yet
        header_string = ""
    else:
        header_string = tokenizer.get_header_string_for_message(message)

    children = [
        {"name": "header", "children": [], "start": 0, "end": len(header_string)},
        {**source_map, "start": len(header_string), "end": source_map["end"] + len(header_string)},
    ]

    if source_info.get("is_last") is None:
        raise ValueError("BUG!! source.is_last should not be None")

    if (not source_info["is_last"]) and tokenizer.should_add_eos_token_to_each_message:
        children.append(
            {
                "name": "eos",
                "children": [],
                "start": children[-1]["end"],
                "end": children[-1]["end"] + len(tokenizer.get_eos_token()),
            }
        )

    return {"name": "chat", "children": children, "start": 0, "end": children[-1]["end"]}


def recursively_eject(elem: PromptElement) -> None:
    """Recursively call onEject handlers on prompt elements."""
    if elem is None or isinstance(elem, (bool, str, int)):
        return

    if isinstance(elem, (list, tuple)):
        for e in elem:
            recursively_eject(e)
        return

    if isinstance(elem, dict):
        if callable(elem.get("on_eject")):
            elem["on_eject"]()

        if isinstance(elem.get("children"), list):
            for child in elem["children"]:
                recursively_eject(child)


def num_tokens_prompt_string_fast(prompt: str, tokenizer: PriomptTokenizer) -> int:
    if isinstance(prompt, list):
        tokens = 0
        for p in prompt:
            tokens += num_tokens_prompt_string_fast(p, tokenizer)
        return tokens
    return tokenizer.estimate_tokens_fast(prompt)


def count_msg_tokens_fast(message: ChatPromptMessage, tokenizer: PriomptTokenizer) -> int:
    if message["role"] == "function":
        # Add extra 2 tokens for good measure
        return (
            tokenizer.estimate_tokens_fast(message["name"])
            + num_tokens_prompt_string_fast(message["content"], tokenizer)
            + 2
        )

    elif message["role"] == "assistant" and message.get("function_call"):
        function_tokens = count_function_call_message_tokens(message["function_call"], tokenizer)
        content_tokens = (
            num_tokens_prompt_string_fast(message["content"], tokenizer)
            if message.get("content") is not None
            else 0
        )
        return function_tokens + content_tokens

    else:
        num_tokens = num_tokens_prompt_string_fast(message.get("content", ""), tokenizer)
        if message["role"] == "user" and message.get("images"):
            for image in message["images"]:
                num_tokens += num_tokens_for_image(
                    image["image_url"]["dimensions"], image["image_url"]["detail"]
                )
        return num_tokens

    stringified_function = ujson.dumps(
        {
            "name": func["name"],
            "description": func["description"],
            "parameters": func["parameters"],
        },
        indent=2,
    )
    # Multiply by 1.5 and add 10 to be safe until more testing
    raw = tokenizer.num_tokens(stringified_function)
    return math.ceil(raw * 1.5) + 10


def count_tool_tokens_approx(tool: ToolDefinition, tokenizer: PriomptTokenizer) -> int:
    stringified_tool = ujson.dumps(
        {
            "name": tool["function"]["name"],
            "description": tool["function"]["description"],
            "parameters": tool["function"]["parameters"],
        },
        indent=2,
    )
    # Multiply by 1.5 and add 10 to be safe until more testing
    raw = tokenizer.num_tokens(stringified_tool)
    return math.ceil(raw * 1.5) + 10


def count_tokens_approx_fast(
    tokenizer: PriomptTokenizer,
    prompt: RenderedPrompt,
    options: Optional[Dict[str, bool]] = None,
) -> int:
    """
    Fast approximate token counting for rendered prompts.
    Only use when accuracy is not critical.
    """
    if options is None:
        options = {}

    tokens = 0
    if is_plain_prompt(prompt):
        tokens += num_tokens_prompt_string_fast(prompt, tokenizer)
    elif is_chat_prompt(prompt):
        msg_tokens = [count_msg_tokens_fast(msg, tokenizer) for msg in prompt["messages"]]
        # docs: https://platform.openai.com/docs/guides/chat/introduction
        tokens += (
            sum(msg_tokens)
            + CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR * len(prompt["messages"])
            + CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT
        )
        if options.get("last_message_is_incomplete"):
            tokens = tokens - (CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT + 1)
    elif is_prompt_content(prompt):
        # Count tokens for each text element
        tokens += num_tokens_prompt_string_fast(prompt["content"], tokenizer)
        if prompt.get("images"):
            for image in prompt["images"]:
                tokens += num_tokens_for_image(
                    image["image_url"]["dimensions"], image["image_url"]["detail"]
                )
    else:
        tokens += num_tokens_prompt_string_fast(prompt["text"], tokenizer)

    if prompt_has_functions(prompt):
        # Assume extra 2 tokens per function
        function_tokens = [
            count_function_tokens_approx(func, tokenizer) + 2 for func in prompt["functions"]
        ]
        tokens += sum(function_tokens)

    if prompt_has_tools(prompt):
        # Assume extra 2 tokens per tool
        tool_tokens = [count_tool_tokens_approx(tool, tokenizer) + 2 for tool in prompt["tools"]]
        tokens += sum(tool_tokens)

    return tokens
