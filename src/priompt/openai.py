from __future__ import annotations
from typing import Dict, List, Literal, Optional, TypedDict, Union
from typing_extensions import NotRequired

# Constants
CHATML_PROMPT_EXTRA_TOKEN_COUNT_LINEAR_FACTOR = 4
CHATML_PROMPT_EXTRA_TOKEN_COUNT_CONSTANT = 3


class Content(TypedDict):
    type: Literal["text", "image_url"]
    text: NotRequired[str]
    image_url: NotRequired[Dict[str, Union[str, Dict[str, int]]]]


class ChatCompletionFunctions(TypedDict):
    name: str
    description: str
    parameters: Dict


class ChatCompletionRequestMessageFunctionCall(TypedDict):
    name: str
    arguments: str


class ChatCompletionRequestMessage(TypedDict):
    role: str  # system, user, assistant, function, or tool
    content: Optional[Union[str, List[Content]]]
    name: NotRequired[str]
    function_call: NotRequired[ChatCompletionRequestMessageFunctionCall]


class ChatCompletionRequestMessageWithoutImages(ChatCompletionRequestMessage):
    pass


class CreateChatCompletionRequest(TypedDict):
    model: str
    messages: List[ChatCompletionRequestMessage]
    functions: NotRequired[List[ChatCompletionFunctions]]
    function_call: NotRequired[Dict]
    temperature: NotRequired[Optional[float]]
    top_p: NotRequired[Optional[float]]
    n: NotRequired[Optional[int]]
    stream: NotRequired[Optional[bool]]
    stop: NotRequired[Union[str, List[str]]]
    max_tokens: NotRequired[int]
    presence_penalty: NotRequired[Optional[float]]
    frequency_penalty: NotRequired[Optional[float]]
    logit_bias: NotRequired[Optional[Dict]]
    user: NotRequired[str]
    speculation: NotRequired[Union[str, List[int]]]


class CreateCompletionRequest(TypedDict):
    model: str
    prompt: NotRequired[Optional[Union[str, List[str], List[int], List[List[int]]]]]
    suffix: NotRequired[Optional[str]]
    max_tokens: NotRequired[Optional[int]]
    temperature: NotRequired[Optional[float]]
    top_p: NotRequired[Optional[float]]
    n: NotRequired[Optional[int]]
    stream: NotRequired[Optional[bool]]
    logprobs: NotRequired[Optional[int]]
    echo: NotRequired[Optional[bool]]
    stop: NotRequired[Optional[Union[str, List[str]]]]
    presence_penalty: NotRequired[Optional[float]]
    frequency_penalty: NotRequired[Optional[float]]
    best_of: NotRequired[Optional[int]]
    logit_bias: NotRequired[Optional[Dict]]
    user: NotRequired[str]
    speculation: NotRequired[Union[str, List[int]]]


class ChatCompletionResponseMessage(TypedDict):
    role: str
    content: NotRequired[str]
    function_call: NotRequired[ChatCompletionRequestMessageFunctionCall]
    tool_calls: NotRequired[List[ChatCompletionRequestMessageToolCall]]


class ChatCompletionRequestMessageFunctionToolCall(TypedDict):
    type: Literal["function"]
    function: Dict[str, str]


class ChatCompletionRequestMessageToolCall(TypedDict, total=False):
    id: str
    index: int
    type: Literal["function"]
    function: Dict[str, str]


class StreamChatCompletionResponseChoicesInner(TypedDict):
    index: int
    delta: ChatCompletionResponseMessage
    finish_reason: Optional[str]


class StreamChatCompletionResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: List[StreamChatCompletionResponseChoicesInner]


def has_images(message: ChatCompletionRequestMessage) -> bool:
    return isinstance(message.get("content"), list)


def has_no_images(message: ChatCompletionRequestMessage) -> bool:
    return isinstance(message.get("content"), str)


def approximate_tokens_using_bytecount(text: str, tokenizer: str) -> int:
    byte_length = len(text.encode())
    if tokenizer in ("cl100k_base", "o200k_base"):
        return byte_length // 4
    return byte_length // 3
