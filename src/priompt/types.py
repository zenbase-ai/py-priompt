from __future__ import annotations
from typing_extensions import NotRequired, TypeAlias
from typing import Literal

from beartype.typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
    TypedDict,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    from .openai import ChatCompletionResponseMessage, StreamChatCompletionResponse
    from .tokenizer import PriomptTokenizer

T = TypeVar("T")
DictT = TypeVar("DictT", bound=Dict[str, Any])
ReturnT = TypeVar("ReturnT")
PropsT = TypeVar("PropsT", bound=Dict[str, Any])

OutputHandler: TypeAlias = Callable[[T, Optional[Dict[str, int]]], None]
NodeCallback: TypeAlias = Callable[[], None]
Number: TypeAlias = Union[int, float]
JSONSchema7: TypeAlias = Dict  # JSON Schema type (simplified)


# Basic types for messages
class FunctionBody(TypedDict):
    name: str
    description: str
    parameters: JSONSchema7


class First(TypedDict):
    type: Literal["first"]
    children: List[Scope]
    on_eject: NotRequired[Callable[[], None]]
    on_include: NotRequired[Callable[[], None]]


class Empty(TypedDict):
    type: Literal["empty"]
    token_count: Optional[int]
    token_function: Optional[Callable[[Callable[[str], int]], int]]


class BreakToken(TypedDict):
    type: Literal["breaktoken"]


class CaptureProps(TypedDict):
    on_output: NotRequired[OutputHandler["ChatCompletionResponseMessage"]]
    on_stream: NotRequired[OutputHandler[Iterable["ChatCompletionResponseMessage"]]]
    on_stream_response_object: NotRequired[OutputHandler[Iterable["StreamChatCompletionResponse"]]]


class Capture(CaptureProps):
    type: Literal["capture"]


class ConfigProps(TypedDict):
    max_response_tokens: NotRequired[Union[int, Literal["tokens_reserved", "tokens_remaining"]]]
    stop: NotRequired[Union[str, List[str]]]


class IsolateProps(TypedDict):
    token_limit: int


class ImageDimensions(TypedDict):
    width: Number
    height: Number


class Isolate(IsolateProps):
    type: Literal["isolate"]
    children: List[Node]
    cached_render_output: NotRequired[RenderOutput]


class ChatImage(TypedDict):
    type: Literal["image"]
    bytes: bytes
    detail: Literal["low", "high", "auto"]
    dimensions: ImageDimensions


class ScopeProps(TypedDict):
    name: NotRequired[str]
    p: NotRequired[Number]
    prel: NotRequired[Number]
    on_eject: NotRequired[Callable[[], None]]
    on_include: NotRequired[Callable[[], None]]


class Scope(TypedDict):
    type: Literal["scope"]
    children: List[Node]
    absolute_priority: Optional[Number]
    relative_priority: Optional[Number]
    name: NotRequired[str]
    on_eject: NotRequired[Callable[[], None]]
    on_include: NotRequired[Callable[[], None]]


class Config(ConfigProps, Scope):
    type: Literal["config"]


class ChatUserSystemMessage(Scope):
    type: Literal["chat"]
    role: Literal["user", "system"]
    name: NotRequired[str]
    to: NotRequired[str]
    children: List[Node]


class ChatAssistantFunctionToolCall(Scope):
    type: Literal["function"]
    function: Dict[str, str]  # name and arguments as json string


class ChatAssistantMessage(Scope):
    type: Literal["chat"]
    role: Literal["assistant"]
    to: NotRequired[str]
    children: List[Node]
    function_call: NotRequired[Dict[str, str]]  # name and arguments
    tool_calls: NotRequired[List[Dict[str, Union[int, str, ChatAssistantFunctionToolCall]]]]


class ChatFunctionResultMessage(Scope):
    type: Literal["chat"]
    role: Literal["function"]
    name: str
    to: NotRequired[str]
    children: List[Node]


class ChatToolResultMessage(Scope):
    type: Literal["chat"]
    role: Literal["tool"]
    name: str
    to: NotRequired[str]
    children: List[Node]


ChatMessage = Union[
    ChatUserSystemMessage,
    ChatFunctionResultMessage,
    ChatToolResultMessage,
    ChatAssistantMessage,
]


class FunctionDefinition(TypedDict):
    type: Literal["functionDefinition"]
    name: str
    description: str
    parameters: JSONSchema7


class FunctionToolDefinition(TypedDict):
    type: Literal["function"]
    function: Dict[str, Union[str, JSONSchema7, bool]]


class ToolDefinition(TypedDict):
    type: Literal["toolDefinition"]
    tool: FunctionToolDefinition


Node: TypeAlias = Union[
    FunctionDefinition,
    ToolDefinition,
    BreakToken,
    First,
    Isolate,
    Capture,
    Config,
    Scope,
    Empty,
    ChatMessage,
    ChatImage,
    str,
    None,
    int,
    bool,
]

PromptElement: TypeAlias = Union[List[Node], Node]


class BaseProps(TypedDict, total=False):
    """Base properties for prompt elements."""

    p: Number  # absolute priority, max 1e6
    prel: Number  # relative priority
    name: str  # label for debugging purposes
    children: Union[PromptElement, Iterable[PromptElement]]
    on_eject: Callable[[], None]
    on_include: Callable[[], None]


# Chat message types
class ChatPromptSystemMessage(Scope):
    role: Literal["system"]
    name: NotRequired[str]
    to: NotRequired[str]
    content: Union[str, List[str]]


class ChatPromptUserMessage(Scope):
    role: Literal["user"]
    name: NotRequired[str]
    to: NotRequired[str]
    content: Union[str, List[str]]
    images: NotRequired[List[ImagePromptContent]]


class ChatPromptAssistantMessage(Scope):
    role: Literal["assistant"]
    to: NotRequired[str]
    content: NotRequired[Union[str, List[str]]]
    function_call: NotRequired[Dict[str, str]]
    tool_calls: NotRequired[List[Dict[str, Union[int, str, ChatAssistantFunctionToolCall]]]]


class ChatPromptFunctionResultMessage(Scope):
    role: Literal["function"]
    name: str
    to: NotRequired[str]
    content: Union[str, List[str]]


class ChatPromptToolResultMessage(Scope):
    role: Literal["tool"]
    name: NotRequired[str]
    to: NotRequired[str]
    content: Union[str, List[str]]


ChatPromptMessage = Union[
    ChatPromptSystemMessage,
    ChatPromptUserMessage,
    ChatPromptAssistantMessage,
    ChatPromptFunctionResultMessage,
    ChatPromptToolResultMessage,
]


class ChatPrompt(TypedDict):
    type: Literal["chat"]
    messages: List[ChatPromptMessage]


PromptString: TypeAlias = Union[str, List[str]]


class PromptContentWrapper(TypedDict):
    type: Literal["prompt_content"]
    content: PromptString
    images: NotRequired[List["ImagePromptContent"]]


class TextPromptContent(TypedDict):
    type: Literal["text"]
    text: str


class ImagePromptContent(TypedDict):
    type: Literal["image_url"]
    image_url: Dict[str, Union[str, Literal["low", "high", "auto"], Dict[str, int]]]


PromptContent = Union[TextPromptContent, ImagePromptContent]


class TextPrompt(TypedDict):
    type: Literal["text"]
    text: PromptString


class ChatAndFunctionPromptFunction(TypedDict):
    name: str
    description: str
    parameters: JSONSchema7


class FunctionPrompt(TypedDict):
    functions: List[ChatAndFunctionPromptFunction]


class ChatAndToolPromptToolFunction(TypedDict):
    type: Literal["function"]
    function: Dict[str, Union[str, JSONSchema7]]


class ToolPrompt(TypedDict):
    tools: List[ChatAndToolPromptToolFunction]


RenderedPrompt: TypeAlias = Union[
    PromptString,
    ChatPrompt,
    Dict[str, Union[ChatPrompt, FunctionPrompt]],
    Dict[str, Union[ChatPrompt, ToolPrompt]],
    Dict[str, Union[TextPrompt, FunctionPrompt]],
    Dict[str, Union[TextPrompt, ToolPrompt]],
    PromptContentWrapper,
]


class Prompt(Generic[PropsT, ReturnT]):
    config: Optional[PreviewConfig[PropsT, ReturnT]]

    def __call__(self, props: Dict) -> Union[PromptElement, Iterable[PromptElement]]: ...


class PreviewConfig(TypedDict, Generic[PropsT, ReturnT]):
    id: str
    prompt: Prompt[PropsT, ReturnT]
    dump: NotRequired[Callable[[PropsT], str]]
    hydrate: NotRequired[Callable[[str], PropsT]]


class RenderOptions(TypedDict):
    token_limit: int
    tokenizer: "PriomptTokenizer"
    count_tokens_fast_unsafe: NotRequired[bool]
    should_build_source_map: NotRequired[bool]
    last_message_is_incomplete: NotRequired[bool]


class SourceMap(TypedDict):
    name: str
    children: NotRequired[List[SourceMap]]
    string: NotRequired[str]
    start: int
    end: int


class AbsoluteSourceMap(TypedDict):
    name: str
    children: NotRequired[List[AbsoluteSourceMap]]
    string: NotRequired[str]
    start: int
    end: int
    __brand: Literal["absolute"]


class RenderOutput(TypedDict):
    prompt: RenderedPrompt
    token_count: int
    token_limit: int
    tokenizer: "PriomptTokenizer"
    tokens_reserved: int
    priority_cutoff: int
    output_handlers: List[OutputHandler["ChatCompletionResponseMessage"]]
    stream_handlers: List[OutputHandler[Iterable["ChatCompletionResponseMessage"]]]
    stream_response_object_handlers: List[OutputHandler[Iterable["StreamChatCompletionResponse"]]]
    config: ConfigProps
    duration_ms: NotRequired[int]
    source_map: NotRequired[SourceMap]
