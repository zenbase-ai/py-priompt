from abc import ABC, abstractmethod
from typing import (
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeAlias,
    Union,
)

import tiktoken

if TYPE_CHECKING:
    from .types import PromptContent

# Constants for tokenizer names
CL100K_BASE = "cl100k_base"
R50K_BASE = "r50k_base"
P50K_BASE = "p50k_base"
GPT2_TOKENIZER = "gpt2"
LLAMA3_TOKENIZER = "llama3"
O200K_BASE = "o200k_base"
CODESTRAL_BASE = "codestral"

UsableTokenizer: TypeAlias = Literal[
    "cl100k_base",
    "cl100k_base_special_tokens",
    "r50k_base",
    "p50k_base",
    "gpt2",
    "llama3",
    "o200k_base",
    "o200k_base_special_tokens",
    "codestral",
]

OpenAIMessageRole: TypeAlias = Literal["system", "user", "assistant", "tool"]


# Constants for tokens
CL100K_SYSTEM_TOKENS = [100264, 9125, 100266]
CL100K_USER_TOKENS = [100264, 882, 100266]
CL100K_TOOL_TOKENS = [100264, 14506, 100266]
CL100K_ASSISTANT_TOKENS = [100264, 78191, 100266]
CL100K_END_TOKEN = 100265
CL100K_SYSTEM_TOKENS_STRING = "<|im_start|>system<|im_sep|>"
CL100K_USER_TOKENS_STRING = "<|im_start|>user<|im_sep|>"
CL100K_ASSISTANT_TOKENS_STRING = "<|im_start|>assistant<|im_sep|>"
CL100K_END_TOKEN_STRING = "<|im_end|>"

# Add O200K token constants
O200K_SYSTEM_TOKENS = [200006, 17360, 200008]
O200K_USER_TOKENS = [200006, 1428, 200008]
O200K_TOOL_TOKENS = [200006, 17952, 200008]
O200K_ASSISTANT_TOKENS = [200006, 173781, 200008]
O200K_END_TOKEN = 200007
O200K_SYSTEM_TOKENS_STRING = "<|im_start|>system<|im_sep|>"
O200K_USER_TOKENS_STRING = "<|im_start|>user<|im_sep|>"
O200K_TOOL_TOKENS_STRING = "<|im_start|>tool<|im_sep|>"
O200K_ASSISTANT_TOKENS_STRING = "<|im_start|>assistant<|im_sep|>"
O200K_END_TOKEN_STRING = "<|im_end|>"


def content_array_to_string_content(content: List[Union[str, "PromptContent"]]) -> List[str]:
    new_content: List[str] = []
    for c in content:
        if isinstance(c, str):
            new_content.append(c)
        elif isinstance(c, dict) and c.get("type") == "text":
            new_content.append(c["text"])
        elif isinstance(c, dict) and c.get("type") == "image_url":
            # Do nothing with images
            pass
    return new_content


def openai_chat_messages_to_prompt(messages: List[Dict], tokenizer: UsableTokenizer) -> str:
    if tokenizer not in ("o200k_base", "cl100k_base"):
        raise ValueError(
            f"Invalid tokenizer: {tokenizer}. Only o200k_base, cl100k_base tokenizers are supported."
        )

    parts = []
    for i, msg in enumerate(messages):
        header_string = openai_get_header_string_for_message(msg, tokenizer)
        if isinstance(msg["content"], list):
            new_content = "".join(content_array_to_string_content(msg["content"]))
        else:
            new_content = msg["content"]

        if i != 0:
            # OpenAI always adds the eos token before every non-starting message
            end_token_string = (
                O200K_END_TOKEN_STRING if tokenizer.startswith("o200k") else CL100K_END_TOKEN_STRING
            )
            parts.append(end_token_string + header_string + new_content)
        else:
            parts.append(header_string + new_content)

    return "".join(parts)


def openai_get_header_string_for_message(message: Dict, tokenizer: UsableTokenizer) -> str:
    if tokenizer not in ("o200k_base", "cl100k_base"):
        raise ValueError(
            f"Invalid tokenizer: {tokenizer}. Only o200k_base, cl100k_base tokenizers are supported."
        )

    header_string = ""
    role = message["role"]
    if role == "system":
        header_string = (
            O200K_SYSTEM_TOKENS_STRING
            if tokenizer.startswith("o200k")
            else CL100K_SYSTEM_TOKENS_STRING
        )
    elif role == "user":
        header_string = (
            O200K_USER_TOKENS_STRING if tokenizer.startswith("o200k") else CL100K_USER_TOKENS_STRING
        )
    elif role == "assistant":
        header_string = (
            O200K_ASSISTANT_TOKENS_STRING
            if tokenizer.startswith("o200k")
            else CL100K_ASSISTANT_TOKENS_STRING
        )
    elif role == "tool":
        header_string = (
            O200K_TOOL_TOKENS_STRING if tokenizer.startswith("o200k") else CL100K_USER_TOKENS_STRING
        )
    else:
        raise ValueError(f"Unknown role {role}")

    if "name" in message and message["name"] is not None:
        header_string = inject_name_string(header_string, message["name"])
    return header_string


def inject_name_string(tokens: str, name: str) -> str:
    return tokens.replace("<|im_sep|>", f":{name}<|im_sep|>")


def inject_name(tokens: List[int], name: str, tokenizer: UsableTokenizer) -> List[int]:
    # i don't really know if this is the right way to format it....
    name_tokens = encode_tokens(":" + name, {"tokenizer": tokenizer})
    return [*tokens[:-1], *name_tokens, tokens[-1]]


def inject_to(tokens: List[int], to: str, tokenizer: UsableTokenizer) -> List[int]:
    # Adjusting the function to handle 'to' parameter injection
    to_tokens = encode_tokens(" to=" + to, {"tokenizer": tokenizer})
    return [*tokens[:-1], *to_tokens, tokens[-1]]


# Add Codestral constants
CODESTRAL_BOS_TOKEN = "<s>"
CODESTRAL_EOS_TOKEN = "</s>"
CODESTRAL_EOS_TOKEN_ID = 2


def apply_codestral_chat_template(messages: List[Dict], options: Optional[Dict] = None) -> str:
    chat_template = CODESTRAL_BOS_TOKEN

    if messages[0]["role"] == "system":
        chat_template += f"[INST] <<SYS>>\n{messages[0]['content']}\n<</SYS>>\n\n"
        if messages[1]["role"] != "user":
            raise ValueError("Second message must be a user message if first is system")
        else:
            chat_template += f"{messages[1]['content']} [/INST]"

        if len(messages) > 3:
            raise ValueError("Too many messages")
        elif len(messages) == 3:
            if messages[2]["role"] == "assistant":
                chat_template += messages[2]["content"]
            else:
                raise ValueError("Third message with system prompt must be an assistant message")
    elif messages[0]["role"] == "user":
        chat_template += f"[INST] {messages[0]['content']} [/INST]"
        if len(messages) > 2:
            raise ValueError("Too many messages")
        elif len(messages) == 2:
            if messages[1]["role"] == "assistant":
                chat_template += messages[1]["content"]
            else:
                raise ValueError("Second message with user prompt must be an assistant message")
    else:
        raise ValueError("First message must be a system message or a user message")
    return chat_template


def encode_tokens(text: str, opts: Dict[str, UsableTokenizer]) -> List[int]:
    tokenizer_name = opts["tokenizer"]

    if tokenizer_name not in ("cl100k_base", "o200k_base"):
        raise ValueError(f"Unknown tokenizer {tokenizer_name}")

    return tiktoken.get_encoding(tokenizer_name).encode(text, disallowed_special=())


def decode_tokens(tokens: List[int], opts: Dict[str, UsableTokenizer]) -> str:
    tokenizer_name = opts["tokenizer"]

    if tokenizer_name not in ("cl100k_base", "o200k_base"):
        raise ValueError(f"Unknown tokenizer {tokenizer_name}")

    return tiktoken.get_encoding(tokenizer_name).decode(tokens)


def num_tokens(text: str, opts: Dict[str, UsableTokenizer]) -> int:
    tokenizer_name = opts["tokenizer"]

    if tokenizer_name not in ("cl100k_base", "o200k_base"):
        raise ValueError(f"Unknown tokenizer {tokenizer_name}")

    return len(tiktoken.get_encoding(tokenizer_name).encode(text, disallowed_special=()))


def estimate_tokens_using_bytecount(text: str, tokenizer: UsableTokenizer) -> Tuple[int, int]:
    """Returns a very conservative [lower, upper] bound on the number of tokens"""
    byte_length = len(text.encode("utf-8"))
    if tokenizer in ("cl100k_base", "o200k_base"):
        return (int(byte_length / 10), int(byte_length / 2.5))
    # conservative!
    return (int(byte_length / 10), int(byte_length / 2))


def estimate_tokens_using_charcount(text: str, tokenizer: UsableTokenizer) -> Tuple[int, int]:
    """Returns a very conservative [lower, upper] bound on the number of tokens"""
    length = len(text)
    if tokenizer in ("cl100k_base", "o200k_base"):
        return (int(length / 10), int(length / 1.5))
    # conservative!
    return (int(length / 10), length)


def num_tokens_for_image(dimensions: Dict[str, int], detail: Literal["low", "high", "auto"]) -> int:
    if detail == "low":
        return 85
    elif detail in ["high", "auto"]:
        # First, we rescale to fit within 2048 x 2048
        largest_ratio = max(dimensions["width"] / 2048, dimensions["height"] / 2048)
        if largest_ratio > 1:
            dimensions["width"] = int(dimensions["width"] / largest_ratio)
            dimensions["height"] = int(dimensions["height"] / largest_ratio)

        # Next, we scale the shortest side to be 768 px
        smallest_ratio = min(dimensions["width"] / 768, dimensions["height"] / 768)
        dimensions["width"] = int(dimensions["width"] / smallest_ratio)
        dimensions["height"] = int(dimensions["height"] / smallest_ratio)

        # Finally, we calculate the number of 512 x 512 blocks needed to cover the image
        # and pay 85 tokens per block
        num_width_blocks = (dimensions["width"] + 511) // 512
        num_height_blocks = (dimensions["height"] + 511) // 512
        return num_width_blocks * num_height_blocks * 85
    else:
        raise ValueError(f"Unknown detail level {detail}")


def openai_get_header_tokens_for_message(message: Dict, tokenizer: UsableTokenizer) -> List[int]:
    if tokenizer not in ("o200k_base", "cl100k_base"):
        raise ValueError(
            f"Invalid tokenizer: {tokenizer}. Only o200k_base, cl100k_base tokenizers are supported."
        )

    header_tokens: List[int]
    role = message["role"]

    role_tokens_map = {
        "system": (O200K_SYSTEM_TOKENS, CL100K_SYSTEM_TOKENS),
        "user": (O200K_USER_TOKENS, CL100K_USER_TOKENS),
        "assistant": (O200K_ASSISTANT_TOKENS, CL100K_ASSISTANT_TOKENS),
        "tool": (O200K_TOOL_TOKENS, CL100K_TOOL_TOKENS),
    }

    if role not in role_tokens_map:
        raise ValueError(f"Unknown role {role}")

    o200k_tokens, cl100k_tokens = role_tokens_map[role]
    header_tokens = o200k_tokens if tokenizer.startswith("o200k") else cl100k_tokens

    if "name" in message and message["name"] is not None:
        header_tokens = inject_name(header_tokens, message["name"], tokenizer)
    if "to" in message and message["to"] is not None:
        header_tokens = inject_to(header_tokens, message["to"], tokenizer)
    return header_tokens


def openai_chat_messages_to_tokens(messages: List[Dict], tokenizer: UsableTokenizer) -> List[int]:
    if tokenizer not in ("o200k_base", "cl100k_base"):
        raise ValueError(
            f"Invalid tokenizer: {tokenizer}. Only o200k_base, cl100k_base, and cl100k_base_special_tokens tokenizers are supported."
        )

    # OpenAI always adds the eos token before every non-starting message
    eos_token = O200K_END_TOKEN if tokenizer.startswith("o200k") else CL100K_END_TOKEN

    result_tokens: List[int] = []
    for i, msg in enumerate(messages):
        header_tokens = openai_get_header_tokens_for_message(msg, tokenizer)

        if isinstance(msg["content"], list):
            content_strings = content_array_to_string_content(msg["content"])
            content_tokens = []
            for content in content_strings:
                tokens = encode_tokens(content, {"tokenizer": tokenizer})
                content_tokens.extend(tokens)
        else:
            content_tokens = encode_tokens(msg["content"], {"tokenizer": tokenizer})

        if i != 0:
            result_tokens.append(eos_token)

        result_tokens.extend(header_tokens)
        result_tokens.extend(content_tokens)

    return result_tokens


def get_tokenizer_by_name_only_for_openai_tokenizers(name: UsableTokenizer) -> "PriomptTokenizer":
    if name == "cl100k_base":
        return CL100KTokenizer()
    elif name == "o200k_base":
        return O200KTokenizer()
    elif name == "codestral":
        return CodestralTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer {name}")


class PriomptTokenizer(ABC):
    name: str
    should_add_eos_token_to_each_message: bool

    @classmethod
    @abstractmethod
    def encode_tokens(cls, text: str) -> List[int]: ...

    @classmethod
    @abstractmethod
    def decode_tokens(cls, tokens: List[int]) -> str: ...

    @classmethod
    @abstractmethod
    def num_tokens(cls, text: str) -> int: ...

    @classmethod
    def estimate_tokens_fast(cls, text: str) -> int:
        return cls.estimate_tokens_using_char_count(text)[1]

    @classmethod
    @abstractmethod
    def estimate_tokens_using_char_count(cls, text: str) -> Tuple[int, int]: ...

    @classmethod
    @abstractmethod
    def get_header_string_for_message(cls, message: Dict) -> str: ...

    @classmethod
    @abstractmethod
    def get_header_tokens_for_message(cls, message: Dict) -> List[int]: ...

    @classmethod
    @abstractmethod
    def get_eos_token_id(cls) -> int: ...

    @classmethod
    @abstractmethod
    def get_eos_token(cls) -> str: ...

    @classmethod
    @abstractmethod
    def apply_chat_template(
        cls,
        messages: List[Dict],
        options: Optional[Dict] = None,
    ) -> str: ...

    @classmethod
    @abstractmethod
    def apply_chat_template_tokens(
        cls,
        messages: List[Dict],
        options: Optional[Dict] = None,
    ) -> List[int]: ...


class CL100KTokenizer(PriomptTokenizer):
    name = "cl100k_base"
    should_add_eos_token_to_each_message = True

    @classmethod
    def encode_tokens(cls, text: str) -> List[int]:
        return encode_tokens(text, {"tokenizer": cls.name})

    @classmethod
    def decode_tokens(cls, tokens: List[int]) -> str:
        return decode_tokens(tokens, {"tokenizer": cls.name})

    @classmethod
    def num_tokens(cls, text: str) -> int:
        return num_tokens(text, {"tokenizer": cls.name})

    @classmethod
    def estimate_tokens_using_char_count(cls, text: str) -> Tuple[int, int]:
        return estimate_tokens_using_charcount(text, cls.name)

    @classmethod
    def get_eos_token(cls) -> str:
        return CL100K_END_TOKEN_STRING

    @classmethod
    def get_eos_token_id(cls) -> int:
        return CL100K_END_TOKEN

    @classmethod
    def get_header_string_for_message(cls, message: Dict) -> str:
        return openai_get_header_string_for_message(message, cls.name)

    @classmethod
    def get_header_tokens_for_message(cls, message: Dict) -> List[int]:
        return openai_get_header_tokens_for_message(message, cls.name)

    @classmethod
    def apply_chat_template(cls, messages: List[Dict], options: Optional[Dict] = None) -> str:
        return openai_chat_messages_to_prompt(messages, cls.name)

    @classmethod
    def apply_chat_template_tokens(
        cls, messages: List[Dict], options: Optional[Dict] = None
    ) -> List[int]:
        return openai_chat_messages_to_tokens(messages, cls.name)


class O200KTokenizer(PriomptTokenizer):
    name = "o200k_base"
    should_add_eos_token_to_each_message = True

    @classmethod
    def encode_tokens(cls, text: str) -> List[int]:
        return encode_tokens(text, {"tokenizer": cls.name})

    @classmethod
    def decode_tokens(cls, tokens: List[int]) -> str:
        return decode_tokens(tokens, {"tokenizer": cls.name})

    @classmethod
    def num_tokens(cls, text: str) -> int:
        return num_tokens(text, {"tokenizer": cls.name})

    @classmethod
    def estimate_tokens_using_char_count(cls, text: str) -> Tuple[int, int]:
        return estimate_tokens_using_charcount(text, cls.name)

    @classmethod
    def get_eos_token(cls) -> str:
        return O200K_END_TOKEN_STRING

    @classmethod
    def get_eos_token_id(cls) -> int:
        return O200K_END_TOKEN

    @classmethod
    def get_header_string_for_message(cls, message: Dict) -> str:
        return openai_get_header_string_for_message(message, cls.name)

    @classmethod
    def get_header_tokens_for_message(cls, message: Dict) -> List[int]:
        return openai_get_header_tokens_for_message(message, cls.name)

    @classmethod
    def apply_chat_template(cls, messages: List[Dict], options: Optional[Dict] = None) -> str:
        return openai_chat_messages_to_prompt(messages, cls.name)

    @classmethod
    def apply_chat_template_tokens(
        cls, messages: List[Dict], options: Optional[Dict] = None
    ) -> List[int]:
        return openai_chat_messages_to_tokens(messages, cls.name)


class CodestralTokenizer(PriomptTokenizer):
    name = "codestral"
    should_add_eos_token_to_each_message = True

    @classmethod
    def encode_tokens(cls, text: str) -> List[int]:
        return encode_tokens(text, {"tokenizer": cls.name})

    @classmethod
    def decode_tokens(cls, tokens: List[int]) -> str:
        return decode_tokens(tokens, {"tokenizer": cls.name})

    @classmethod
    def num_tokens(cls, text: str) -> int:
        return num_tokens(text, {"tokenizer": cls.name})

    @classmethod
    def estimate_tokens_using_char_count(text: str) -> Tuple[int, int]:
        return estimate_tokens_using_charcount(text, "codestral")

    @classmethod
    def get_eos_token(cls) -> str:
        return CODESTRAL_EOS_TOKEN

    @classmethod
    def get_eos_token_id(cls) -> int:
        return CODESTRAL_EOS_TOKEN_ID

    @classmethod
    def get_header_string_for_message(cls, message: Dict) -> str:
        return openai_get_header_string_for_message(message, cls.name)

    @classmethod
    def get_header_tokens_for_message(cls, message: Dict) -> List[int]:
        return openai_get_header_tokens_for_message(message, cls.name)

    @classmethod
    def apply_chat_template(cls, messages: List[Dict], options: Optional[Dict] = None) -> str:
        return apply_codestral_chat_template(messages)

    @classmethod
    def apply_chat_template_tokens(
        cls, messages: List[Dict], options: Optional[Dict] = None
    ) -> List[int]:
        return openai_chat_messages_to_tokens(messages, cls.name)
