from beartype.typing import Dict, Any

from priompt import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    Scope,
    Isolate,
    render,
    component,
)
from priompt.tokenizer import get_tokenizer_by_name_only_for_openai_tokenizers
from priompt.types import PromptElement
from priompt.lib import is_plain_prompt, prompt_string_to_string, prompt_to_tokens


@component
def TestIsolate(props: Dict[str, Any]) -> PromptElement:
    """Helper component for isolation tests"""
    if props.get("isolate"):
        return Isolate(
            *props.get("children", []),
            token_limit=props["token_limit"],
            p=props.get("p"),
            prel=props.get("prel"),
        )
    else:
        return Scope(
            *props.get("children", []),
            p=props.get("p"),
            prel=props.get("prel"),
        )


@component
def Test(props: Dict[str, Any]) -> PromptElement:
    """Test component that uses Isolate"""
    return [
        "This is the start of the prompt.",
        TestIsolate(
            {
                "token_limit": 100,
                "isolate": props["isolate"],
                "children": [
                    Scope(
                        f"This is an SHOULDBEINCLUDEDONLYIFISOLATED user message number {i}",
                        prel=-i - 2000,
                    )
                    for i in range(1000)
                ],
            }
        ),
        *[Scope(f"This is user message number {i}", prel=-i - 1000) for i in range(1000)],
        TestIsolate(
            {
                "token_limit": 100,
                "isolate": props["isolate"],
                "children": [
                    Scope(
                        f"{i},xl,x,,{('SHOULDBEINCLUDEDONLYIFNOTISOLATED' if i > 100 else '')}",
                        prel=-i,
                    )
                    for i in range(1000)
                ],
            }
        ),
    ]


def test_isolate():
    """Test isolation functionality"""
    rendered_isolated = render(
        Test({"isolate": True}),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )

    assert rendered_isolated["token_count"] <= 1000
    assert is_plain_prompt(rendered_isolated["prompt"])
    if is_plain_prompt(rendered_isolated["prompt"]):
        prompt_str = prompt_string_to_string(rendered_isolated["prompt"])
        assert "SHOULDBEINCLUDEDONLYIFISOLATED" in prompt_str
        assert "SHOULDBEINCLUDEDONLYIFNOTISOLATED" not in prompt_str

    rendered_unisolated = render(
        Test({"isolate": False}),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )

    assert rendered_unisolated["token_count"] <= 1000
    assert is_plain_prompt(rendered_unisolated["prompt"])
    if is_plain_prompt(rendered_unisolated["prompt"]):
        prompt_str = prompt_string_to_string(rendered_unisolated["prompt"])
        assert "SHOULDBEINCLUDEDONLYIFISOLATED" not in prompt_str
        assert "SHOULDBEINCLUDEDONLYIFNOTISOLATED" in prompt_str


def SimplePrompt(props: Dict[str, Any]) -> PromptElement:
    """Simple prompt component for testing"""
    return [
        "This is the start of the p",
        {"type": "breaktoken"} if props.get("breaktoken") else None,
        "rompt. This is the second part of the prompt.",
    ]


def test_prompt_to_tokens():
    """Test token conversion functionality"""
    donotbreak = render(
        SimplePrompt({"breaktoken": False}),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )

    tokens = prompt_to_tokens(
        donotbreak["prompt"], get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base")
    )

    assert tokens == [
        2028,
        374,
        279,
        1212,
        315,
        279,
        10137,
        13,
        1115,
        374,
        279,
        2132,
        961,
        315,
        279,
        10137,
        13,
    ]

    dobreak = render(
        SimplePrompt({"breaktoken": True}),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )

    assert dobreak["token_count"] == donotbreak["token_count"] + 1
    tokens2 = prompt_to_tokens(
        dobreak["prompt"], get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base")
    )

    assert tokens2 == [
        2028,
        374,
        279,
        1212,
        315,
        279,
        281,
        15091,
        13,
        1115,
        374,
        279,
        2132,
        961,
        315,
        279,
        10137,
        13,
    ]


def SimpleMessagePrompt(props: Dict[str, Any]) -> PromptElement:
    """Simple message prompt component for testing"""
    return [
        SystemMessage(
            [
                "This is the start of the prompt.",
                "\n",
                {"type": "breaktoken"} if props.get("breaktoken") else None,
                "\n",
                "This is the second part of the prompt.",
            ]
        ),
        UserMessage("hi!"),
    ]


def test_message_prompt_to_tokens():
    """Test token conversion for message prompts"""
    donotbreak = render(
        SimpleMessagePrompt({"breaktoken": False}),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
            "last_message_is_incomplete": True,
        },
    )

    tokens = prompt_to_tokens(
        donotbreak["prompt"], get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base")
    )

    assert tokens == [
        100264,
        9125,
        100266,
        2028,
        374,
        279,
        1212,
        315,
        279,
        10137,
        382,
        2028,
        374,
        279,
        2132,
        961,
        315,
        279,
        10137,
        13,
        100265,
        100264,
        882,
        100266,
        6151,
        0,
    ]

    dobreak = render(
        SimpleMessagePrompt({"breaktoken": True}),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
            "last_message_is_incomplete": True,
        },
    )

    assert dobreak["token_count"] == donotbreak["token_count"] + 1
    tokens2 = prompt_to_tokens(
        dobreak["prompt"], get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base")
    )

    potential_tokens2 = (
        [
            100264,
            9125,
            100266,
            2028,
            374,
            279,
            1212,
            315,
            279,
            10137,
            627,  # ".\n"
            198,  # "\n"
            2028,
            374,
            279,
            2132,
            961,
            315,
            279,
            10137,
            13,
            100265,
            100264,
            882,
            100266,
            6151,
            0,
        ],
        [
            100264,
            9125,
            100266,
            2028,
            374,
            279,
            1212,
            315,
            279,
            10137,
            382,  # ".\n\n"
            2028,
            374,
            279,
            2132,
            961,
            315,
            279,
            10137,
            13,
            100265,
            100264,
            882,
            100266,
            6151,
            0,
        ],
    )
    assert tokens2 in potential_tokens2


def SpecialTokensPrompt() -> PromptElement:
    """Prompt component with special tokens for testing"""
    return [
        SystemMessage("<|im_start|>"),
        UserMessage("<|diff_marker|>"),
        AssistantMessage("<|endoftext|>"),
    ]


def test_special_tokens():
    """Test handling of special tokens"""
    special_tokens = render(
        SpecialTokensPrompt(),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
            "last_message_is_incomplete": True,
        },
    )

    assert special_tokens["token_count"] >= 24
    tokens = prompt_to_tokens(
        special_tokens["prompt"],
        get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
    )

    assert tokens == [
        100264,
        9125,
        100266,
        27,
        91,
        318,
        5011,
        91,
        29,
        100265,
        100264,
        882,
        100266,
        27,
        91,
        13798,
        27363,
        91,
        29,
        100265,
        100264,
        78191,
        100266,
        27,
        91,
        8862,
        728,
        428,
        91,
        29,
    ]


def TestConfig(props: Dict[str, Any]) -> PromptElement:
    """Test component for config functionality"""
    return [
        "This is the start of the prompt.",
        {"type": "config", "stop": "\n"},
        {"type": "config", "max_response_tokens": "tokens_reserved"},
    ]


def test_config():
    """Test configuration functionality"""
    rendered = render(
        TestConfig({"num_configs": 1}),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )

    assert rendered["token_count"] <= 1000
    assert is_plain_prompt(rendered["prompt"])
    assert rendered["config"]["stop"] == "\n"
    assert rendered["config"]["max_response_tokens"] == "tokens_reserved"
