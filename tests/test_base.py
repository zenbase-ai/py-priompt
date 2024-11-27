from typing import Dict, Any

from priompt import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    is_plain_prompt,
    render,
    prompt_to_tokens,
)
from priompt.tokenizer import get_tokenizer_by_name_only_for_openai_tokenizers
from priompt.types import PromptElement


def Isolate(props: Dict[str, Any]) -> PromptElement:
    """Helper component for isolation tests"""
    if props.get("isolate"):
        return {
            "type": "isolate",
            "p": props.get("p"),
            "prel": props.get("prel"),
            "token_limit": props["token_limit"],
            "children": props.get("children", []),
        }
    else:
        return {
            "type": "scope",
            "p": props.get("p"),
            "prel": props.get("prel"),
            "children": props.get("children", []),
        }


def Test(props: Dict[str, Any]) -> PromptElement:
    """Test component that uses Isolate"""
    return [
        "This is the start of the prompt.",
        Isolate(
            {
                "token_limit": 100,
                "isolate": props["isolate"],
                "children": [
                    {
                        "type": "scope",
                        "prel": -i - 2000,
                        "children": [
                            f"This is an SHOULDBEINCLUDEDONLYIFISOLATED user message number {i}"
                        ],
                    }
                    for i in range(1000)
                ],
            }
        ),
        *[
            {"type": "scope", "prel": -i - 1000, "children": [f"This is user message number {i}"]}
            for i in range(1000)
        ],
        Isolate(
            {
                "token_limit": 100,
                "isolate": props["isolate"],
                "children": [
                    {
                        "type": "scope",
                        "prel": -i,
                        "children": [
                            f"{i},xl,x,,{('SHOULDBEINCLUDEDONLYIFNOTISOLATED' if i > 100 else '')}"
                        ],
                    }
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
        prompt_str = (
            rendered_isolated["prompt"]
            if isinstance(rendered_isolated["prompt"], str)
            else "".join(rendered_isolated["prompt"])
        )
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
        prompt_str = (
            rendered_unisolated["prompt"]
            if isinstance(rendered_unisolated["prompt"], str)
            else "".join(rendered_unisolated["prompt"])
        )
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

    assert donotbreak["token_count"] == len(tokens)
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

    assert dobreak["token_count"] == len(tokens2)
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

    assert donotbreak["token_count"] == len(tokens)
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

    assert dobreak["token_count"] == len(tokens2)
    assert tokens2 == [
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
        627,
        198,
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
        special_tokens["prompt"], get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base")
    )

    assert special_tokens["token_count"] == len(tokens)
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


def test_special_tokens_encoded():
    """Test handling of all special tokens encoded"""
    special_tokens = render(
        SpecialTokensPrompt(),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers(
                "cl100k_base_special_tokens"
            ),
            "last_message_is_incomplete": True,
        },
    )

    assert special_tokens["token_count"] >= 24
    tokens = prompt_to_tokens(
        special_tokens["prompt"],
        get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base_special_tokens"),
    )

    assert special_tokens["token_count"] == len(tokens)
    assert tokens == [
        100264,
        9125,
        100266,
        100264,
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
        {"type": "config", "max_response_tokens": "tokensReserved"},
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
    assert rendered["config"]["max_response_tokens"] == "tokensReserved"
