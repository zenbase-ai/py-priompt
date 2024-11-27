from typing import List

from priompt import (
    PromptElement,
    render,
    SourceMap,
    SystemMessage,
    UserMessage,
)
from priompt.tokenizer import get_tokenizer_by_name_only_for_openai_tokenizers
from priompt.lib import prompt_to_string_VULNERABLE_TO_PROMPT_INJECTION


def validate_source_map(absolute_source_map: SourceMap, text: str) -> bool:
    """Validate that source map leaf nodes match their text ranges."""

    def get_leaves(source_map: SourceMap) -> List[SourceMap]:
        children = source_map.get("children", [])
        if children and len(children) > 0:
            return [leaf for child in children for leaf in get_leaves(child)]
        return [source_map]

    leaves = get_leaves(absolute_source_map)
    leaves_with_string = [leaf for leaf in leaves if "string" in leaf]

    incorrect_leaves = [
        leaf for leaf in leaves_with_string if leaf["string"] != text[leaf["start"] : leaf["end"]]
    ]

    if incorrect_leaves:
        print(
            "Failed to validate source map, incorrect leaves are",
            sorted(incorrect_leaves, key=lambda x: x["start"]),
        )
        return False

    return True


def absolutify_source_map(source_map: SourceMap, offset: int = 0) -> SourceMap:
    """Convert relative source map positions to absolute positions."""
    new_offset = source_map["start"] + offset
    children = None
    if source_map.get("children"):
        children = [absolutify_source_map(child, new_offset) for child in source_map["children"]]

    return {
        **source_map,
        "start": new_offset,
        "end": source_map["end"] + offset,
        "children": children,
    }


def trivial_messages(message: str) -> PromptElement:
    """Simple test prompt with system and user messages."""
    return [
        SystemMessage(message),
        UserMessage(["Testing sourcemap!", "\n", "abcdef"]),
    ]


def emoji_and_japanese_messages() -> PromptElement:
    """Test prompt with emoji and Japanese text."""
    return [
        SystemMessage("🫨"),
        UserMessage(
            {"type": "scope", "name": "lemon", "children": ["🍋"]},
            {"type": "scope", "name": "japanese", "children": ["これはレモン"]},
        ),
    ]


def complex_messages(message: str) -> PromptElement:
    """Complex test prompt with nested scopes and multiple messages."""
    lines = message.split("\n")

    return [
        SystemMessage("The System Message"),
        UserMessage(
            [
                {
                    "type": "scope",
                    "name": "the first line",
                    "children": [
                        "This is the first line",
                        "\n",
                        {"type": "scope", "name": "buffer line", "children": [lines[0]]},
                        "\n",
                    ],
                },
                {
                    "type": "scope",
                    "name": "lines",
                    "relative_priority": -10,
                    "children": [
                        {"type": "scope", "relative_priority": -i, "children": [line]}
                        for i, line in enumerate(lines)
                    ],
                },
            ],
            p=1000,
        ),
        UserMessage(
            [
                {
                    "type": "scope",
                    "name": "code",
                    "children": [
                        "const rendered = await render(TestPromptSplit(props.message))",
                        'token_limit: 100, tokenizer: getTokenizerByName("cl100k_base"),',
                        "buildSourceMap: true,",
                    ],
                },
                {
                    "type": "scope",
                    "name": "checks",
                    "relative_priority": -10,
                    "children": [
                        'const promptString = Priompt.promptToString_VULNERABLE_TO_PROMPT_INJECTION(rendered.prompt, getTokenizerByName("cl100k_base")) const sourceMap = rendered.sourceMap'
                    ],
                },
            ]
        ),
    ]


def test_simple_sourcemap():
    """Test simple source map generation."""
    rendered = render(
        trivial_messages("System message for sourcemap test."),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
            "should_build_source_map": True,
        },
    )

    assert rendered["source_map"] is not None

    prompt_string = prompt_to_string_VULNERABLE_TO_PROMPT_INJECTION(
        rendered["prompt"], get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base")
    )

    assert rendered["source_map"]["start"] == 0
    assert rendered["source_map"]["end"] == len(prompt_string)

    assert validate_source_map(absolutify_source_map(rendered["source_map"], 0), prompt_string)


def test_emoji_and_japanese():
    """Test source map with emoji and Japanese text."""
    rendered = render(
        emoji_and_japanese_messages(),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
            "should_build_source_map": True,
        },
    )

    assert rendered["source_map"] is not None

    prompt_string = prompt_to_string_VULNERABLE_TO_PROMPT_INJECTION(
        rendered["prompt"], get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base")
    )

    assert rendered["source_map"]["start"] == 0
    assert rendered["source_map"]["end"] == len(prompt_string)

    assert validate_source_map(absolutify_source_map(rendered["source_map"], 0), prompt_string)


def test_complex_sourcemap():
    """Test complex source map generation."""
    message = """one lever that we haven't really touched is
    creating features that depend substantially
    on the user investing time
    in configuring them to work well
    eg if a user could spend a day configuring an agent
    that would consistently make them 50% more productive, it'd be worth it
    or a big company spending a month to
    structure their codebase in such a way that many more things than before can be ai automated.
    The polish we're missing on cmd+k is making it rly fast and snappy for the 1-2 line use case
    but it should be able to handle 100s of lines of code in a reasonable timeframe
    """

    rendered = render(
        complex_messages(message),
        {
            "token_limit": 300,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
            "should_build_source_map": True,
        },
    )

    prompt_string = prompt_to_string_VULNERABLE_TO_PROMPT_INJECTION(
        rendered["prompt"], get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base")
    )

    source_map = rendered["source_map"]
    assert source_map is not None
    assert source_map["start"] == 0
    assert source_map["end"] == len(prompt_string)

    assert validate_source_map(absolutify_source_map(source_map, 0), prompt_string)
