from pathlib import Path

from syrupy.assertion import SnapshotAssertion
import pytest

from priompt import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    FunctionMessage,
    ToolResultMessage,
    Function,
    Image,
    Br,
    Hr,
    Empty,
    BreakToken,
    First,
    Scope,
    render,
)
from priompt.tokenizer import get_tokenizer_by_name_only_for_openai_tokenizers
from priompt.lib import prompt_to_openai_chat_messages, is_chat_prompt, prompt_has_functions


def test_system_message_basic():
    rendered = render(
        SystemMessage("hi this is a system message"),
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )
    assert is_chat_prompt(rendered["prompt"])


def test_system_message_with_names():
    # Create messages with names
    rendered = render(
        [
            SystemMessage("hi this is a system message", name="TestName"),
            UserMessage("hi this is a user message", name="carl"),
            UserMessage("hi this is a user message"),
        ],
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )
    assert is_chat_prompt(rendered["prompt"])

    # Test system message
    system_msg = rendered["prompt"]["messages"][0]

    assert system_msg["role"] == "system"
    assert system_msg["name"] == "TestName"

    # Test first user message
    user_msg1 = rendered["prompt"]["messages"][1]
    assert user_msg1["role"] == "user"
    assert user_msg1["name"] == "carl"

    # Test second user message
    user_msg2 = rendered["prompt"]["messages"][2]
    assert user_msg2["role"] == "user"
    assert "name" not in user_msg2


def test_function_invalid_name():
    # Test that invalid function names raise ValueError
    with pytest.raises(ValueError, match="Invalid function name"):
        Function(name="invalid name with spaces", description="test", parameters={"type": "object"})


def test_function():
    prompt = [
        Function(
            name="echo",
            description="Echo a message to the user.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo.",
                    }
                },
                "required": ["message"],
            },
        ),
        UserMessage("say hi"),
    ]

    # Render the prompt
    rendered = render(
        prompt,
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )

    # Test rendered output
    assert is_chat_prompt(rendered["prompt"])
    assert prompt_has_functions(rendered["prompt"])

    # Test function definition in rendered output
    assert rendered["prompt"]["functions"] == [
        {
            "name": "echo",
            "description": "Echo a message to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo.",
                    }
                },
                "required": ["message"],
            },
        }
    ]


def test_all_message_types():
    # Create test prompt with all message types
    prompt = [
        Function(
            name="echo",
            description="Echo a message to the user.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo.",
                    }
                },
                "required": ["message"],
            },
        ),
        SystemMessage("System message"),
        UserMessage("User message"),
        AssistantMessage(
            function_call={
                "name": "echo",
                "arguments": '{"message": "this is a test echo"}',
            }
        ),
        FunctionMessage("this is a test echo", name="echo"),
        AssistantMessage('print("Hello world!")', to="python"),
        ToolResultMessage("Hello world!", name="python", to="all"),
    ]

    # Render the prompt
    rendered = render(
        prompt,
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )

    # Test rendered output structure
    assert is_chat_prompt(rendered["prompt"])
    assert prompt_has_functions(rendered["prompt"])

    # Test function definition
    assert rendered["prompt"]["functions"] == [
        {
            "name": "echo",
            "description": "Echo a message to the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The message to echo.",
                    }
                },
                "required": ["message"],
            },
        }
    ]

    # Test messages
    expected_messages = [
        {
            "role": "system",
            "content": "System message",
        },
        {
            "role": "user",
            "content": "User message",
        },
        {
            "role": "assistant",
            "content": "",
            "function_call": {
                "name": "echo",
                "arguments": '{"message": "this is a test echo"}',
            },
        },
        {
            "role": "function",
            "name": "echo",
            "content": "this is a test echo",
        },
        {
            "role": "assistant",
            "to": "python",
            "content": 'print("Hello world!")',
        },
        {
            "role": "tool",
            "name": "python",
            "to": "all",
            "content": "Hello world!",
        },
    ]

    assert rendered["prompt"]["messages"] == expected_messages

    # Test OpenAI message conversion
    openai_messages = prompt_to_openai_chat_messages(rendered["prompt"])
    assert len(openai_messages) == 6
    assert openai_messages[0]["role"] == "system"
    assert openai_messages[1]["role"] == "user"
    assert openai_messages[2]["role"] == "assistant"
    assert openai_messages[3]["role"] == "function"
    assert openai_messages[4]["role"] == "assistant"
    assert openai_messages[5]["role"] == "tool"

    # Verify 'to' field is not present in OpenAI messages
    assert "to" not in openai_messages[4]
    assert "to" not in openai_messages[5]


def test_large_prompt():
    # Create a large prompt with varying priority levels
    prompt = [
        SystemMessage("You are a helpful assistant."),
        UserMessage(
            [
                Scope(
                    f"This is line {i + 1} of the large prompt.",
                    Br() if i % 10 == 0 else None,
                    Hr() if i % 20 == 0 else None,
                    p=i * 10,
                )
                for i in range(100)
            ]
        ),
        AssistantMessage("Understood. How can I help you with this large prompt?"),
        UserMessage(
            [
                Scope(
                    "This is a high priority message that should always be included.",
                    p=1000,
                ),
                Scope(
                    "This is a medium priority message that might be included.",
                    p=500,
                ),
                *[
                    Scope(
                        f"This is a message with priority {i * 10}.",
                        p=i * 10,
                    )
                    for i in range(50)
                ],
            ],
        ),
    ]

    # Test with default token limit
    rendered = render(
        prompt,
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )

    assert is_chat_prompt(rendered["prompt"])
    assert len(rendered["prompt"]["messages"]) == 4
    assert rendered["prompt"]["messages"][0]["role"] == "system"
    assert rendered["prompt"]["messages"][1]["role"] == "user"
    assert rendered["prompt"]["messages"][2]["role"] == "assistant"
    assert rendered["prompt"]["messages"][3]["role"] == "user"

    # Check high priority message inclusion
    assert (
        "This is a high priority message that should always be included."
        in rendered["prompt"]["messages"][3]["content"]
    )

    # Check dynamic priority messages
    dynamic_priority_count = rendered["prompt"]["messages"][3]["content"].count(
        "This is a message with priority"
    )
    assert dynamic_priority_count > 0
    assert dynamic_priority_count < 50

    # Check token count
    assert rendered["token_count"] <= 2000


def test_different_token_limits():
    # Create the same large prompt
    prompt = [
        SystemMessage("You are a helpful assistant."),
        UserMessage(
            [
                Scope(
                    f"This is line {i + 1} of the large prompt.",
                    Br() if i % 10 == 0 else None,
                    Hr() if i % 20 == 0 else None,
                    p=i * 10,
                )
                for i in range(100)
            ]
        ),
        AssistantMessage("Understood. How can I help you with this large prompt?"),
        UserMessage(
            [
                Scope(
                    "This is a high priority message that should always be included.",
                    p=1000,
                ),
                *[
                    Scope(
                        f"This is a message with priority {i * 10}.",
                        p=i * 10,
                    )
                    for i in range(50)
                ],
            ]
        ),
    ]

    # Test with different token limits
    token_limits = [500, 1000, 1500, 2000, 3000]

    for limit in token_limits:
        rendered = render(
            prompt,
            {
                "token_limit": limit,
                "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
            },
        )

        assert is_chat_prompt(rendered["prompt"])
        assert rendered["token_count"] <= limit

        # Check dynamic priority messages scaling with token limit
        user_message_content = rendered["prompt"]["messages"][3]["content"]
        dynamic_priority_count = user_message_content.count("This is a message with priority")

        if limit > 1000:
            assert dynamic_priority_count > 0


def test_nested_priority_levels():
    # Create nested priority prompt
    prompt = UserMessage(
        [
            Scope(
                "Top level high priority",
                Scope(
                    "Nested medium priority",
                    Scope(
                        "Deeply nested low priority",
                        Scope("Very deeply nested very low priority", p=50),
                        p=100,
                    ),
                    p=500,
                ),
                p=1000,
            ),
            *[
                Scope(
                    f"Priority level {(10 - i) * 100}",
                    Scope(
                        f"Nested priority level {(10 - i) * 50}",
                        p=(10 - i) * 50,
                    ),
                    p=(10 - i) * 100,
                )
                for i in range(10)
            ],
        ]
    )

    rendered = render(
        prompt,
        {
            "token_limit": 70,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
        },
    )

    assert is_chat_prompt(rendered["prompt"])
    content = rendered["prompt"]["messages"][0]["content"]

    # Check high priority content inclusion
    assert "Top level high priority" in content
    assert "Nested medium priority" in content
    assert "Priority level 1000" in content
    assert "Priority level 900" in content

    # Check lower priority content limitation
    lower_priority_count = sum(1 for i in range(1, 10) if f"Priority level {i}" in content)
    assert lower_priority_count < 9

    assert rendered["token_count"] <= 500


@pytest.fixture
def mock_image():
    with (Path(__file__).parent / "mocks" / "image.png").open("rb") as f:
        yield f.read()


def test_complex_prompt_with_nested_elements(snapshot: SnapshotAssertion, mock_image: bytes):
    # Create complex prompt
    prompt = [
        SystemMessage(
            [
                "This is a complex system message with ",
                Scope("high priority content", p=1000),
                " and ",
                Scope("medium priority content", p=500),
                ".",
                Br(),
                "New line in system message.",
            ]
        ),
        UserMessage(
            [
                "User message with a ",
                BreakToken(),
                " break and ",
                Br(),
                " line break.",
                Scope(
                    [
                        Image(
                            mock_image,
                            detail="auto",
                            dimensions={"width": 100, "height": 100},
                        ),
                        "Image with some text",
                    ],
                    p=800,
                ),
            ]
        ),
        AssistantMessage(
            [
                "Assistant response with ",
                Scope(
                    "lower priority information",
                    prel=-100,
                ),
                BreakToken(),
                "After break token in assistant message",
            ]
        ),
        Function(
            name="testFunction",
            description="A test function",
            parameters={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "number"},
                },
                "required": ["param1"],
            },
        ),
        FunctionMessage(
            [
                "Function result with nested scopes:",
                Scope(
                    [
                        {
                            "type": "scope",
                            "absolute_priority": 750,
                            "children": ["Highest priority"],
                        },
                        "Medium priority",
                        Br(),
                        "New line in function message",
                    ],
                    p=700,
                ),
            ],
            name="testFunction",
        ),
        *[
            UserMessage(
                [
                    f"Message {num} in array",
                    Scope(
                        "Even numbered message",
                        BreakToken(),
                        "After break in even message",
                        p=600,
                    )
                    if num % 2 == 0
                    else None,
                    Br(),
                ]
            )
            for num in range(1, 4)
        ],
        Empty(tokens=5),
    ]

    rendered = render(
        prompt,
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
            "should_build_source_map": True,
        },
    )

    # Remove duration for snapshot
    rendered.pop("duration_ms", None)
    assert rendered == snapshot


def test_nested_arrays_and_first_elements(snapshot: SnapshotAssertion):
    prompt = [
        SystemMessage(
            "System message for nested arrays test",
            Br(),
            "New line in system message",
        ),
        *[
            Scope(
                [
                    UserMessage(
                        f"Nested array message {num}",
                        BreakToken(),
                        "After break in nested message",
                        First(
                            Scope(
                                f"First choice for {num}",
                                p=800,
                            ),
                            Scope(
                                f"Second choice for {num}",
                                p=700,
                            ),
                            Scope(
                                f"Third choice for {num}",
                                Br(),
                                "New line in third choice",
                                p=600,
                            ),
                        ),
                    )
                    for num in subarray
                ],
                p=1000 - i * 100,
            )
            for i, subarray in enumerate([[1, 2], [3, 4]])
        ],
        AssistantMessage(
            First(
                Scope(
                    "High priority assistant response",
                    BreakToken(),
                    "After break in high priority",
                    p=900,
                ),
                Scope(
                    "Medium priority assistant response",
                    p=800,
                ),
                Scope(
                    "Low priority assistant response",
                    Br(),
                    "New line in low priority",
                    p=700,
                ),
            )
        ),
    ]

    rendered = render(
        prompt,
        {
            "token_limit": 1000,
            "tokenizer": get_tokenizer_by_name_only_for_openai_tokenizers("cl100k_base"),
            "should_build_source_map": True,
        },
    )

    rendered.pop("duration_ms", None)
    assert rendered == snapshot
