# PyPriompt

PyPriompt (_Python + priority + prompt_) is a Python port of the [Priompt](https://github.com/anysphere/priompt) library. It uses priorities to decide what to include in the context window.

Priompt is an attempt at a _prompt design_ library, inspired by web design libraries like React. Read more about the motivation [here](https://arvid.xyz/prompt-design).

_Note: Although the original test suite passes, this was ported in a day with the help of Cursor AI, so there may be bugs. Please open an issue if you find any!_

## Installation

```bash
uv add priompt
rye add priompt
poetry add priompt
pip install priompt
```

## Principles

Prompts are rendered from a Python component, which can look something like this:

```python
from priompt import (
  component,
  SystemMessage,
  Scope,
  UserMessage,
  Empty,
  PromptElement,
)

@component
def example_prompt(
    name: str,
    message: str,
    history: list[dict[str, str]],
) -> PromptElement:
    capitalized_name = name[0].upper() + name[1:]

    return [
        SystemMessage(f"The user's name is {capitalized_name}. Please respond to them kindly."),
        *[
            Scope(
                UserMessage(m["message"]) if m["case"] == "user" else AssistantMessage(m["message"]),
                prel=-(len(history) - i),
            )
            for i, m in enumerate(history)
        ],
        UserMessage(message),
        Empty(1000),
    ]
```

A component is rendered only once. Each child has a priority, where a higher priority means that the child is more important to include in the prompt. If no priority is specified, the child is included if and only if its parent is included. Absolute priorities are specified with `p` and relative ones are specified with `prel`.

In the example above, we always include the system message and the latest user message, and are including as many messages from the history as possible, where later messages are prioritized over earlier messages.

The key promise of the priompt renderer is:

> Let $T$ be the token limit and $\text{Prompt}(p_\text{cutoff})$ be the function that creates a prompt by including all scopes with priority $p_\text{scope} \geq p_\text{cutoff}$, and no other. Then, the rendered prompt is $\text{\textbf{P}} = \text{Prompt}(p_\text{opt-cutoff})$ where $p_\text{opt-cutoff}$ is the minimum value such that $|\text{Prompt}(p_\text{opt-cutoff})| \leq T$.

The building blocks of a priompt prompt are:

1. `Scope`: this allows you to set priorities `p` for absolute or `prel` for relative.
2. `First`: the first child with a sufficiently high priority will be included, and all children below it will not. This is useful for fallbacks for implementing something like "when the result is too long we want to say `(result omitted)`".
3. `Empty`: for specifying empty space, useful for reserving tokens for generation.
4. `Capture`: capture the output and parse it right within the prompt.
5. `Isolate`: isolate a section of the prompt with its own token limit. This is useful for guaranteeing that the start of the prompt will be the same for caching purposes. it would be nice to extend this to allow token limits like `100% - 100`.
6. `Br`: force a token break at a particular location, which is useful for ensuring exact tokenization matches between two parts of a prompt (e.g. when implementing something like speculative edits).
7. `Config`: specify a few common configuration properties, such as `stop` token and `maxResponseTokens`, which can make the priompt dump more self-contained and help with evals.

You can create components all you want, just like in React. The builtin components are:

1. `UserMessage`, `AssistantMessage` and `SystemMessage`: for building message-based prompts.
2. `Image`: for adding images into the prompt.
3. `Tools`: for specifying tools that the AI can call using a JSON schema.

## Advanced features

1. `on_eject` and `on_include`: callbacks that can be passed into any scope, which are called when the scope is either excluded or included in the final prompt. This allows you to change your logic depending on if something is too large for the prompt.
2. Sourcemaps: when setting `should_build_source_map` to `true`, the renderer computes a map between the actual characters in the prompt and the part of the JSX tree that they came from. This can be useful to figure out where cache misses are coming from in the prompt.
3. Prepend `DO_NOT_DUMP` to your priompt props key to prevent it from being dumped, which is useful for really big objects.

## Caveats

1. We've discovered that adding priorities to everything is sort of an anti-pattern. It is possible that priorities are the wrong abstraction. We have found them useful though for including long files in the prompt in a line-by-line way.
2. The Priompt renderer has no builtin support for creating cacheable prompts. If you overuse priorities, it is easy to make hard-to-cache prompts, which may increase your cost or latency for LLM inference. We are interested in good solutions here, but for now it is up to the prompt designer to think about caching.
   1. *Update: Priompt sourcemaps help with caching debugging!*
3. The current version of priompt only supports around 10K scopes reasonably fast (this is enough for most use cases). If you want to include a file in the prompt that is really long (>10K lines), and you split it line-by-line, you probably want to implement something like "for lines farther than 1000 lines away from the cursor position we have coarser scopes of 10 lines at a time".
4. For latency-critical prompts you want to monitor the time usage in the priompt preview dashboard. If there are too many scopes you may want to optimize for performance.
5. The Priompt renderer is not always guaranteed to produce the perfect $p_\text{opt-cutoff}$. For example, if a higher-priority child of a `First` has more tokens than a lower-priority child, the currently implemented binary search renderer may return a (very slightly) incorrect result.

## TODOs

- [ ] Verify `Capture`, `Tool`, and `Tools` work.
- [ ] Verify compatibility with `priompt-preview`.
- [ ] Add async support for rendering and tool calls (requires using `anysphere/tiktoken-rs`)

## Contributions

Contributions are very welcome! This entire repo is MIT-licensed.
