from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    TypedDict,
    Union,
)
import asyncio
import os

import yaml
import ujson

from .lib import render
from .tokenizer import get_tokenizer_by_name_only_for_openai_tokenizers, UsableTokenizer

if TYPE_CHECKING:
    from .types import PreviewConfig, Prompt, PromptElement, RenderOutput

T = TypeVar("T")
ReturnT = TypeVar("ReturnT")


class PreviewManagerGetPromptQuery(TypedDict):
    prompt_id: str
    props_id: str
    token_limit: int
    tokenizer: "UsableTokenizer"
    should_build_source_map: bool


class PreviewManagerGetRemotePromptQuery(TypedDict):
    prompt_id: str
    prompt_dump: str
    token_limit: int
    tokenizer: "UsableTokenizer"


class PreviewManagerGetRemotePropsQuery(TypedDict):
    prompt_id: str
    prompt_dump: str


class PreviewManagerGetPromptOutputQuery(TypedDict):
    prompt_id: str
    props_id: str
    token_limit: int
    tokenizer: "UsableTokenizer"
    completion: Union[Dict[str, Any], List[Dict[str, Any]]]
    stream: bool
    should_build_source_map: bool


class LiveModeOutput(TypedDict):
    live_mode_id: str


class LiveModeData(TypedDict):
    live_mode_id: str
    prompt_element: "PromptElement"


def get_project_root() -> str:
    if os.environ.get("PRIOMPT_PREVIEW_BASE_ABSOLUTE_PATH"):
        return os.environ["PRIOMPT_PREVIEW_BASE_ABSOLUTE_PATH"]
    if os.environ.get("PRIOMPT_PREVIEW_BASE_RELATIVE_PATH"):
        return str(Path(os.getcwd()) / os.environ["PRIOMPT_PREVIEW_BASE_RELATIVE_PATH"])
    return os.getcwd()


def config_from_prompt(prompt: "Prompt[T, ReturnT]") -> "PreviewConfig[T]":
    if hasattr(prompt, "config"):
        return prompt.config
    return PreviewConfig(id=prompt.name, prompt=prompt)


def dump_props(config: "PreviewConfig[T]", props: Dict[str, Any]) -> str:
    has_no_dump = any(key.startswith("DO_NOT_DUMP") for key in props.keys())

    if has_no_dump:
        object_to_dump = {
            key: value for key, value in props.items() if not key.startswith("DO_NOT_DUMP")
        }
    else:
        object_to_dump = props

    return (
        config.dump(object_to_dump)
        if hasattr(config, "dump")
        else default_yaml_dump(object_to_dump)
    )


def default_yaml_dump(props: Any) -> str:
    return ujson.dumps(props, indent=2)


def default_yaml_load(dump: str) -> Any:
    return yaml.safe_load(dump)


class PreviewManagerImpl:
    def __init__(self):
        self.should_dump = os.environ.get("ENV") == "development"
        self.previews: Dict[str, "PreviewConfig"] = {}
        self.last_live_mode_data: Optional[LiveModeData] = None
        self.last_live_mode_output_event = asyncio.Event()
        self.live_mode_result_future: Optional[asyncio.Future] = None

    def get_config(self, prompt_id: str) -> "PreviewConfig":
        return self.previews[prompt_id]

    def get_previews(self) -> Dict[str, Dict[str, List[str]]]:
        result = {}
        for prompt_id in self.previews:
            prompt_path = Path(get_project_root()) / "priompt" / prompt_id
            dumps_path = prompt_path / "dumps"

            prompt_path.mkdir(parents=True, exist_ok=True)
            dumps_path.mkdir(parents=True, exist_ok=True)

            props_ids = [f.stem for f in dumps_path.glob("*.yaml")]
            saved_ids = [f.stem for f in prompt_path.glob("*.yaml")]

            result[prompt_id] = {"dumps": props_ids, "saved": saved_ids}
        return result

    def get_prompt(self, query: PreviewManagerGetPromptQuery) -> "RenderOutput":
        element = self.get_element(query.prompt_id, query.props_id)

        return render(
            element,
            tokenizer=get_tokenizer_by_name_only_for_openai_tokenizers(query.tokenizer),
            token_limit=query.token_limit,
            should_build_source_map=query.should_build_source_map,
        )

    def register_config(self, config: "PreviewConfig[T]") -> None:
        if config.id in self.previews:
            if os.environ.get("ALLOW_PROMPT_REREGISTRATION") == "true":
                print(f"Warning: preview id {config.id} already registered")
            else:
                raise ValueError(f"Preview id {config.id} already registered")

        if (
            hasattr(config.prompt, "config")
            and config.prompt.config is not None
            and config.prompt.config.id != config.id
            and os.environ.get("NODE_ENV") == "development"
        ):
            raise ValueError(
                f"Prompt id {config.prompt.config.id} does not match config id {config.id}."
                " Prompts and configs need to be in a 1-to-1 mapping."
            )

        config.prompt.config = config
        self.previews[config.id] = config


# Create global instance
PreviewManager = PreviewManagerImpl()
