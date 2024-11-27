from __future__ import annotations
from typing import Generic, TypeVar, Optional, Dict, List, TypedDict

T = TypeVar("T")


class Output(TypedDict, Generic[T]):
    output: T
    priority: int | None


class NoPriorityOutput(TypedDict, Generic[T]):
    output: T
    priority: None


class OutputCatcher(Generic[T]):
    def __init__(self):
        self.outputs: List[Output[T]] = []
        self.no_priority_outputs: List[NoPriorityOutput[T]] = []

    def on_output(self, output: T, options: Optional[Dict] = None) -> None:
        """
        Add an output with optional priority.
        Args:
            output: The output value
            options: Optional dict with 'p' key for priority
        """
        if options and "p" in options:
            self.outputs.append(Output(output=output, priority=options["p"]))
            self.outputs.sort(key=lambda x: x["priority"], reverse=True)
        else:
            self.no_priority_outputs.append(NoPriorityOutput(output=output))

    def get_outputs(self) -> List[T]:
        """
        Get a sorted list of outputs, with highest priority first,
        followed by unprioritized outputs in order added.
        """
        all_outputs = self.outputs + self.no_priority_outputs
        return [o["output"] for o in all_outputs]

    def get_output(self) -> Optional[T]:
        """Get the first/highest priority output if any exist."""
        if self.outputs:
            return self.outputs[0]["output"]
        elif self.no_priority_outputs:
            return self.no_priority_outputs[0]["output"]
        return None
