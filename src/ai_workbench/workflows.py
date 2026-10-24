from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .domain import Proposal
from .models import ModelAdapter
from .proposals import parse_proposal

ASK_INSTRUCTION = """Mode: ASK (read-only explanation).
Answer the user's question using the supplied project context. Do not propose file replacements
and do not emit <<<FILE ...>>> blocks. No project files may change.

User request:
{task}"""

CHANGE_INSTRUCTION = """Mode: CHANGE (reviewed whole-file proposal).
Address the task using the supplied project context. If changes are needed, respond
only with one or more complete replacement files in exactly this format:

<<<FILE relative/path.py>>>
complete file contents
<<<END FILE>>>

Do not include commentary outside the file blocks. The application will validate
and display the proposal for explicit user review; you cannot write files directly.
Paths are project-relative. The automatically supplied project-state files
.ai/context.md, .ai/memory.md, and .ai/next-steps.md are valid proposal targets
when the task genuinely requires changing them. Do not omit the .ai/ prefix.

Task:
{task}"""

AUTOPILOT_STEP_TASK = """Execute exactly this one current project step, then stop:

{step}

The automatically supplied .ai/goal.md defines the destination and must not be
modified. Do not modify protected acceptance tests or any other protected path.
You may update implementation files and .ai/memory.md with durable discoveries.
After completing this step, update .ai/next-steps.md with the next actionable step.
Return every change as a complete file using the required proposal protocol."""


def autopilot_step_task(step: str) -> str:
    return AUTOPILOT_STEP_TASK.format(step=step)


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    response: str
    proposal: Proposal | None = None


class PromptWorkflow:
    def ask(self, adapter: ModelAdapter, context: str, task: str) -> WorkflowResult:
        response = adapter.complete(context, ASK_INSTRUCTION.format(task=task))
        return WorkflowResult(response=response)

    def change(
        self,
        adapter: ModelAdapter,
        root: Path,
        context: str,
        task: str,
        on_response: Callable[[], None] | None = None,
    ) -> WorkflowResult:
        response = adapter.complete(context, CHANGE_INSTRUCTION.format(task=task))
        if on_response is not None:
            on_response()
        return WorkflowResult(response=response, proposal=parse_proposal(root, response))
