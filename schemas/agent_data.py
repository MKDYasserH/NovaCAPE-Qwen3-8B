"""Agent Data Protocol recordings (input) and the (state, action) pairs built from them (output)."""
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

Split = Literal["train", "val"]
VALIDATION_SPLIT: Split = "val"


# --- One ADP recording, as stored in <subset>/raw/full_std*.jsonl. Unknown fields are ignored. ---

class ToolCall(BaseModel):
    function_name: str | None = None
    arguments: JsonValue = None

    def render(self) -> str:
        """go({"location": "bed 1"})"""
        return f"{self.function_name}({json.dumps(self.arguments, ensure_ascii=False)})"


class WebExtra(BaseModel):
    axtree: JsonValue = None  # accessibility tree: a text version of the web page


class ResultExtra(BaseModel):
    web: WebExtra | None = None


class ObservationResult(BaseModel):
    content: JsonValue = None
    extra: ResultExtra | None = None

    def text(self) -> str:
        """The tool's output; web agents leave it empty and put the page in extra.web.axtree."""
        value = self.content or (self.extra and self.extra.web and self.extra.web.axtree)
        if not value:
            return ""
        return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


class Observation(BaseModel):
    results: list[ObservationResult] | None = None


class RecordingStep(BaseModel):
    source: str  # "system", "user" or "agent"
    message: str | None = None
    tool_calls: list[ToolCall] | None = None
    observation: Observation | None = None

    @field_validator("message", mode="before")
    @classmethod
    def join_text_parts(cls, message: JsonValue) -> JsonValue:
        """A few subsets (swe-smith, codescout, ...) store messages as [{"type": "text", "text": ...}] parts."""
        if not isinstance(message, list):
            return message
        return "\n".join(part["text"] for part in message if isinstance(part, dict) and part.get("type") == "text")

    def action_text(self) -> str:
        """What an agent step did: its message, then one line per tool call."""
        message = [self.message] if self.message and self.message.strip() else []
        calls = [call.render() for call in self.tool_calls or []]
        return "\n".join(message + calls).strip()

    def observation_text(self) -> str:
        results = self.observation.results if self.observation else None
        return "\n".join(text for result in results or [] if (text := result.text()))


class Recording(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)

    trajectory_id: str | None = None
    session_id: str | None = None
    steps: list[RecordingStep] = []


# --- What we build from them. ---

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class AgentPair(BaseModel):
    """One parquet row: the conversation so far (state) and what the agent did next (action)."""
    subset: str
    traj: str
    step: int  # index in Recording.steps
    split: Split
    state: list[ChatMessage] = Field(min_length=1)
    action: str = Field(min_length=1)

    def to_row(self) -> dict:
        return {**self.model_dump(exclude={"state"}), "state": json.dumps([m.model_dump() for m in self.state], ensure_ascii=False)}


class SubsetReport(BaseModel):
    subset: str
    pairs: int
    recordings_read: int = 0
    recordings_skipped: int = 0  # lines that are not valid JSON or don't match Recording
    already_built: bool = False


class EmbeddingReport(BaseModel):
    subset: str
    pairs: int
    tokens: int = 0
    minutes: float = 0.0
    already_embedded: bool = False
