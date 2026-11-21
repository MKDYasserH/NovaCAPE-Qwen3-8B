"""Ask a decision model served over HTTP (serving/server.py, or any clm-serve) to choose between options."""
import json
import secrets
import time
from pathlib import Path
from urllib.request import Request, urlopen

from schemas.serving import ChoiceQuestion, Decision, OptionProbability

QUESTION_ID = "choice"  # each request asks one question; its answer comes back under this id
FIRST_REQUEST_TIMEOUT_S = 1500  # a cold server loads Qwen3-8B before answering
# The server's URL is public, so it only answers requests carrying this key. Git-ignored; the game page reads it too.
SESSION_FILE = Path(__file__).parent / "game" / "session.json"


def session_key() -> str:
    if not SESSION_FILE.exists():
        SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
        SESSION_FILE.write_text(json.dumps({"key": secrets.token_urlsafe(24)}))
    return json.loads(SESSION_FILE.read_text())["key"]


class DecisionClient:
    def __init__(self, endpoint: str, api_key: str):
        self.url = endpoint.rstrip("/") + "/v1/systemone"
        self.api_key = api_key

    def choose(self, state: str, question: ChoiceQuestion) -> Decision:
        """The wire format is clm-serve's: a state, and a choice question whose options are texts."""
        criteria = {option.name: option.description for option in question.options}
        payload = {"state": state, "questions": {QUESTION_ID: {"type": "choice", "instructions": question.instructions, "criteria": criteria}}}
        request = Request(self.url, data=json.dumps(payload).encode(), method="POST",
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        started = time.perf_counter()
        with urlopen(request, timeout=FIRST_REQUEST_TIMEOUT_S) as response:
            answer = json.load(response)["answers"][QUESTION_ID]
        return Decision(
            probabilities=[OptionProbability(name=name, probability=p) for name, p in answer["probabilities"].items()],
            latency_ms=(time.perf_counter() - started) * 1000,
        )
