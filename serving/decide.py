r"""Zero-shot decisions and classification from the command line. Runs the model on this machine, or asks a server.

    python -m serving.decide                       # the example: which kind of memory is this?
    python -m serving.decide --state "I can't log in since the update." \
        --instructions "Which team should handle this ticket?" \
        --option billing="Payments, invoices, refunds." --option technical="Bugs, errors, login problems."
    python -m serving.decide --url URL             # the same, against a server (serving/server.py)
"""
import argparse

from schemas.serving import ChoiceQuestion, Decision, Option

EXAMPLE_STATE = "The summer I learned to ride a bike at my grandparents' house."
EXAMPLE_QUESTION = ChoiceQuestion(
    instructions="Which kind of memory is this?",
    options=[
        Option(name="childhood", description="A memory from childhood: being a kid, school, playing, growing up."),
        Option(name="travel", description="A memory of a trip: travelling, discovering a new place."),
        Option(name="work", description="A memory from working life: a job, colleagues, a career."),
        Option(name="love", description="A memory of a romantic relationship."),
    ],
)


def parse_option(text: str) -> Option:
    name, separator, description = text.partition("=")
    if not separator or not name or not description:
        raise argparse.ArgumentTypeError(f'expected name="description", got {text!r}')
    return Option(name=name, description=description)


def decide(url: str | None, state: str, question: ChoiceQuestion) -> Decision:
    if url:
        from serving.client import DecisionClient, session_key  # HTTP only: no torch needed on this machine

        return DecisionClient(url, session_key()).choose(state, question)
    import transformers  # loaded only for a local run

    from serving.decision_model import DecisionModel

    transformers.logging.set_verbosity_error()  # loading the base model reports the unused lm_head; harmless
    return DecisionModel.load().choose(state, question)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="a server to ask instead of running the model here")
    parser.add_argument("--state", default=EXAMPLE_STATE, help="the text to decide about")
    parser.add_argument("--instructions", default=EXAMPLE_QUESTION.instructions)
    parser.add_argument("--option", type=parse_option, action="append", help='name="description", at least twice')
    args = parser.parse_args()
    if args.option and len(args.option) < 2:
        parser.error("give at least two --option, one per choice")
    question = ChoiceQuestion(instructions=args.instructions, options=args.option or EXAMPLE_QUESTION.options)
    decision = decide(args.url, args.state, question)
    print(args.state)
    for option in sorted(decision.probabilities, key=lambda o: -o.probability):
        print(f"  {option.name:<10} {option.probability:.3f}")
    print(f"chosen: {decision.chosen}  ·  {decision.latency_ms:.0f} ms")


if __name__ == "__main__":
    main()
