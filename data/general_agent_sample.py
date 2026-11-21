"""Collect varied, non-benchmark ADP demonstrations for the general post-training run."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from data.load_agent_data import SubsetBuilder, settings
from schemas.agent_data import Recording

DOMAINS = ("alfworld", "db", "kg", "webshop")
DEST = Path("downloads/post_inputs") / f"general-agents-{settings.revision}.json"


def main():
    rows, seen_states = [], set()
    for domain in DOMAINS:
        subset = f"agenttuning_{domain}"
        url = f"https://huggingface.co/datasets/{settings.repo}/resolve/{settings.revision}/{subset}/raw/full_std.jsonl"
        builder = SubsetBuilder(subset)
        counts = {"train": 0, "dev": 0}
        with urlopen(url, timeout=60) as stream:
            for i, line in enumerate(stream, 1):
                recording = Recording.model_validate_json(line)
                user = "\n".join(s.message for s in recording.steps if s.source == "user" and s.message)
                if not user:
                    continue
                group = hashlib.sha256(" ".join(user.casefold().split()).encode()).hexdigest()
                split = "dev" if int(group[:8], 16) % 5 == 0 else "train"
                builder.report.recordings_read = i
                for pair in builder._pairs_from(recording):
                    if counts[split] >= (640 if split == "train" else 160):
                        break
                    context = "\n".join(m.content for m in pair.state if m.role != "system")
                    if len(context) > 2000:
                        context = context[:480] + "\n…\n" + context[-1500:]
                    action = pair.action.strip()
                    if context in seen_states or not 1 <= len(action) <= 2400 or action.casefold() in {"ok.", "ok"}:
                        continue
                    seen_states.add(context)
                    counts[split] += 1
                    rows.append(dict(group=group, trajectory=pair.traj, source=domain, split=split,
                                     texts=[context, action]))
        print(domain, counts, flush=True)
        if min(counts.values()) < 16:
            raise ValueError(f"Too few distinct development or training cases in {domain}")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(dict(source_revision=settings.revision, rows=rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
