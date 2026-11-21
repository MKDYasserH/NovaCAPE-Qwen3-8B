"""Prepare a small, instruction-disjoint sample of OpenBMB's generic preference data."""
import hashlib
import json
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

REVISION = "40b436560ca83a8dba36114c22ab3c66e43f6d5e"
SOURCES = ("flan", "evol_instruct", "ultrachat", "sharegpt", "false_qa", "truthful_qa")
QUOTAS = {"train": 320, "dev": 64}
DEST = Path("downloads/post_inputs") / f"general-preferences-{REVISION}.json"


def preference(row):
    question = row["instruction"].strip()
    if not 10 <= len(question) <= 2800:
        return None
    answers = []
    for answer in row["completions"]:
        response, critique = answer["response"].strip(), (answer.get("critique") or "").strip()
        if 3 <= len(response) <= 3500 and 20 <= len(critique) <= 1600:
            answers.append((float(answer["fine-grained_score"]), response, critique))
    if len(answers) < 2:
        return None
    low, *_, high = sorted(answers)
    if high[0] - low[0] < 1 or high[1] == low[1] or high[2] == low[2]:
        return None
    return [question, high[1], low[1], high[2], low[2]]


def main():
    rows, seen = [], set()
    for source in SOURCES:
        counts = dict.fromkeys(QUOTAS, 0)
        url = f"https://huggingface.co/datasets/openbmb/UltraFeedback/resolve/{REVISION}/{source}.jsonl"
        with urlopen(Request(url, headers={"User-Agent": "S1-Nova-research"}), timeout=90) as stream:
            for line in stream:
                sample = preference(json.loads(line))
                if sample is None:
                    continue
                key = " ".join(unicodedata.normalize("NFKC", sample[0]).casefold().split())
                ident = hashlib.sha256(key.encode()).hexdigest()
                split = "dev" if int(ident[:8], 16) % 5 == 0 else "train"
                if ident in seen or counts[split] >= QUOTAS[split]:
                    continue
                seen.add(ident)
                counts[split] += 1
                rows.append(dict(id=ident, source=source, split=split, texts=sample))
                if counts == QUOTAS:
                    break
        print(source, counts, flush=True)
    if any(sum(r['split'] == split for r in rows) < limit for split, limit in QUOTAS.items()):
        raise ValueError("Insufficient preference examples")
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps({"source_revision": REVISION, "rows": rows}, ensure_ascii=False))


if __name__ == "__main__":
    main()
