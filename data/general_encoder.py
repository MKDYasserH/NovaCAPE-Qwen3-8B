"""Encode general training examples once with the same frozen encoder used in evaluation."""
import hashlib
import json

import modal

from data.qwen_service import encoder_service, resources, root, volume

app = modal.App("nova-general-preference-embedding")


@app.function(**resources, timeout=1800, max_containers=1)
def cache_preferences(stem: str):
    import numpy as np
    from clm.embedder import Embedder

    volume.reload()
    folder = root / "post_inputs"
    source = folder / (stem + ".json")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = folder / (stem + ".manifest.json")
    if manifest.is_file() and json.loads(manifest.read_text())["sha256"] == digest:
        return
    rows = json.loads(source.read_text())["rows"]
    texts = list(dict.fromkeys(text for row in rows for text in row["texts"]))
    lookup = {s: i for i, s in enumerate(texts)}
    with encoder_service():
        vectors, tokens = Embedder(batch=64).embed(texts)
    if vectors.shape != (len(texts), 4096) or not np.isfinite(vectors).all():
        raise ValueError("Incomplete preference embeddings")
    output = folder / (stem + ".npz")
    with output.with_suffix(".partial").open("wb") as stream:
        np.savez(stream, vectors=vectors.astype(np.float16),
                 indices=np.array([[lookup[s] for s in r["texts"]] for r in rows]),
                 train=np.array([r["split"] == "train" for r in rows]))
    output.with_suffix(".partial").replace(output)
    manifest.write_text(json.dumps(dict(sha256=digest, rows=len(rows), texts=len(texts), tokens=tokens)))
    volume.commit()
    print(manifest.read_text(), flush=True)
