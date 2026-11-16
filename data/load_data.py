import os, shutil, tempfile, time

from huggingface_hub import hf_hub_download


def fetch(job):
    repo, revision, kind, files, size = job
    done = 0
    for file, size in zip(files, size):
        destination = os.path.join("/data/", kind, file)
        if os.path.exists(destination) and os.path.getsize(destination) == size:
            continue
        with tempfile.TemporaryDirectory() as temp_dir:
            path = hf_hub_download(
                repo_id=repo, filename=file, repo_type="dataset", revision=revision, local_dir=temp_dir
            )
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copyfile(path, destination + ".part"); os.replace(destination + ".part", destination)
        done += size
    vol.commit()
    return done
