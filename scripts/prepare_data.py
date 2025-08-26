import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import requests

SMD_ENTITIES = ([f"machine-1-{i}" for i in range(1, 9)] +
                [f"machine-2-{i}" for i in range(1, 10)] +
                [f"machine-3-{i}" for i in range(1, 12)])


def download(url, proxy=None):
    proxies = {"http": proxy, "https": proxy} if proxy else None
    for attempt in range(3):
        try:
            response = requests.get(url, proxies=proxies, timeout=(20, 180))
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", nargs="+", default=["machine-1-1"], help="SMD entities or all")
    parser.add_argument("--output", type=Path, default=Path("data/Machine"))
    parser.add_argument("--proxy", default=None)
    parser.add_argument("--revision", default="master", help="OmniAnomaly Git revision; resolved to a commit when possible")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    entities = SMD_ENTITIES if args.entities == ["all"] else args.entities
    if any(e not in SMD_ENTITIES for e in entities):
        parser.error("Unknown SMD entity")
    revision = args.revision
    if revision == "master":
        try:
            revision = json.loads(download("https://api.github.com/repos/NetManAIOps/OmniAnomaly/commits/master", args.proxy))["sha"]
        except (requests.RequestException, KeyError, ValueError):
            print("Could not resolve master; content SHA256 will still be recorded", flush=True)
    base = f"https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/{revision}/ServerMachineDataset"
    def prepare(entity):
        manifest = {"dataset": "SMD", "entity": entity, "revision": revision, "files": {}}
        for suffix in ["train", "test", "test_label"]:
            out = args.output / f"{entity}_{suffix}.npy"
            url = f"{base}/{suffix}/{entity}.txt"
            if not out.exists():
                content = download(url, args.proxy)
                data = np.loadtxt(io.BytesIO(content), delimiter=",", dtype=np.float32)
                if suffix == "test_label":
                    if not np.isin(data, [0, 1]).all():
                        raise ValueError("Invalid labels downloaded")
                    data = data.astype(np.int64)
                np.save(out, data)
            data = np.load(out)
            manifest["files"][suffix] = {"url": url, "shape": list(data.shape),
                                            "npy_sha256": hashlib.sha256(out.read_bytes()).hexdigest()}
        path = args.output / f"{entity}_manifest.json"
        if not path.exists():
            path.write_text(json.dumps(manifest, indent=2))
        print(f"Prepared {entity}", flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(prepare, entities))


if __name__ == "__main__":
    main()
