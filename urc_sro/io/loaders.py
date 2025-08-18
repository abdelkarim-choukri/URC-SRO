from __future__ import annotations
import os, json
from typing import List
from datetime import datetime
from ..types import Document, Metadata

def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def _read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def load_corpus_from_dir(root: str) -> List[Document]:
    """
    Load a corpus from a directory:
      - *.txt  -> id = filename (without ext), text = file content
      - *.jsonl (each line needs at least {"id":..., "text":...})
    """
    docs: List[Document] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if name.lower().endswith(".txt"):
                doc_id = os.path.splitext(name)[0]
                docs.append(Document(
                    id=doc_id,
                    text=_read_txt(path),
                    meta=Metadata(source="fs", uri=os.path.abspath(path), timestamp=datetime.fromtimestamp(os.path.getmtime(path)))
                ))
            elif name.lower().endswith(".jsonl"):
                for row in _read_jsonl(path):
                    if "id" in row and "text" in row:
                        docs.append(Document(
                            id=str(row["id"]),
                            text=str(row["text"]),
                            meta=Metadata(source="fs", uri=os.path.abspath(path))
                        ))
    return docs
