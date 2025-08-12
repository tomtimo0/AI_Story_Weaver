"""inspect_chroma.py
查看 ChromaDB 向量库内容的小工具。

用法示例：
- 列出前 10 条：
  python inspect_chroma.py --limit 10
- 按文件名过滤（元数据）：
  python inspect_chroma.py --where '{"filename":"xxx.jpg"}'
- 文本查询（向量检索）：
  python inspect_chroma.py --query "旧车票" --limit 5

注意：示例依赖 `.env` 的 `CHROMA_PERSIST_DIR` 与 `TEXT_EMBEDDING_MODEL`。
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from utils import EmbeddingService, get_or_create_collection, load_config


def pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def list_docs(limit: int, where: Dict[str, Any] | None) -> None:
    cfg = load_config()
    col = get_or_create_collection(cfg.chroma_persist_dir, name="images")
    result = col.get(limit=limit, where=where, include=["metadatas", "documents"])  # type: ignore[arg-type]
    ids = result.get("ids", [])
    docs = result.get("documents", [])
    metas = result.get("metadatas", [])
    total = len(ids)

    print(f"共返回 {total} 条（limit={limit}）\n")
    for i in range(total):
        print(f"[{i+1}] id={ids[i]}")
        print("  meta:")
        print(pretty(metas[i]))
        print("  doc:")
        print(docs[i])
        print("-")


def query_text(query: str, limit: int) -> None:
    cfg = load_config()
    emb = EmbeddingService(cfg.embedding_model_name)
    col = get_or_create_collection(cfg.chroma_persist_dir, name="images")

    qv = emb.encode([query])[0]
    res = col.query(
        query_embeddings=[qv],
        n_results=limit,
        include=["documents", "metadatas", "distances"],
    )
    docs: List[str] = res.get("documents", [[]])[0] or []
    metas: List[dict] = res.get("metadatas", [[]])[0] or []
    dists: List[float] = res.get("distances", [[]])[0] or []

    print(f"查询：{query} ；返回 {len(docs)} 条\n")
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        dist = dists[i - 1] if i - 1 < len(dists) else None
        print(f"[{i}] distance={dist}")
        print("  meta:")
        print(pretty(meta))
        print("  doc:")
        print(doc)
        print("-")


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect ChromaDB contents")
    ap.add_argument("--limit", type=int, default=10, help="返回条数上限")
    ap.add_argument("--where", type=str, default=None, help="元数据过滤 JSON，如 {\"filename\":\"a.jpg\"}")
    ap.add_argument("--query", type=str, default=None, help="文本查询内容（向量检索）")
    args = ap.parse_args()

    where_obj = None
    if args.where:
        try:
            where_obj = json.loads(args.where)
        except Exception as e:
            raise SystemExit(f"where 解析失败：{e}")

    if args.query:
        query_text(args.query, args.limit)
    else:
        list_docs(args.limit, where_obj)


if __name__ == "__main__":
    main()
