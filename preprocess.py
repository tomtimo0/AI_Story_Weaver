"""preprocess.py
预处理：遍历图片目录，为每张图片生成“故事化”的语义描述，并存入 ChromaDB。
- 优先使用多模态 API（通义千问 Qwen-VL / OpenAI 兼容 Vision）对图片自动描述；无可用多模态时回退到文件名占位描述。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from prompts import SYSTEM_IMAGE_ANNOTATOR
from utils import (
    EmbeddingService,
    LLMClient,
    add_image_document,
    get_or_create_collection,
    load_config,
    list_image_paths,
)


def _filename_to_hint(path: Path) -> str:
    name = path.stem
    name = re.sub(r"[_\-]+", " ", name)
    return name.strip() or "未命名"


def _llm_caption_image(llm: LLMClient, path: Path) -> Optional[str]:
    """使用多模态 API 对图片生成短描述。不可用时返回 None。"""
    instruction = "请用2-3句中文，描述图片的场景、角色/物体，以及可推动故事的细节线索。"
    return llm.chat_with_image(path, instruction)


def _llm_describe_from_hint(client: LLMClient, hint: str) -> str:
    user = f"文件名线索：{hint}\n请生成 2-3 句适合故事创作的中文描述。"
    return client.chat(SYSTEM_IMAGE_ANNOTATOR, user)


def _fallback_describe(hint: str) -> str:
    return (
        f"一张与‘{hint}’相关的图片：场景安静、光影柔和，带有轻微的怀旧情绪。"
        f" 画面中存在可继续讲述的细节线索。"
    )


def main() -> None:
    cfg = load_config()
    paths = list_image_paths(cfg.image_dir)
    if not paths:
        print(f"[预处理] 未发现图片，目录为空：{cfg.image_dir}")
        return

    print(f"[预处理] 共发现 {len(paths)} 张图片。")

    llm = LLMClient(cfg)
    embeddings = EmbeddingService(cfg.embedding_model_name)
    collection = get_or_create_collection(cfg.chroma_persist_dir, name="images")

    for idx, p in enumerate(paths, 1):
        # 1) 优先多模态 API
        desc: Optional[str] = _llm_caption_image(llm, p)

        # 2) 失败则尝试基于文件名的 LLM 文本生成
        if not desc:
            hint = _filename_to_hint(p)
            if cfg.openai_api_key or cfg.tongyi_api_key or cfg.siliconflow_api_key:
                try:
                    desc = _llm_describe_from_hint(llm, hint).strip()
                except Exception:
                    desc = None
        # 3) 仍失败则回退占位
        if not desc:
            hint = _filename_to_hint(p)
            desc = _fallback_describe(hint)

        doc = desc
        meta = {"path": str(p.as_posix()), "filename": p.name, "hint": _filename_to_hint(p)}
        vec = embeddings.encode([doc])[0]
        add_image_document(collection, vec, doc, meta)
        print(f"[预处理] ({idx}/{len(paths)}) 已入库：{p.name}")

    print("[预处理] 完成。向量库保存在：", cfg.chroma_persist_dir)


if __name__ == "__main__":
    main() 