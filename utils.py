"""utils.py
通用工具函数：环境变量、LLM 调用、Embedding、ChromaDB 读写与检索。
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import chromadb
from chromadb.api.models.Collection import Collection
from dotenv import load_dotenv
from PIL import Image
from sentence_transformers import SentenceTransformer

# OpenAI 兼容客户端（OpenAI / 硅基流动等）
try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

# 通义千问（DashScope）
try:
    import dashscope  # type: ignore
except Exception:  # pragma: no cover
    dashscope = None  # type: ignore


# ----------------------------
# 环境与配置
# ----------------------------

@dataclass
class AppConfig:
    # 通用
    llm_provider: str
    llm_model: str
    temperature: float

    # OpenAI 及兼容（OpenAI / 硅基流动）
    openai_api_key: Optional[str]
    openai_base_url: Optional[str]

    # 硅基流动
    siliconflow_api_key: Optional[str]
    siliconflow_base_url: str

    # 通义千问（DashScope）
    tongyi_api_key: Optional[str]

    # 多模态图片描述模型
    image_caption_model: str

    # Embedding / 数据
    embedding_model_name: str
    chroma_persist_dir: str
    image_dir: str
    story_target_length: int


def load_config() -> AppConfig:
    """加载 .env 与环境变量，返回应用配置。"""
    load_dotenv(override=True)
    return AppConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "openai").lower(),
        llm_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=float(os.getenv("TEMPERATURE", "0.3")),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
        siliconflow_api_key=os.getenv("SILICONFLOW_API_KEY"),
        siliconflow_base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
        tongyi_api_key=os.getenv("DASHSCOPE_API_KEY") or os.getenv("TONGYI_API_KEY"),
        image_caption_model=os.getenv("IMAGE_CAPTION_MODEL", "qwen-vl-plus"),
        embedding_model_name=os.getenv(
            "TEXT_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"
        ),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        image_dir=os.getenv("IMAGE_DIR", "./assets/images"),
        story_target_length=int(os.getenv("STORY_TARGET_LENGTH", "5")),
    )


# ----------------------------
# LLM 封装与回退
# ----------------------------

class LLMClient:
    """封装多厂商 LLM 客户端。

    支持：
    - openai（默认）: 官方 OpenAI 或其它 OpenAI 兼容网关（可通过 OPENAI_BASE_URL 指定）
    - siliconflow: 使用 OpenAI 兼容接口（base_url=https://api.siliconflow.cn/v1）
    - tongyi: 使用 DashScope SDK（通义千问）
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self._client = None
        self._provider = config.llm_provider

        if self._provider == "openai":
            if OpenAI and (config.openai_api_key is not None):
                try:
                    self._client = OpenAI(
                        api_key=config.openai_api_key,
                        base_url=config.openai_base_url or None,
                    )
                except Exception:
                    self._client = None
        elif self._provider == "siliconflow":
            if OpenAI and (config.siliconflow_api_key is not None):
                try:
                    self._client = OpenAI(
                        api_key=config.siliconflow_api_key,
                        base_url=config.siliconflow_base_url,
                    )
                except Exception:
                    self._client = None
        elif self._provider == "tongyi":
            if dashscope and (config.tongyi_api_key is not None):
                dashscope.api_key = config.tongyi_api_key
                self._client = "dashscope"  # 占位标识

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        """对话式生成文本；若无可用 LLM，则回退到简单模板。"""
        model = self.config.llm_model
        temp = self.config.temperature

        try:
            if self._provider in {"openai", "siliconflow"} and self._client is not None:
                completion = self._client.chat.completions.create(
                    model=model,
                    temperature=temp,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return completion.choices[0].message.content or ""

            if self._provider == "tongyi" and self._client == "dashscope":
                # 通义千问（DashScope）消息体
                if dashscope is None:
                    raise RuntimeError("dashscope SDK 未安装")
                from dashscope import Generation  # type: ignore

                response = Generation.call(
                    model=model or "qwen-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    result_format="message",
                    temperature=temp,
                )
                # 兼容不同返回结构
                try:
                    return (
                        response.get("output", {})
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                except Exception:
                    pass
        except Exception:
            pass

        # 回退策略：拼接与简易扩写
        return _fallback_complete(system_prompt, user_prompt)

    def chat_with_image(self, image_path: Path, user_instruction: str) -> Optional[str]:
        """多模态对话：给定本地图片与用户指令，返回生成文本。失败返回 None。"""
        try:
            if self._provider in {"openai", "siliconflow"} and self._client is not None:
                data_url = encode_image_to_data_url(image_path)
                completion = self._client.chat.completions.create(
                    model=self.config.llm_model,
                    temperature=self.config.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": user_instruction},
                                {"type": "image_url", "image_url": {"url": data_url}},
                            ],
                        }
                    ],
                )
                return completion.choices[0].message.content or ""

            if self._provider == "tongyi" and self._client == "dashscope":
                from dashscope import MultiModalConversation  # type: ignore

                abs_path = image_path.resolve().as_posix()
                response = MultiModalConversation.call(
                    model=self.config.image_caption_model or "qwen-vl-plus",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"image": f"file://{abs_path}"},
                                {"text": user_instruction},
                            ],
                        }
                    ],
                )
                # 兼容不同返回
                try:
                    # 新版本 content 为列表
                    content = (
                        response.get("output", {})
                        .get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", [])
                    )
                    if isinstance(content, list) and content:
                        # 寻找首个 text 块
                        for blk in content:
                            if isinstance(blk, dict) and blk.get("text"):
                                return blk["text"]
                    # 旧版可能为字符串
                    if isinstance(content, str):
                        return content
                except Exception:
                    pass
        except Exception:
            return None
        return None


def _fallback_complete(system_prompt: str, user_prompt: str) -> str:
    """离线回退：基于简易规则生成可读文本，确保 demo 可运行。"""
    hint = user_prompt.strip().splitlines()[-1][:80]
    return f"（离线模式）基于线索：{hint}，继续讲述一个连贯温暖的片段，并在结尾留下悬念……"


# ----------------------------
# Embedding 与 ChromaDB
# ----------------------------

class EmbeddingService:
    """管理文本向量化模型。"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _ensure(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._ensure()
        vectors = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
        return [v.tolist() for v in vectors]


def get_or_create_collection(persist_dir: str, name: str = "images") -> Collection:
    """获取或创建 ChromaDB 集合。"""
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(name=name)


def add_image_document(
    collection: Collection,
    embeddings: List[float],
    document: str,
    metadata: dict,
    doc_id: Optional[str] = None,
) -> str:
    """向集合中添加图片描述文档。"""
    if doc_id is None:
        doc_id = _stable_id(document + json.dumps(metadata, ensure_ascii=False))
    collection.add(ids=[doc_id], embeddings=[embeddings], documents=[document], metadatas=[metadata])
    return doc_id


def query_similar_documents(
    collection: Collection,
    query_embedding: List[float],
    n_results: int = 5,
) -> Tuple[List[str], List[dict]]:
    """基于向量查询相似文档，返回 (documents, metadatas)。"""
    result = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    docs: List[str] = result.get("documents", [[]])[0] or []
    metas: List[dict] = result.get("metadatas", [[]])[0] or []
    return docs, metas


def _stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ----------------------------
# 图片工具
# ----------------------------

def list_image_paths(image_dir: str) -> List[Path]:
    """列出图片路径（jpg/png/jpeg），按文件名排序。"""
    p = Path(image_dir)
    if not p.exists():
        return []
    exts = {".jpg", ".jpeg", ".png"}
    paths = [x for x in p.iterdir() if x.suffix.lower() in exts and x.is_file()]
    return sorted(paths, key=lambda x: x.name)


def safe_open_image(path: Path) -> Optional[Image.Image]:
    try:
        return Image.open(path)
    except Exception:
        return None


def encode_image_to_data_url(path: Path) -> str:
    """将本地图片编码为 data URL（用于 OpenAI 兼容的 Vision 输入）。"""
    ext = path.suffix.lower().replace(".", "") or "jpeg"
    mime = f"image/{'jpeg' if ext in {'jpg', 'jpeg'} else ext}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}" 