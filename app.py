"""app.py
最小可运行的 Streamlit 应用：
- 选择起始图片或输入主题
- 生成开篇 → 线索提取 → 向量检索 → 续写，直至达到目标段数
- 无向量库或无匹配时，采用回退策略继续文本生成
"""
from __future__ import annotations

import json
from typing import List, Optional, Set, Tuple

import streamlit as st
from PIL import Image

from prompts import (
    SYSTEM_CLUE_EXTRACTOR,
    SYSTEM_STORY_WRITER,
    USER_CLUE_TEMPLATE,
    USER_CONTINUE_TEMPLATE,
    USER_OPENING_TEMPLATE,
)
from utils import (
    EmbeddingService,
    LLMClient,
    get_or_create_collection,
    list_image_paths,
    load_config,
    query_similar_documents,
    safe_open_image,
)


st.set_page_config(page_title="AI 故事编织者", page_icon="📖", layout="wide")

cfg = load_config()
llm = LLMClient(cfg)
emb = EmbeddingService(cfg.embedding_model_name)
collection = get_or_create_collection(cfg.chroma_persist_dir, name="images")


def extract_clues(paragraph: str) -> List[str]:
    """从段落结尾提取 1-3 个关键词，用于检索下一张图片。"""
    prompt = USER_CLUE_TEMPLATE.format(paragraph=paragraph)
    text = llm.chat(SYSTEM_CLUE_EXTRACTOR, prompt).strip()
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            return [str(x) for x in arr][:3]
    except Exception:
        pass
    # 回退：取最后一句中的名词样式词（简单近似）
    last = paragraph.split("。")[-2:]  # 取末尾句子片段
    fallback = "，".join(last).strip()
    return [fallback[:16]] if fallback else ["线索"]


def gen_opening(input_hint: str) -> str:
    user = USER_OPENING_TEMPLATE.format(input_hint=input_hint)
    return llm.chat(SYSTEM_STORY_WRITER, user).strip()


def gen_continuation(prev_paragraph: str, image_description: str) -> str:
    user = USER_CONTINUE_TEMPLATE.format(
        prev_paragraph=prev_paragraph, image_description=image_description
    )
    return llm.chat(SYSTEM_STORY_WRITER, user).strip()


def retrieve_next_image(clues: List[str]) -> Optional[Tuple[str, dict]]:
    """基于线索向量检索候选图片，返回 (document, metadata)。找不到则 None。"""
    query = "; ".join(clues)
    vec = emb.encode([query])[0]
    docs, metas = query_similar_documents(collection, vec, n_results=5)
    if not docs:
        return None
    # 简单返回第一个候选
    return docs[0], metas[0]


# ----------------------------
# UI
# ----------------------------

st.title("📖 AI 故事编织者")

with st.sidebar:
    st.header("设置")
    images = list_image_paths(cfg.image_dir)
    image_names = [p.name for p in images]
    start_mode = st.radio("选择开端方式", ["选择起始图片", "输入主题"], index=0)

    start_image_name = None
    start_topic = None
    if start_mode == "选择起始图片":
        start_image_name = st.selectbox("起始图片", options=["(不选)"] + image_names, index=0)
    else:
        start_topic = st.text_input("故事主题", placeholder="例如：雨后的城市、遗失的车票……")

    target_len = st.number_input("目标段落数", min_value=3, max_value=12, value=cfg.story_target_length, step=1)
    run = st.button("开始生成", type="primary")

col_left, col_right = st.columns([1, 1])

# 展示图片库预览
with col_left:
    st.subheader("图片库预览")
    if images:
        thumbs = images[:8]
        for p in thumbs:
            img = safe_open_image(p)
            if img is not None:
                st.image(img, caption=p.name, use_column_width=True)
    else:
        st.info("请将图片放入目录：" + cfg.image_dir)

# 故事生成区域
with col_right:
    st.subheader("生成结果")
    if run:
        used_paths: Set[str] = set()
        story: List[Tuple[str, Optional[str]]] = []  # (paragraph, image_path)

        # 开篇
        if start_mode == "选择起始图片" and start_image_name and start_image_name != "(不选)":
            p = next((x for x in images if x.name == start_image_name), None)
            if p is not None:
                opening_hint = f"起始图片：{p.name}"
                first_paragraph = gen_opening(opening_hint)
                story.append((first_paragraph, str(p)))
                used_paths.add(str(p))
            else:
                first_paragraph = gen_opening(start_image_name)
                story.append((first_paragraph, None))
        else:
            topic = start_topic or "一个普通但不平凡的夜晚"
            first_paragraph = gen_opening(topic)
            story.append((first_paragraph, None))

        # 循环续写
        while len(story) < int(target_len):
            last_paragraph = story[-1][0]
            clues = extract_clues(last_paragraph)
            candidate = retrieve_next_image(clues)

            if candidate is None:
                # 回退：无图也续写
                next_para = gen_continuation(last_paragraph, "（无新图片）")
                story.append((next_para, None))
                continue

            doc, meta = candidate
            image_path = meta.get("path")
            if image_path in used_paths:
                # 避免重复，回退
                next_para = gen_continuation(last_paragraph, doc)
                story.append((next_para, None))
                continue

            next_para = gen_continuation(last_paragraph, doc)
            story.append((next_para, image_path))
            used_paths.add(image_path)

        # 展示结果
        for idx, (para, img_path) in enumerate(story, 1):
            st.markdown(f"**第 {idx} 段**")
            if img_path:
                try:
                    st.image(Image.open(img_path), use_column_width=True)
                except Exception:
                    st.caption(f"[图片无法显示] {img_path}")
            st.write(para)
            st.divider()

        # 导出（简单）
        if st.download_button(
            label="导出 Markdown",
            data="\n\n".join([p for p, _ in story]).encode("utf-8"),
            file_name="story.md",
            mime="text/markdown",
        ):
            st.toast("已导出 story.md") 