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
    USER_IMAGE_SELECTION_TEMPLATE,
    USER_SEQUENCE_SELECTION_TEMPLATE,
    USER_PARAGRAPH_FROM_IMAGE_TEMPLATE,
)
from utils import (
    EmbeddingService,
    LLMClient,
    get_candidate_images,
    get_random_candidate_images,
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


def select_image_sequence(story_theme: str, target_length: int, n_candidates: int = 20) -> List[Tuple[str, dict]]:
    """选择完整的图片序列，让 LLM 从候选中选择符合故事结构的序列。"""
    # 随机获取更多候选图片
    candidates = get_random_candidate_images(collection, n_candidates=n_candidates, used_paths=set())
    if not candidates:
        return []
    
    # 构造候选图片描述
    candidate_descriptions = []
    for i, (doc, meta) in enumerate(candidates, 1):
        filename = meta.get("filename", "unknown")
        candidate_descriptions.append(f"{i}. {filename}: {doc}")
    
    candidate_text = "\n".join(candidate_descriptions)
    
    # 让 LLM 选择图片序列
    selection_prompt = USER_SEQUENCE_SELECTION_TEMPLATE.format(
        story_theme=story_theme,
        target_length=target_length,
        candidate_images=candidate_text
    )
    
    try:
        selection = llm.chat(SYSTEM_STORY_WRITER, selection_prompt).strip()
        # 解析选择的编号序列
        indices = [int(x.strip()) - 1 for x in selection.split(",")]
        selected_sequence = []
        for idx in indices:
            if 0 <= idx < len(candidates):
                selected_sequence.append(candidates[idx])
        
        # 确保序列长度符合要求
        if len(selected_sequence) == target_length:
            return selected_sequence
    except (ValueError, IndexError):
        pass
    
    # 回退：返回前 target_length 个候选
    return candidates[:target_length] if len(candidates) >= target_length else candidates


def generate_paragraph_for_image(story_context: str, paragraph_index: int, total_paragraphs: int, 
                                image_description: str, previous_paragraph: str = "", 
                                previous_paragraphs: List[str] = None) -> str:
    """基于指定图片生成对应的故事段落。"""
    if previous_paragraphs is None:
        previous_paragraphs = []
    
    prompt = USER_PARAGRAPH_FROM_IMAGE_TEMPLATE.format(
        story_context=story_context,
        paragraph_index=paragraph_index,
        total_paragraphs=total_paragraphs,
        image_description=image_description,
        previous_paragraph=previous_paragraph or "（这是开篇）"
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        # 限制生成长度，防止重复
        paragraph = llm.chat(SYSTEM_STORY_WRITER, prompt, max_tokens=400).strip()
        
        # 检测段落内重复
        if _has_repetitive_content(paragraph):
            st.warning(f"第{paragraph_index}段检测到内部重复（尝试{attempt+1}/{max_retries}）")
            continue
        
        # 检测与之前段落的重复
        if _is_similar_to_previous(paragraph, previous_paragraphs):
            st.warning(f"第{paragraph_index}段与前面段落重复（尝试{attempt+1}/{max_retries}）")
            # 添加更强的反重复提示
            prompt += f"\n\n重要提醒：这一段必须与前面的内容完全不同，要基于新的图片描述创作全新的情节。"
            continue
        
        # 如果通过所有检测，返回结果
        return paragraph
    
    # 如果多次重试都失败，使用最严格的提示重新生成
    st.error(f"第{paragraph_index}段多次生成重复，使用严格模式重新生成...")
    strict_prompt = f"""基于图片描述创作故事段落：
图片描述：{image_description}
段落位置：第{paragraph_index}段（共{total_paragraphs}段）
上一段：{previous_paragraph or "（开篇）"}

要求：
1. 严格基于当前图片内容创作，不要重复之前的情节
2. 与上一段自然衔接但情节要有新发展
3. 150-200字，语言简洁
4. 绝对不能重复之前出现过的对话或描述

直接输出故事段落："""
    
    return llm.chat(SYSTEM_STORY_WRITER, strict_prompt, max_tokens=300).strip()


def _has_repetitive_content(text: str, threshold: int = 3) -> bool:
    """检测文本中是否有重复的句子或段落。"""
    sentences = [s.strip() for s in text.split('。') if s.strip()]
    if len(sentences) < threshold:
        return False
    
    # 检查相同句子出现次数
    sentence_counts = {}
    for sentence in sentences:
        if len(sentence) > 10:  # 只检查较长的句子
            sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
    
    # 如果任何句子出现超过2次，认为是重复
    return any(count >= threshold for count in sentence_counts.values())


def _is_similar_to_previous(current_paragraph: str, previous_paragraphs: List[str], similarity_threshold: float = 0.7) -> bool:
    """检测当前段落是否与之前的段落过于相似。"""
    if not previous_paragraphs:
        return False
    
    # 将段落分解为句子
    current_sentences = set(s.strip() for s in current_paragraph.split('。') if s.strip() and len(s.strip()) > 5)
    
    for prev_paragraph in previous_paragraphs:
        prev_sentences = set(s.strip() for s in prev_paragraph.split('。') if s.strip() and len(s.strip()) > 5)
        
        # 计算句子重叠度
        if current_sentences and prev_sentences:
            overlap = len(current_sentences.intersection(prev_sentences))
            total_unique = len(current_sentences.union(prev_sentences))
            
            if total_unique > 0:
                similarity = overlap / total_unique
                if similarity > similarity_threshold:
                    return True
    
    return False


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
                st.image(img, caption=p.name, use_container_width=True)
    else:
        st.info("请将图片放入目录：" + cfg.image_dir)

# 故事生成区域
with col_right:
    st.subheader("生成结果")
    if run:
        # 确定故事主题
        if start_mode == "选择起始图片" and start_image_name and start_image_name != "(不选)":
            story_theme = f"以图片'{start_image_name}'为开端的故事"
        else:
            story_theme = start_topic or "一个普通但不平凡的夜晚"
        
        st.write(f"📖 故事主题：{story_theme}")
        st.write(f"📏 目标长度：{target_len}段")
        
        # 第一步：选择完整的图片序列
        with st.spinner("🎯 正在选择图片序列..."):
            image_sequence = select_image_sequence(story_theme, int(target_len))
        
        if not image_sequence:
            st.error("❌ 无法获取足够的图片，请检查图片库")
            st.stop()
        
        st.success(f"✅ 已选择 {len(image_sequence)} 张图片")
        
        # 显示选中的图片序列
        with st.expander("🖼️ 查看选中的图片序列"):
            cols = st.columns(min(len(image_sequence), 4))
            for i, (doc, meta) in enumerate(image_sequence):
                with cols[i % 4]:
                    try:
                        img_path = meta.get("path")
                        if img_path:
                            st.image(Image.open(img_path), caption=f"{i+1}. {meta.get('filename', 'unknown')}", use_container_width=True)
                        st.caption(doc[:100] + "..." if len(doc) > 100 else doc)
                    except Exception:
                        st.caption(f"{i+1}. {meta.get('filename', 'unknown')}")
        
        # 第二步：基于图片序列生成对应的故事段落
        story: List[Tuple[str, Optional[str]]] = []  # (paragraph, image_path)
        
        with st.spinner("✍️ 正在生成故事..."):
            for i, (doc, meta) in enumerate(image_sequence, 1):
                # 生成当前段落
                previous_paragraph = story[-1][0] if story else ""
                previous_paragraphs = [p[0] for p in story]  # 获取之前所有段落
                
                paragraph = generate_paragraph_for_image(
                    story_context=story_theme,
                    paragraph_index=i,
                    total_paragraphs=len(image_sequence),
                    image_description=doc,
                    previous_paragraph=previous_paragraph,
                    previous_paragraphs=previous_paragraphs
                )
                
                image_path = meta.get("path")
                story.append((paragraph, image_path))
                
                # 显示生成进度
                st.write(f"✅ 第 {i} 段已生成")

        # 展示结果
        for idx, (para, img_path) in enumerate(story, 1):
            st.markdown(f"**第 {idx} 段**")
            if img_path:
                try:
                    st.image(Image.open(img_path), use_container_width=True)
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