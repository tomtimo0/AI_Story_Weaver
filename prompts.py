"""prompts.py
集中管理 Prompt 模板。
"""
from typing import Final

# 系统提示：故事写作
SYSTEM_STORY_WRITER: Final[str] = (
    "你是一名专业的中文短篇故事作家。写作要求：\n"
    "1. 每段控制在150-300字，语言简洁生动\n"
    "2. 绝对不要重复相同的句子、对话或情节\n"
    "3. 每句话都要推进故事发展，避免冗余描述\n"
    "4. 注重人物对话和行动，减少环境描写\n"
    "5. 制造适度的戏剧冲突和悬念\n"
    "6. 直接输出故事内容，不要任何解释或标注"
)

# 系统提示：线索提取
SYSTEM_CLUE_EXTRACTOR: Final[str] = (
    "你是一名关键词提取助手。请从给定段落的结尾提取 1-3 个最能推动后续剧情检索的名词或短语。"
    "仅输出 JSON 数组，例如：[\"旧车票\", \"海边\"]."
)

# 系统提示：图片语义标注（预处理）
SYSTEM_IMAGE_ANNOTATOR: Final[str] = (
    "你是一名专业的图片描述助手。请简洁全面地描述图片内容，包括："
    "1. 主要物体/人物及其动作状态"
    "2. 场景环境和背景"
    "3. 色彩、光线、氛围特征"
    "4. 任何值得注意的细节"
    "用2-3句话概括，语言简洁准确，便于理解图片的核心内容。"
)

# 用户模板：开篇
USER_OPENING_TEMPLATE: Final[str] = (
    "请根据以下输入创作故事的第一段，段末自然埋下一个线索：\n\n输入：{input_hint}"
)

# 用户模板：续写
USER_CONTINUE_TEMPLATE: Final[str] = (
    "这是上一段：\n{prev_paragraph}\n\n这是新图片的描述：\n{image_description}\n\n请自然衔接，写出下一段，段末埋下线索。"
)

# 用户模板：线索提取（已弃用，保留兼容）
USER_CLUE_TEMPLATE: Final[str] = (
    "从这段话的结尾提取 1-3 个可用于检索下一张图片的关键词：\n{paragraph}\n\n请仅输出 JSON 数组。"
)

# 用户模板：图片选择（已弃用，保留兼容）
USER_IMAGE_SELECTION_TEMPLATE: Final[str] = (
    "这是当前故事段落：\n{current_paragraph}\n\n"
    "以下是候选图片及其描述：\n{candidate_images}\n\n"
    "请根据故事情节发展，选择最适合用于续写下一段的图片。"
    "仅输出选中图片的编号（如：1）。"
)

# 用户模板：图片序列选择
USER_SEQUENCE_SELECTION_TEMPLATE: Final[str] = (
    "故事主题/开端：{story_theme}\n"
    "目标故事长度：{target_length}段\n\n"
    "以下是候选图片及其描述：\n{candidate_images}\n\n"
    "请从这些图片中选择{target_length}张，组成一个富有戏剧性的故事序列。\n"
    "选择标准：\n"
    "1. 图片之间要有内在联系，能够构成完整的故事线\n"
    "2. 要包含冲突、转折或悬念元素\n"
    "3. 避免选择过于相似或平淡的图片\n"
    "4. 序列要符合起承转合的戏剧结构：\n"
    "   - 起：引入人物和情境，设置悬念\n"
    "   - 承：发展冲突，增加复杂性\n"
    "   - 转：情节转折，达到高潮\n"
    "   - 合：解决冲突，给出结局\n"
    "仅输出选中图片的编号，用逗号分隔（如：1,5,3,8）。"
)

# 用户模板：基于图片序列生成故事段落
USER_PARAGRAPH_FROM_IMAGE_TEMPLATE: Final[str] = (
    "故事背景：{story_context}\n"
    "当前段落：第{paragraph_index}段（共{total_paragraphs}段）\n"
    "当前图片描述：{image_description}\n"
    "上一段内容：{previous_paragraph}\n\n"
    "创作要求：\n"
    "1. 必须严格基于当前图片内容创作，图片中的元素要成为情节的核心\n"
    "2. 与上一段自然衔接，但要有明显的新发展\n"
    "3. 150-250字，语言简洁生动\n"
    "4. 如果是最后一段，要有完整结尾\n"
    "5. 绝对不能重复之前段落的情节、对话或描述\n"
    "\n"
    "直接输出故事段落："
) 