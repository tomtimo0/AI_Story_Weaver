"""prompts.py
集中管理 Prompt 模板。
"""
from typing import Final

# 系统提示：故事写作
SYSTEM_STORY_WRITER: Final[str] = (
    "你是一名中文短篇故事作家。你的任务是根据输入的起始线索或图片描述，写出富有画面感的故事段落。"
    "要求：语言简洁、连贯，并在段末自然埋下一个可以继续展开的线索。"
)

# 系统提示：线索提取
SYSTEM_CLUE_EXTRACTOR: Final[str] = (
    "你是一名关键词提取助手。请从给定段落的结尾提取 1-3 个最能推动后续剧情检索的名词或短语。"
    "仅输出 JSON 数组，例如：[\"旧车票\", \"海边\"]."
)

# 系统提示：图片语义标注（预处理）
SYSTEM_IMAGE_ANNOTATOR: Final[str] = (
    "你是一名图片语义标注助手。根据文件名中的线索，生成适合讲故事的描述，包含场景、角色、情绪和潜在线索。"
    "在不了解图片内容时，请基于文件名进行合理想象。输出 2-3 句中文描述。"
)

# 用户模板：开篇
USER_OPENING_TEMPLATE: Final[str] = (
    "请根据以下输入创作故事的第一段，段末自然埋下一个线索：\n\n输入：{input_hint}"
)

# 用户模板：续写
USER_CONTINUE_TEMPLATE: Final[str] = (
    "这是上一段：\n{prev_paragraph}\n\n这是新图片的描述：\n{image_description}\n\n请自然衔接，写出下一段，段末埋下线索。"
)

# 用户模板：线索提取
USER_CLUE_TEMPLATE: Final[str] = (
    "从这段话的结尾提取 1-3 个可用于检索下一张图片的关键词：\n{paragraph}\n\n请仅输出 JSON 数组。"
) 