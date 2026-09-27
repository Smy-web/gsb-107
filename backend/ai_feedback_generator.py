import logging
import re
from typing import Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)

# 分隔符：独占一行，行内除空白外只有 3 个及以上连字符（---、----、----- 都算）。
_SEPARATOR_RE = re.compile(r"^\s*-{3,}\s*$")

# 邮件主题前面的标签，例如「主题：」「Subject:」。
_SUBJECT_LABEL_RE = re.compile(r"^(主题|subject)\s*[:：]\s*", re.IGNORECASE)

# 标题行前面可能出现的列表 / 编号前缀，例如「- 」「2. 」「二、」。
_LIST_MARKER_RE = re.compile(r"^[\s\-\*•\d\.、\)）]+")


class FeedbackGenerationError(Exception):
    """模型输出不可用（为空、被内容策略拦截、被 max_tokens 截断）时抛出。

    消息是可读的中文原因，上游把它拼进「生成反馈失败: {...}」时运维能看懂。
    """


def _split_sections(content: str, max_sections: int) -> List[str]:
    """按分隔符行切分，最多切出 max_sections 段。

    只认前 max_sections - 1 个分隔符行；之后出现的分隔线（例如第三段温暖
    评语里自带的 Markdown 水平线）原样保留在最后一段里，不再当分段点。
    """
    lines = content.split("\n")
    sep_positions = [i for i, line in enumerate(lines) if _SEPARATOR_RE.match(line)]
    cuts = sep_positions[: max_sections - 1]
    parts = []
    start = 0
    for pos in cuts:
        parts.append("\n".join(lines[start:pos]).strip())
        start = pos + 1
    parts.append("\n".join(lines[start:]).strip())
    return parts


class AIFeedbackGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _require_content(response, context: str) -> str:
        """取出模型输出文本；输出不可用时抛 FeedbackGenerationError。"""
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            raise FeedbackGenerationError(
                f"{context}：模型输出被 max_tokens 截断（finish_reason=length），"
                "半成品不允许落库，请调大 max_tokens 后重试"
            )
        content = getattr(choice.message, "content", None)
        if content is None:
            raise FeedbackGenerationError(
                f"{context}：模型返回的 content 是 None（可能被内容策略拦截），没有可用输出"
            )
        content = content.strip()
        if not content:
            raise FeedbackGenerationError(f"{context}：模型返回了空字符串，没有可用输出")
        return content

    @staticmethod
    def _sanitize_mentor_comments(mentor_comments) -> List[str]:
        """老数据里可能混进 None 或非字符串元素：跳过并告警，不让整个接口崩。"""
        comments = []
        for item in mentor_comments or []:
            if isinstance(item, str) and item.strip():
                comments.append(item)
            elif item is not None:
                logger.warning("跳过非字符串的导师评语元素: %r", item)
        return comments

    def generate_feedback(
        self,
        project_title: str,
        project_description: str,
        student_names: str,
        project_type: str,
        transcript: str,
        technical_feedback: str = "",
        mentor_comments: List[str] = None
    ) -> Dict:
        comments = self._sanitize_mentor_comments(mentor_comments)
        mentor_comments_text = "\n".join(comments)

        summary_prompt = f"""
        请为以下青少年编程夏令营项目展示生成内容：

        项目信息：
        - 项目名称：{project_title}
        - 项目描述：{project_description}
        - 参与学生：{student_names}
        - 项目类型：{project_type}

        展示会文字记录（按说话人区分）：
        {transcript}

        导师技术点评：
        {technical_feedback}

        其他导师评语：
        {mentor_comments_text}

        请输出以下三部分内容，用---分隔：

        第一部分：技术亮点和改进建议（用中文，分点列出，针对青少年水平）
        
        第二部分：项目内容摘要（150-200字，用中文，简洁明了）
        
        第三部分：温暖鼓励的评语（给家长看的，300-400字，用中文，语气亲切温暖，
        要具体提到孩子的表现亮点，给予鼓励和肯定，同时温和地提出可以继续努力的方向）
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "你是一位充满爱心和智慧的青少年编程教育专家，擅长用温暖鼓励的语言评价孩子们的作品，同时能给出专业的技术建议。"
                },
                {"role": "user", "content": summary_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )

        content = self._require_content(response, "生成评语")
        parts = _split_sections(content, max_sections=3)

        if len(parts) < 2:
            logger.warning(
                "生成评语：模型只输出了 %d 段，缺少第二段（项目摘要）和第三段（温暖评语），对应字段置空",
                len(parts),
            )
        elif len(parts) < 3:
            logger.warning(
                "生成评语：模型只输出了 2 段，没写第三段（温暖评语），warm_feedback 置空"
            )

        technical = parts[0]
        summary = parts[1] if len(parts) >= 2 else ""
        warm = parts[2] if len(parts) >= 3 else ""

        if len(parts) >= 3 and not warm:
            logger.warning("生成评语：模型写了第三段但内容为空，warm_feedback 置空")
        if len(parts) >= 2 and not summary:
            logger.warning("生成评语：模型写了第二段但内容为空，ai_summary 置空")

        return {
            'technical_feedback': technical,
            'improvement_suggestions': self._extract_improvements(technical),
            'ai_summary': summary,
            'warm_feedback': warm,
        }

    def _extract_improvements(self, text: str) -> str:
        """从第一段里切出「改进 / 建议」部分。

        切点落在标题的开头：优先找去掉列表/编号前缀后以「改进」或「建议」
        开头的标题行；找不到就退而求其次，找第一处出现这两个词的行。
        切点取该行内「改进」「建议」最早出现的位置，不会从标题中间下刀。
        """
        if not text:
            return ""
        lines = text.split("\n")
        heading_idx = None
        for i, line in enumerate(lines):
            stripped = _LIST_MARKER_RE.sub("", line)
            if stripped.startswith("改进") or stripped.startswith("建议"):
                heading_idx = i
                break
        if heading_idx is None:
            for i, line in enumerate(lines):
                if "改进" in line or "建议" in line:
                    heading_idx = i
                    break
        if heading_idx is None:
            return text.strip()
        line = lines[heading_idx]
        positions = [p for p in (line.find("改进"), line.find("建议")) if p >= 0]
        start = min(positions)
        return "\n".join([line[start:]] + lines[heading_idx + 1:]).strip()

    def generate_email_content(
        self,
        student_names: str,
        project_title: str,
        warm_feedback: str,
        ai_summary: str
    ) -> Dict:
        email_prompt = f"""
        请为以下学生家长生成一封温馨的电子邮件：

        学生姓名：{student_names}
        项目名称：{project_title}
        项目摘要：{ai_summary}
        老师评语：{warm_feedback}

        请生成：
        1. 邮件主题（温馨亲切）
        2. 邮件正文（HTML格式，包含问候、项目介绍、老师评语、鼓励话语）
        
        用---分隔主题和正文。
        """

        response = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "你是一位亲切的青少年编程夏令营班主任，擅长与家长沟通，用温暖的语言分享孩子的成长。"
                },
                {"role": "user", "content": email_prompt}
            ],
            temperature=0.8,
            max_tokens=1500
        )

        content = self._require_content(response, "生成家长邮件")
        parts = _split_sections(content, max_sections=2)

        if len(parts) >= 2:
            raw_subject, body = parts[0], parts[1]
        else:
            # 模型没给分隔符：全文当正文，主题用项目名兜底，
            # 保证主题和正文不是同一段内容。
            logger.warning("生成家长邮件：模型输出没有分隔符，主题用项目名兜底")
            raw_subject, body = "", parts[0]

        subject = self._clean_subject(raw_subject)
        if not subject:
            subject = f"《{project_title}》项目展示反馈"

        return {
            'subject': subject,
            'body_html': body,
        }

    @staticmethod
    def _clean_subject(raw: str) -> str:
        """主题必须是单行：去掉「主题：」一类前缀标签，取第一个非空行。"""
        for line in raw.split("\n"):
            line = _SUBJECT_LABEL_RE.sub("", line.strip()).strip()
            if line:
                return line
        return ""
