import logging
import re
from typing import Dict, List

from openai import OpenAI

logger = logging.getLogger(__name__)

# 分隔符：独占一行的 3 个及以上连字符，前后允许有空白。
# 按行匹配而不是子串切分，这样 "----" 不会切出孤零零的 "-"，
# 正文里夹在行内的 "---" 也不会被误当成分段点。
_DELIMITER_RE = re.compile(r"^[ \t]*-{3,}[ \t]*$")

# 邮件主题常见的前缀标签，如「主题：」「Subject:」。
_SUBJECT_PREFIX_RE = re.compile(r"^\s*(主题|subject)\s*[:：]\s*", re.IGNORECASE)


class FeedbackGenerationError(Exception):
    """模型输出不可用（内容为空 None / 被 max_tokens 截断）时抛出。

    消息是可读的中文原因，上游把它拼进「生成反馈失败: {...}」后，
    前端和运维能看到有意义的说明，而不是 'NoneType' object has no attribute 'strip'。
    """


def _split_segments(content: str, max_segments: int) -> List[str]:
    """按分隔线切分，最多切出 max_segments 段。

    只有前 max_segments - 1 条分隔线被当作分段点；之后出现的横线
    （比如第三段温暖评语里自带的 Markdown 水平线）原样保留在最后一段里。
    """
    segments = []
    current: List[str] = []
    for line in content.split("\n"):
        if _DELIMITER_RE.match(line) and len(segments) < max_segments - 1:
            segments.append("\n".join(current))
            current = []
        else:
            current.append(line)
    segments.append("\n".join(current))
    return [segment.strip() for segment in segments]


def _extract_improvements(text: str) -> str:
    """从第一段里切出「改进 / 建议」部分。

    切点落在标题所在行的行首（标题的开头），不是「改进」「建议」
    这两个词中间；找不到标题时返回空串，不把整段技术亮点塞进来。
    """
    positions = [pos for pos in (text.find("改进"), text.find("建议")) if pos != -1]
    if not positions:
        return ""
    line_start = text.rfind("\n", 0, min(positions)) + 1
    return text[line_start:].strip()


def _clean_subject(text: str) -> str:
    """把模型给出的主题行清洗成可直接使用的单行邮件主题。"""
    for line in text.splitlines():
        line = _SUBJECT_PREFIX_RE.sub("", line).strip()
        if line:
            return line
    return ""


def _fallback_subject(student_names: str, project_title: str) -> str:
    """模型没给主题（或主题为空）时的兜底主题，保证单行且与正文不同。"""
    return f"{student_names}的编程夏令营作品《{project_title}》展示反馈"


class AIFeedbackGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _extract_content(response) -> str:
        """从响应里取出文本，对 content=None 和截断输出给出确定处理。"""
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise FeedbackGenerationError(
                "模型输出被 max_tokens 截断（finish_reason=length），"
                "评语停在半句，不将半成品落库"
            )
        content = choice.message.content
        if content is None:
            raise FeedbackGenerationError(
                "模型未返回文本内容（content=None），可能被内容策略拦截"
            )
        return content.strip()

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
        # 老数据里可能混进 None 或非字符串元素：只保留字符串，其余跳过。
        if mentor_comments:
            mentor_comments_text = "\n".join(
                comment for comment in mentor_comments if isinstance(comment, str)
            )
        else:
            mentor_comments_text = ""

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

        content = self._extract_content(response)
        if not content:
            logger.warning("模型返回空字符串输出，四个字段全部降级为空串")

        segments = _split_segments(content, max_segments=3)
        technical = segments[0]
        summary = segments[1] if len(segments) > 1 else ""
        warm = segments[2] if len(segments) > 2 else ""

        if len(segments) < 3:
            # 模型压根没写第三段（分隔线不够）。
            logger.warning(
                "模型只输出了 %d 段，缺少第三段（老师的话），warm_feedback 置空",
                len(segments),
            )
        elif not warm:
            # 模型写了第三段（分隔线够）但内容是空的。
            logger.warning("模型输出了第三段但内容为空，warm_feedback 置空")
        if len(segments) < 2:
            logger.warning(
                "模型只输出了 %d 段，缺少第二段（项目摘要），ai_summary 置空",
                len(segments),
            )

        return {
            'technical_feedback': technical,
            'improvement_suggestions': _extract_improvements(technical),
            'ai_summary': summary,
            'warm_feedback': warm,
        }

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

        content = self._extract_content(response)
        segments = _split_segments(content, max_segments=2)

        if len(segments) > 1:
            # 第一条分隔线之前是主题，之后的全部内容（包括正文自带的
            # 水平线）都是正文。
            subject = _clean_subject(segments[0])
            body = segments[1]
        else:
            logger.warning("模型未输出主题/正文分隔符，使用兜底主题，全部输出作为正文")
            subject = ""
            body = segments[0]

        if not subject:
            subject = _fallback_subject(student_names, project_title)

        return {
            'subject': subject,
            'body_html': body,
        }
