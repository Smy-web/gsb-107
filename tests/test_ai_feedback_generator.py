"""AIFeedbackGenerator 解析口径测试。

大模型一律用 FakeClient 顶替，绝不联网；替身会捕获模块实际发出去的
messages 供断言。
"""
import logging
from types import SimpleNamespace

import pytest

from backend.ai_feedback_generator import (
    AIFeedbackGenerator,
    FeedbackGenerationError,
)


class FakeCompletions:
    """顶替 client.chat.completions，捕获调用参数并返回造好的响应。"""

    def __init__(self, content, finish_reason="stop"):
        self._content = content
        self._finish_reason = finish_reason
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content)
        choice = SimpleNamespace(message=message, finish_reason=self._finish_reason)
        return SimpleNamespace(choices=[choice])


class FakeClient:
    def __init__(self, content, finish_reason="stop"):
        self.completions = FakeCompletions(content, finish_reason)
        self.chat = SimpleNamespace(completions=self.completions)


def make_generator(content, finish_reason="stop"):
    gen = AIFeedbackGenerator(api_key="test-key")
    fake = FakeClient(content, finish_reason)
    gen.client = fake
    return gen, fake


FEEDBACK_KWARGS = dict(
    project_title="海底世界",
    project_description="一个关于海洋的 Scratch 动画",
    student_names="小明,小红",
    project_type="scratch",
    transcript="小明：这是我们做的海底世界。",
    technical_feedback="导师手填点评",
    mentor_comments=["很有创意", "代码整洁"],
)

EMAIL_KWARGS = dict(
    student_names="小明,小红",
    project_title="海底世界",
    warm_feedback="孩子们表现很棒。",
    ai_summary="一个海底世界动画。",
)


# ---------------------------------------------------------------- 三段正常输出

def test_three_sections_normal():
    content = "技术亮点：循环用得巧\n---\n这是项目摘要。\n---\n孩子们真棒，继续加油。"
    gen, _ = make_generator(content)
    result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert set(result) == {
        "technical_feedback", "improvement_suggestions", "ai_summary", "warm_feedback"
    }
    assert all(isinstance(v, str) for v in result.values())
    assert result["technical_feedback"] == "技术亮点：循环用得巧"
    assert result["ai_summary"] == "这是项目摘要。"
    assert result["warm_feedback"] == "孩子们真棒，继续加油。"


def test_separator_variants():
    """分隔符多写横线、前后有空白和空行，都要识别。"""
    content = (
        "第一段技术点评\n"
        "\n"
        "  ----  \n"
        "\n"
        "第二段摘要\n"
        "-----\n"
        "第三段温暖评语"
    )
    gen, _ = make_generator(content)
    result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert result["technical_feedback"] == "第一段技术点评"
    assert result["ai_summary"] == "第二段摘要"
    assert result["warm_feedback"] == "第三段温暖评语"


def test_horizontal_rule_inside_third_section_preserved():
    """第三段自带的 Markdown 水平线不是分段点，后半段不许丢。"""
    content = "技术点评\n---\n摘要\n---\n孩子们前半段评语。\n\n---\n\n后半段评语，一句都不能少。"
    gen, _ = make_generator(content)
    result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert "前半段评语" in result["warm_feedback"]
    assert "后半段评语，一句都不能少。" in result["warm_feedback"]


# ---------------------------------------------------------------- 成稿不许加工

def test_summary_not_truncated_not_filtered():
    """13 行、带 ## 和 ** 行的摘要必须完整保留，不删行不截断。"""
    summary_lines = ["## 项目摘要"] + [f"第{i}行摘要内容。" for i in range(1, 12)] + ["**结尾**"]
    assert len(summary_lines) == 13
    content = "技术点评\n---\n" + "\n".join(summary_lines) + "\n---\n温暖评语"
    gen, _ = make_generator(content)
    result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert result["ai_summary"] == "\n".join(summary_lines)


def test_technical_feedback_kept_verbatim():
    tech_lines = [f"亮点{i}：内容。" for i in range(1, 13)]
    content = "\n".join(tech_lines) + "\n---\n摘要\n---\n温暖评语"
    gen, _ = make_generator(content)
    result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert result["technical_feedback"] == "\n".join(tech_lines)


# ---------------------------------------------------------------- 改进建议切点

def test_improvement_cut_at_heading_start():
    """切点落在「改进」开头，不能从标题中间下刀把「改进」两个字丢掉。"""
    first = "技术亮点：\n1. 循环结构用得好\n\n改进建议：\n1. 给变量加注释"
    content = first + "\n---\n摘要\n---\n温暖评语"
    gen, _ = make_generator(content)
    result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert result["improvement_suggestions"].startswith("改进建议")
    assert "给变量加注释" in result["improvement_suggestions"]
    assert "循环结构用得好" not in result["improvement_suggestions"]


def test_improvement_heading_with_list_marker():
    first = "技术亮点若干。\n2. 建议：给变量加注释"
    content = first + "\n---\n摘要\n---\n温暖评语"
    gen, _ = make_generator(content)
    result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert result["improvement_suggestions"].startswith("建议：")


def test_improvement_no_heading_returns_whole():
    first = "通篇都是技术亮点，没有需要补充的部分"
    content = first + "\n---\n摘要\n---\n温暖评语"
    gen, _ = make_generator(content)
    result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert result["improvement_suggestions"] == first


# ---------------------------------------------------------------- 段数不足降级

def test_two_sections_degrades_and_never_duplicates(caplog):
    """只写两段：warm_feedback 置空并告警；改进建议和摘要不许是同一段。

    负例锚点：若实现退化成让 improvement_suggestions 和 ai_summary
    同取 parts[1]，本测试必须变红。
    """
    content = "技术亮点：循环用得好\n\n改进建议：加注释\n---\n这是第二段项目摘要，和第一段完全不同。"
    gen, _ = make_generator(content)
    with caplog.at_level(logging.WARNING):
        result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert result["warm_feedback"] == ""
    assert result["ai_summary"] == "这是第二段项目摘要，和第一段完全不同。"
    assert result["improvement_suggestions"] != result["ai_summary"]
    assert result["improvement_suggestions"].startswith("改进建议")
    assert all(isinstance(v, str) for v in result.values())
    assert any("没写第三段" in r.message for r in caplog.records)


def test_no_separator_at_all(caplog):
    content = "只有一整段话，没有任何分隔符。"
    gen, _ = make_generator(content)
    with caplog.at_level(logging.WARNING):
        result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert result["technical_feedback"] == content
    assert result["ai_summary"] == ""
    assert result["warm_feedback"] == ""
    assert all(isinstance(v, str) for v in result.values())


def test_third_section_written_but_empty_distinguished(caplog):
    """「没写第三段」和「写了但空」要在日志里可分辨。"""
    content = "技术点评\n---\n摘要\n---\n"
    gen, _ = make_generator(content)
    with caplog.at_level(logging.WARNING):
        result = gen.generate_feedback(**FEEDBACK_KWARGS)
    assert result["warm_feedback"] == ""
    messages = [r.message for r in caplog.records]
    assert any("内容为空" in m for m in messages)
    assert not any("没写第三段" in m for m in messages)


# ---------------------------------------------------------------- 输出不可用

def test_content_none_raises_readable_error():
    gen, _ = make_generator(None)
    with pytest.raises(FeedbackGenerationError, match="None"):
        gen.generate_feedback(**FEEDBACK_KWARGS)


def test_empty_string_raises_readable_error():
    gen, _ = make_generator("   ")
    with pytest.raises(FeedbackGenerationError, match="空字符串"):
        gen.generate_feedback(**FEEDBACK_KWARGS)


def test_finish_reason_length_rejected():
    """被 token 上限截断的半成品不许悄悄落库。"""
    gen, _ = make_generator("孩子在这次展示中", finish_reason="length")
    with pytest.raises(FeedbackGenerationError, match="截断"):
        gen.generate_feedback(**FEEDBACK_KWARGS)


# ---------------------------------------------------------------- mentor_comments 脏数据

def test_mentor_comments_with_none_and_non_string():
    gen, fake = make_generator("技术\n---\n摘要\n---\n温暖")
    kwargs = dict(FEEDBACK_KWARGS, mentor_comments=["正常评语", None, 42, "另一条"])
    result = gen.generate_feedback(**kwargs)
    assert all(isinstance(v, str) for v in result.values())
    user_msg = fake.completions.calls[0]["messages"][1]["content"]
    assert "正常评语" in user_msg
    assert "另一条" in user_msg
    assert "None" not in user_msg
    assert "42" not in user_msg


def test_mentor_comments_none_whole():
    gen, _ = make_generator("技术\n---\n摘要\n---\n温暖")
    kwargs = dict(FEEDBACK_KWARGS, mentor_comments=None)
    result = gen.generate_feedback(**kwargs)
    assert all(isinstance(v, str) for v in result.values())


# ---------------------------------------------------------------- 替身捕获 messages

def test_messages_actually_sent():
    gen, fake = make_generator("技术\n---\n摘要\n---\n温暖")
    gen.generate_feedback(**FEEDBACK_KWARGS)
    assert len(fake.completions.calls) == 1
    call = fake.completions.calls[0]
    assert call["messages"][0]["role"] == "system"
    user_msg = call["messages"][1]["content"]
    assert "海底世界" in user_msg
    assert "小明：这是我们做的海底世界。" in user_msg
    assert "导师手填点评" in user_msg
    assert "很有创意" in user_msg


# ---------------------------------------------------------------- 邮件

def test_email_normal():
    content = "主题：孩子们的海底世界\n---\n<p>尊敬的家长您好</p>"
    gen, _ = make_generator(content)
    result = gen.generate_email_content(**EMAIL_KWARGS)
    assert result["subject"] == "孩子们的海底世界"
    assert "\n" not in result["subject"]
    assert not result["subject"].startswith("主题")
    assert result["body_html"] == "<p>尊敬的家长您好</p>"


def test_email_subject_label_and_embedded_newline():
    content = "主题：\n《海底世界》展示反馈\n---\n<p>正文</p>"
    gen, _ = make_generator(content)
    result = gen.generate_email_content(**EMAIL_KWARGS)
    assert result["subject"] == "《海底世界》展示反馈"
    assert "\n" not in result["subject"]


def test_email_body_horizontal_rule_not_eaten():
    """正文里自己的分隔线不许把后半截吃掉。"""
    body = "<p>前半截</p>\n---\n<p>后半截，一句都不能少</p>"
    content = "邮件主题\n---\n" + body
    gen, _ = make_generator(content)
    result = gen.generate_email_content(**EMAIL_KWARGS)
    assert result["subject"] == "邮件主题"
    assert result["body_html"] == body


def test_email_separator_variants():
    content = "邮件主题\n\n  ----  \n\n<p>正文</p>"
    gen, _ = make_generator(content)
    result = gen.generate_email_content(**EMAIL_KWARGS)
    assert result["subject"] == "邮件主题"
    assert result["body_html"] == "<p>正文</p>"


def test_email_no_separator_subject_differs_from_body(caplog):
    content = "<p>模型没写分隔符，整段都是正文</p>"
    gen, _ = make_generator(content)
    with caplog.at_level(logging.WARNING):
        result = gen.generate_email_content(**EMAIL_KWARGS)
    assert result["body_html"] == content
    assert result["subject"] != result["body_html"]
    assert result["subject"] == "《海底世界》项目展示反馈"
    assert "\n" not in result["subject"]


def test_email_content_none_raises():
    gen, _ = make_generator(None)
    with pytest.raises(FeedbackGenerationError):
        gen.generate_email_content(**EMAIL_KWARGS)


def test_email_finish_reason_length_raises():
    gen, _ = make_generator("主题\n---\n<p>写到一半", finish_reason="length")
    with pytest.raises(FeedbackGenerationError, match="截断"):
        gen.generate_email_content(**EMAIL_KWARGS)


def test_email_messages_captured():
    gen, fake = make_generator("主题\n---\n<p>正文</p>")
    gen.generate_email_content(**EMAIL_KWARGS)
    call = fake.completions.calls[0]
    user_msg = call["messages"][1]["content"]
    assert "小明,小红" in user_msg
    assert "海底世界" in user_msg
