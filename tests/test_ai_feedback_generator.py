"""AIFeedbackGenerator 的解析口径测试。

大模型一律用 FakeCompletions 顶替，不联网；替身会捕获模块实际发出的
messages 供断言。构造响应用的是内存中的合成数据。
"""
import logging
from types import SimpleNamespace

import pytest

from backend.ai_feedback_generator import (
    AIFeedbackGenerator,
    FeedbackGenerationError,
)


class FakeCompletions:
    """顶替 client.chat.completions.create 的替身。

    每次调用把关键字参数（含 messages）存进 self.calls 供断言，
    按队列顺序返回预设响应；队列空了还调用就直接失败。
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        assert self._responses, "替身没有剩余响应，模块多调了一次模型"
        return self._responses.pop(0)


def make_response(content, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


def make_generator(*responses):
    gen = AIFeedbackGenerator(api_key="test-key")
    fake = FakeCompletions(responses)
    gen.client.chat.completions.create = fake.create
    return gen, fake


def feedback_kwargs(**overrides):
    kwargs = dict(
        project_title="海底世界",
        project_description="用 Scratch 做的海底探险游戏",
        student_names="小明,小红",
        project_type="scratch",
        transcript="小明：大家好，这是我们的作品。",
        technical_feedback="",
        mentor_comments=None,
    )
    kwargs.update(overrides)
    return kwargs


def email_kwargs(**overrides):
    kwargs = dict(
        student_names="小明,小红",
        project_title="海底世界",
        warm_feedback="孩子们表现很棒。",
        ai_summary="一个海底探险游戏。",
    )
    kwargs.update(overrides)
    return kwargs


SEG_TECH = "技术亮点：\n- 角色移动流畅\n- 用了广播机制\n改进建议：\n- 给变量加注释"
SEG_SUMMARY = "本项目是一个海底探险游戏，玩家控制小鱼躲避障碍。"
SEG_WARM = "小明和小红在这次展示中表现非常出色，值得肯定。"


def three_segment_content():
    return f"{SEG_TECH}\n---\n{SEG_SUMMARY}\n---\n{SEG_WARM}"


# ---------- 正常三段 ----------

def test_three_segments_map_to_fields_in_order():
    gen, _ = make_generator(make_response(three_segment_content()))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result == {
        "technical_feedback": SEG_TECH,
        "improvement_suggestions": "改进建议：\n- 给变量加注释",
        "ai_summary": SEG_SUMMARY,
        "warm_feedback": SEG_WARM,
    }


def test_result_always_has_exactly_four_string_keys():
    gen, _ = make_generator(make_response(three_segment_content()))
    result = gen.generate_feedback(**feedback_kwargs())
    assert set(result) == {
        "technical_feedback",
        "improvement_suggestions",
        "ai_summary",
        "warm_feedback",
    }
    assert all(isinstance(value, str) for value in result.values())


def test_messages_actually_sent_to_model():
    gen, fake = make_generator(make_response(three_segment_content()))
    gen.generate_feedback(**feedback_kwargs(mentor_comments=["创意不错"]))
    assert len(fake.calls) == 1
    messages = fake.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    user_prompt = messages[1]["content"]
    assert "海底世界" in user_prompt
    assert "小明,小红" in user_prompt
    assert "创意不错" in user_prompt


# ---------- 摘要完整保留 ----------

def test_summary_kept_in_full_no_line_filter_no_truncation():
    lines = ["## 项目摘要"] + [f"第{i}行内容。" for i in range(1, 13)]
    lines.append("**重点：合作完成。**")
    summary = "\n".join(lines)  # 14 行，含 ## 和 ** 开头的行
    content = f"{SEG_TECH}\n---\n{summary}\n---\n{SEG_WARM}"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["ai_summary"] == summary


def test_technical_feedback_kept_in_full():
    content = f"{SEG_TECH}\n---\n{SEG_SUMMARY}\n---\n{SEG_WARM}"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["technical_feedback"] == SEG_TECH


# ---------- 改进建议切点 ----------

def test_improvement_cut_at_heading_start_not_middle():
    gen, _ = make_generator(make_response(three_segment_content()))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["improvement_suggestions"].startswith("改进建议")
    assert "给变量加注释" in result["improvement_suggestions"]


def test_improvement_does_not_include_highlights_before_heading():
    gen, _ = make_generator(make_response(three_segment_content()))
    result = gen.generate_feedback(**feedback_kwargs())
    assert "技术亮点" not in result["improvement_suggestions"]
    assert "角色移动流畅" not in result["improvement_suggestions"]


def test_improvement_empty_when_no_heading():
    tech = "技术亮点：\n- 角色移动流畅"
    content = f"{tech}\n---\n{SEG_SUMMARY}\n---\n{SEG_WARM}"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["improvement_suggestions"] == ""
    assert result["technical_feedback"] == tech


# ---------- 分隔符的脏形态 ----------

def test_four_and_five_dash_delimiters():
    content = f"{SEG_TECH}\n----\n{SEG_SUMMARY}\n-----\n{SEG_WARM}"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["ai_summary"] == SEG_SUMMARY
    assert result["warm_feedback"] == SEG_WARM
    assert not result["warm_feedback"].startswith("-")


def test_delimiter_with_surrounding_spaces_and_blank_lines():
    content = f"{SEG_TECH}\n\n  ---  \n\n\n{SEG_SUMMARY}\n\t---\t\n\n{SEG_WARM}"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["technical_feedback"] == SEG_TECH
    assert result["ai_summary"] == SEG_SUMMARY
    assert result["warm_feedback"] == SEG_WARM


def test_horizontal_rule_inside_third_segment_is_kept():
    warm = f"{SEG_WARM}\n\n---\n\n后半段评语：继续加油，期待下次展示。"
    content = f"{SEG_TECH}\n---\n{SEG_SUMMARY}\n---\n{warm}"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["warm_feedback"] == warm
    assert "后半段评语" in result["warm_feedback"]


def test_inline_dashes_are_not_delimiters():
    tech = "技术亮点：变量命名清晰 --- 值得表扬"
    content = f"{tech}\n---\n{SEG_SUMMARY}\n---\n{SEG_WARM}"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["technical_feedback"] == tech


# ---------- 段数不足的降级 ----------

def test_two_segments_degrade_warm_empty_and_no_duplication(caplog):
    content = f"{SEG_TECH}\n---\n{SEG_SUMMARY}"
    gen, _ = make_generator(make_response(content))
    with caplog.at_level(logging.WARNING):
        result = gen.generate_feedback(**feedback_kwargs())
    assert result["technical_feedback"] == SEG_TECH
    assert result["ai_summary"] == SEG_SUMMARY
    assert result["warm_feedback"] == ""
    assert result["improvement_suggestions"] != result["ai_summary"]
    assert "缺少第三段" in caplog.text


def test_improvement_and_summary_are_never_the_same_segment():
    """负例钉桩：若实现退化成改进建议和摘要取同一段，本测试必须变红。"""
    content = f"{SEG_TECH}\n---\n{SEG_SUMMARY}"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["improvement_suggestions"] != ""
    assert result["ai_summary"] != ""
    assert result["improvement_suggestions"] != result["ai_summary"]


def test_third_segment_written_but_empty_logs_differently(caplog):
    content = f"{SEG_TECH}\n---\n{SEG_SUMMARY}\n---\n"
    gen, _ = make_generator(make_response(content))
    with caplog.at_level(logging.WARNING):
        result = gen.generate_feedback(**feedback_kwargs())
    assert result["warm_feedback"] == ""
    assert "第三段但内容为空" in caplog.text
    assert "缺少第三段" not in caplog.text


def test_single_segment_only():
    gen, _ = make_generator(make_response(SEG_TECH))
    result = gen.generate_feedback(**feedback_kwargs())
    assert result["technical_feedback"] == SEG_TECH
    assert result["ai_summary"] == ""
    assert result["warm_feedback"] == ""
    assert result["improvement_suggestions"] == "改进建议：\n- 给变量加注释"


def test_empty_string_output_degrades_to_empty_fields(caplog):
    gen, _ = make_generator(make_response(""))
    with caplog.at_level(logging.WARNING):
        result = gen.generate_feedback(**feedback_kwargs())
    assert result == {
        "technical_feedback": "",
        "improvement_suggestions": "",
        "ai_summary": "",
        "warm_feedback": "",
    }
    assert caplog.text


# ---------- content=None 与截断 ----------

def test_content_none_raises_readable_error_not_attribute_error():
    gen, _ = make_generator(make_response(None))
    with pytest.raises(FeedbackGenerationError) as exc_info:
        gen.generate_feedback(**feedback_kwargs())
    assert "content" in str(exc_info.value)


def test_finish_reason_length_raises_and_stores_nothing():
    truncated = f"{SEG_TECH}\n---\n{SEG_SUMMARY}\n---\n孩子在这次展示中"
    gen, _ = make_generator(make_response(truncated, finish_reason="length"))
    with pytest.raises(FeedbackGenerationError) as exc_info:
        gen.generate_feedback(**feedback_kwargs())
    assert "截断" in str(exc_info.value)


def test_email_content_none_raises():
    gen, _ = make_generator(make_response(None))
    with pytest.raises(FeedbackGenerationError):
        gen.generate_email_content(**email_kwargs())


def test_email_finish_reason_length_raises():
    gen, _ = make_generator(make_response("主题\n---\n<p>半截", finish_reason="length"))
    with pytest.raises(FeedbackGenerationError):
        gen.generate_email_content(**email_kwargs())


# ---------- mentor_comments 脏数据 ----------

def test_mentor_comments_with_none_and_non_string_items():
    gen, fake = make_generator(make_response(three_segment_content()))
    result = gen.generate_feedback(
        **feedback_kwargs(mentor_comments=["创意不错", None, 42, "继续加油"])
    )
    assert set(result) == {
        "technical_feedback",
        "improvement_suggestions",
        "ai_summary",
        "warm_feedback",
    }
    user_prompt = fake.calls[0]["messages"][1]["content"]
    assert "创意不错" in user_prompt
    assert "继续加油" in user_prompt
    assert "None" not in user_prompt
    assert "42" not in user_prompt


def test_mentor_comments_none_and_empty_list():
    for comments in (None, []):
        gen, _ = make_generator(make_response(three_segment_content()))
        result = gen.generate_feedback(**feedback_kwargs(mentor_comments=comments))
        assert result["ai_summary"] == SEG_SUMMARY


# ---------- 邮件 ----------

def test_email_subject_stripped_of_label_and_newlines():
    content = "主题：孩子的海底世界之旅\n\n---\n<p>正文</p>"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_email_content(**email_kwargs())
    assert result["subject"] == "孩子的海底世界之旅"
    assert "\n" not in result["subject"]
    assert "主题" not in result["subject"]
    assert result["body_html"] == "<p>正文</p>"


def test_email_subject_english_label_and_extra_lines():
    content = "Subject: Great job!\n另一行废话\n---\n<p>正文</p>"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_email_content(**email_kwargs())
    assert result["subject"] == "Great job!"


def test_email_body_keeps_content_after_inner_horizontal_rule():
    body = "<p>前半段</p>\n---\n<p>后半段不能丢</p>"
    content = f"主题：反馈\n---\n{body}"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_email_content(**email_kwargs())
    assert result["body_html"] == body
    assert "后半段不能丢" in result["body_html"]


def test_email_multi_dash_delimiter_with_spaces():
    content = "主题：反馈\n\n  ----  \n\n<p>正文</p>"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_email_content(**email_kwargs())
    assert result["subject"] == "反馈"
    assert result["body_html"] == "<p>正文</p>"


def test_email_without_delimiter_subject_differs_from_body():
    body = "<p>只有正文，模型没写分隔符</p>"
    gen, _ = make_generator(make_response(body))
    result = gen.generate_email_content(**email_kwargs())
    assert result["body_html"] == body
    assert result["subject"] != result["body_html"]
    assert result["subject"]
    assert "\n" not in result["subject"]


def test_email_empty_subject_line_falls_back():
    content = "\n---\n<p>正文</p>"
    gen, _ = make_generator(make_response(content))
    result = gen.generate_email_content(**email_kwargs())
    assert result["subject"]
    assert result["subject"] != result["body_html"]
    assert result["body_html"] == "<p>正文</p>"


def test_email_messages_actually_sent():
    gen, fake = make_generator(make_response("主题：反馈\n---\n<p>正文</p>"))
    gen.generate_email_content(**email_kwargs())
    messages = fake.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "海底世界" in messages[1]["content"]
