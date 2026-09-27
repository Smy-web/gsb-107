from openai import OpenAI
from typing import Dict, List

class AIFeedbackGenerator:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

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
        mentor_comments_text = "\n".join(mentor_comments) if mentor_comments else ""
        
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

        content = response.choices[0].message.content.strip()
        parts = content.split('---')
        
        result = {
            'technical_feedback': parts[0].strip() if len(parts) > 0 else "",
            'improvement_suggestions': parts[1].strip() if len(parts) > 1 else "",
            'ai_summary': parts[1].strip() if len(parts) > 1 else "",
            'warm_feedback': parts[2].strip() if len(parts) > 2 else ""
        }

        if len(parts) >= 3:
            result['ai_summary'] = self._extract_summary(parts[1].strip())
            result['improvement_suggestions'] = self._extract_improvements(parts[0].strip())
            result['technical_feedback'] = parts[0].strip()

        return result

    def _extract_summary(self, text: str) -> str:
        lines = text.split('\n')
        return '\n'.join([l for l in lines if not l.strip().startswith('##') and not l.strip().startswith('**')][:10])

    def _extract_improvements(self, text: str) -> str:
        if '改进' in text or '建议' in text:
            idx = max(text.find('改进'), text.find('建议'))
            return text[idx:]
        return text

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

        content = response.choices[0].message.content.strip()
        parts = content.split('---')
        
        return {
            'subject': parts[0].strip(),
            'body_html': parts[1].strip() if len(parts) > 1 else parts[0].strip()
        }
