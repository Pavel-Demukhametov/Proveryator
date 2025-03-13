
# from g4f.client import Client
# from g4f.Provider import RetryProvider, ChatGLM, DDG, Free2GPT, GizAI, Liaobots, OIVSCode, PollinationsAI
# import json

# class ChatGPTQAGenerator:
#     def __init__(self):
#         # Определяем список провайдеров и инициализируем клиента
#         providers = [DDG, PollinationsAI, Liaobots, OIVSCode, Free2GPT, GizAI, ChatGLM]
#         self.client = Client(provider=RetryProvider(providers, shuffle=False))
#         self.model = "gpt-4o-mini"  # можно изменить модель по необходимости
        
#         # Промт для генерации вопроса с учётом правильного ответа
#         self.prompt_template = """Проанализируй приведенный ниже текст на русском языке и сгенерируй соответствующий вопрос на основе его содержания, учитывая предоставленный правильный ответ.
# Не включай никаких комментариев, ответь только JSON-объектом следующего вида:
# {{
#   "question": "<текст вопроса на русском языке>",
#   "answers": [<верный и 3 неверных ответа на вопрос>],
#   "correct_answer": <индекс верного ответа в массиве answers>
# }}

# Текст для анализа: "{}"
# Правильный ответ: "{}"
# """

#     def generate_qa(self, text: str, correct_answer: str) -> dict:
#         # Формируем промт с подстановкой текста и правильного ответа
#         prompt = self.prompt_template.format(text, correct_answer)
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": prompt}],
#             )
#             response_content = response.choices[0].message.content
#             # Парсим JSON-ответ
#             result = json.loads(response_content)
#             return result
#         except Exception as e:
#             print(f"Ошибка при генерации вопроса: {e}")
#             return {}


from g4f.client import Client
from g4f.Provider import RetryProvider, ChatGLM, DDG, Free2GPT, GizAI, Liaobots, OIVSCode, PollinationsAI
import json


class ChatGPTQAGenerator:
    def __init__(self):
        providers = [DDG, PollinationsAI, Liaobots, OIVSCode, Free2GPT, GizAI, ChatGLM]
        self.client = Client(provider=RetryProvider(providers, shuffle=False))
        self.model = "gpt-4o-mini"
        
        self.mc_prompt_template = """Проанализируй приведенный ниже текст на русском языке и сгенерируй вопрос с четырьмя вариантами ответов на основе его содержания, связанный с темой "{}". Один из вариантов должен быть правильным.
Не включай никаких комментариев, ответь только JSON-объектом следующего вида:
{{
  "Вопрос": "<текст вопроса на русском языке>",
  "Варианты": [<вариант1>, <вариант2>, <вариант3>, <вариант4>],
  "Правильный_ответ": <индекс верного ответа в массиве Варианты, начиная с 0>
}}

Текст для анализа: "{}"
"""
        
        self.open_prompt_template = """Проанализируй приведенный ниже текст на русском языке и сгенерируй открытый вопрос на основе его содержания, связанный с темой "{}". Также предоставь правильный ответ на этот вопрос.
Не включай никаких комментариев, ответь только JSON-объектом следующего вида:
{{
  "Вопрос": "<текст вопроса на русском языке>",
  "Правильный_ответ": "<правильный ответ>"
}}

Текст для анализа: "{}"
"""

    def generate_qa(self, text: str, keyword: str, is_open: bool) -> dict:
        """
        Генерирует вопрос на основе текста и ключевого слова.
        
        :param text: Текст для анализа
        :param keyword: Ключевое слово или тема
        :param is_open: Флаг, указывающий на тип вопроса (True для открытых, False для MC)
        :return: Словарь с вопросом и ответами
        """
        if is_open:
            prompt = self.open_prompt_template.format(keyword, text)
        else:
            prompt = self.mc_prompt_template.format(keyword, text)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )
            response_content = response.choices[0].message.content
            result = json.loads(response_content)
            if is_open and "Варианты" in result:
                del result["Варианты"]
            
            return result
        except Exception as e:
            print(f"Ошибка при генерации вопроса: {e}")
            return {}