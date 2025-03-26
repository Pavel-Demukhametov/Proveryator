import json
import logging
import random
import re
import requests
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from typing import List, Optional, Dict
import asyncio
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_access_token():
    oauth_url = 'https://ngw.devices.sberbank.ru:9443/api/v2/oauth'
    uid = '639a832c-d12a-4c2f-9207-f654e0d86fa6'
    client_base64 = 'NjM5YTgzMmMtZDEyYS00YzJmLTkyMDctZjY1NGUwZDg2ZmE2OmE4MTY2MTg1LWQ1OWEtNGVkOC05ZmEzLWRmYjRlOTdjNDAxOA=='
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': uid,
        'Authorization': 'Basic ' + client_base64
    }
    payload = 'scope=GIGACHAT_API_PERS'
    # Путь к сертификату может потребовать корректировки
    response = requests.post(oauth_url, headers=headers, data=payload, verify="C:\\Users\\purit\\Downloads\\russian_trusted_root_ca.cer")
    return response.json().get('access_token')

def parse_distractors(response_content: str) -> List[str]:
    distractors = []
    response_content = re.sub(r'Неправильные варианты ответа[:\-]?\s*', '', response_content, flags=re.IGNORECASE)
    response_content = re.sub(r'3 неправильных варианта ответа[:\-]?\s*', '', response_content, flags=re.IGNORECASE)
    numbered = re.findall(r'\d+[.)]\s*([^;]+)', response_content)
    if numbered and len(numbered) >= 3:
        distractors = [d.strip() for d in numbered[:3]]
        logger.info("Дистракторы успешно распознаны с нумерацией.")
        return distractors
    split_distractors = [d.strip() for d in response_content.split(';') if d.strip()]
    if len(split_distractors) >= 3:
        distractors = split_distractors[:3]
        logger.info("Дистракторы успешно распознаны без нумерации.")
        return distractors
    additional = re.split(r'\d+[.)]\s*', response_content)
    additional = [d.strip() for d in additional if d.strip()]
    if len(additional) >= 3:
        distractors = additional[:3]
        logger.info("Дистракторы успешно распознаны при разделении по нумерации без точек с запятой.")
        return distractors
    sentences = re.split(r'[.!?]\s*', response_content)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 3:
        distractors = sentences[:3]
        logger.warning("Дистракторы распознаны по предложениям, так как другие методы не сработали.")
        return distractors
    logger.error("Не удалось распознать дистракторы из ответа.")
    return distractors

def generate_distractors(access_token: str, text: str, correct_answer: str) -> List[str]:
    prompt = (
        'На основе предоставленного текста и правильного ответа, составь 3 неправильных варианта ответа (дистракторы), '
        'не надо добавлять ещё какой-то текст или нумеровать неправильные варианты. Твой ответ должен содержать только '
        '3 неправильных варианта ответа, разделённых точкой с запятой. Текст: {text}. '
        'Правильный ответ: {correct_answer}'
    )
    formatted_prompt = prompt.format(text=text, correct_answer=correct_answer)
    payload = json.dumps({
        "model": "GigaChat",
        "messages": [{"role": "user", "content": formatted_prompt}],
        "stream": False,
        "repetition_penalty": 1
    })
    completions_url = 'https://gigachat.devices.sberbank.ru/api/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + access_token
    }
    try:
        response = requests.post(completions_url, headers=headers, data=payload, verify="C:\\Users\\purit\\Downloads\\russian_trusted_root_ca.cer")
        response.raise_for_status()
        response_json = response.json()
        logger.info("Ответ от GigaChat получен успешно.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка при обращении к GigaChat API: {e}")
        return []
    if "choices" not in response_json or not response_json["choices"]:
        logger.error("Ответ от GigaChat не содержит необходимых данных.")
        return []
    response_content = response_json["choices"][0]["message"]["content"]
    distractors = parse_distractors(response_content)
    if len(distractors) < 3:
        additional = [d.strip() for d in response_content.split(';') if d.strip()]
        for d in additional:
            if d not in distractors:
                distractors.append(d)
            if len(distractors) == 3:
                break
    if len(distractors) < 3:
        return []
    logger.info(f"Дистракторы успешно сгенерированы: {distractors}")
    return distractors

class QAGenerator:
    """
    Интерфейс генерации вопроса должен быть совместим с первым примером:
    
    - Конструктор без параметров (с возможностью задавать пути к моделям)
    - Метод generate_qa(text: str, keyword: str, is_open: bool) -> dict,
      который возвращает JSON-объект с ключами "Вопрос", "Правильный_ответ" и, при необходимости, "Варианты".
    """
    def __init__(self,
                 question_model_path: str = 'question_generation_model',
                 answer_model_path: str = 'final_model',
                 tokenizer_model: str = "ai-forever/ruT5-base"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.question_model = T5ForConditionalGeneration.from_pretrained(question_model_path).to(self.device)
        self.answer_model = T5ForConditionalGeneration.from_pretrained(answer_model_path).to(self.device)
        self.access_token = get_access_token()
        self.api_available = False
        self.check_api_availability()

        self.mc_prompt_template = (
            'Проанализируй приведенный ниже текст на русском языке и сгенерируй вопрос с четырьмя вариантами ответов на основе его содержания. '
            'Один из вариантов должен быть правильным. Ответ должен быть простым, состоять максимум из нескольких слов. '
            'Остальные варианты ответа должны быть неправильными и не подходить.\n'
            'Не включай никаких комментариев, ответь только JSON-объектом следующего вида:\n'
            '{{\n  "Вопрос": "<текст вопроса на русском языке>",\n  "Варианты": [<вариант1>, <вариант2>, <вариант3>, <вариант4>],\n  "Правильный_ответ": <индекс верного ответа в массиве Варианты, начиная с 0>\n}}\n'
            'Текст для анализа: "{}"\n'
        )
        self.open_prompt_template = (
            'Проанализируй приведенный ниже текст на русском языке и сгенерируй открытый вопрос на основе его содержания. '
            'Также предоставь правильный ответ на этот вопрос. Ответ должен быть простым, состоять максимум из нескольких слов. '
            'Не включай никаких комментариев, ответь только JSON-объектом следующего вида:\n'
            '{{\n  "Вопрос": "<текст вопроса на русском языке>",\n  "Правильный_ответ": "<правильный ответ>"\n}}\n'
            'Текст для анализа: "{}"\n'
        )
    def check_api_availability(self):
        response = requests.get('http://localhost:8673/aaa')
        if response.status_code == 200:
            self.api_available = True
    def generate_api(self, text: str) -> Dict[str, str]:
        try:
            payload = json.dumps({"text": text})
            headers = {'Content-Type': 'application/json'}
            response = requests.post('http://localhost:8673/asxz23', headers=headers, data=payload)
            response.raise_for_status()

            response_data = response.json()
            print(response_data)
            question = response_data.get("Вопрос")
            correct_answer = response_data.get("Правильный_ответ")
            distractors = response_data.get("Варианты", [])

            if not question or correct_answer is None or correct_answer == "" or len(distractors) < 3:
                logger.error("Ответ от внешнего API некорректен.")
                return {}

            return {
                "Вопрос": question,
                "Правильный_ответ": correct_answer,
                "Варианты": distractors
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при обращении к внешнему API: {e}")
            return {}
    def gen_questions(self, text: str, num_questions: int, question_type: str) -> List[str]:
        """
        Генерирует список вопросов на основе исходного текста.
        Параметр question_type здесь не используется напрямую, но оставлен для совместимости.
        """
        input_ids = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).input_ids.to(self.device)
        question_ids = self.question_model.generate(
            input_ids,
            max_length=64,
            num_beams=12,
            num_return_sequences=num_questions,
            temperature=1.2,
            top_p=0.9,
            early_stopping=True,
        )
        questions = [self.tokenizer.decode(q, skip_special_tokens=True) for q in question_ids]
        return questions

    def gen_answer(self, text: str, question: str) -> str:
        """
        Генерирует правильный ответ на вопрос с учётом исходного текста.
        """
        input_text = f"{question} {text}"
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).input_ids.to(self.device)
        answer_ids = self.answer_model.generate(
            input_ids, max_length=128, num_beams=5, early_stopping=True
        )
        answer = self.tokenizer.decode(answer_ids[0], skip_special_tokens=True)
        return answer.strip()
    
    def gen_distractors(self, text: str, answer: str) -> List[str]:
        """
        Генерирует список неправильных вариантов ответа (дистракторов) с использованием GigaChat API.
        """
        return generate_distractors(self.access_token, text, answer)
            
    def generate_qa(self, text: str, keyword: str, is_open: bool) -> dict:
        """
        Генерирует вопрос на основе текста и ключевого слова.
        В зависимости от доступности внешнего API, будет использоваться либо внешний сервис, либо код класса.
        Если API возвращает некорректный ответ, он игнорируется и возвращается пустой словарь.
        """
        if self.api_available:
            result = self.generate_api(text)
            print(result)
            if not result:
                logger.info("Некорректный ответ от API, игнорируем его.")
                return {}
            else:
                logger.info("Вполне себе коректно")
            return result
        else:
            logger.warning("Внешний API не доступен. Используем стандартную логику.")
            question_type = 'open' if is_open else 'mc'
            questions = self.gen_questions(text, 1, question_type)
            if not questions:
                logger.error("Не удалось сгенерировать вопрос.")
                return {}
            question = questions[0]
            correct_answer = self.gen_answer(text, question)
            result = {
                "Вопрос": question,
            }
            if is_open:
                result["Правильный_ответ"] = correct_answer
            else:
                distractors = self.gen_distractors(text, correct_answer)
                if not distractors or len(distractors) < 1:
                    logger.error("Не удалось сгенерировать достаточное количество дистракторов.")
                    return {}
                options = distractors.copy()
                correct_index = random.randint(0, len(options))
                options.insert(correct_index, correct_answer)
                result["Варианты"] = options
                result["Правильный_ответ"] = correct_index
            return result
