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
    except requests.exceptions.RequestException as e:
        logger.error(f"Error GigaChat API: {e}")
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
    return distractors


class QAGenerator:
    def __init__(self, endpoint_url: str = 'http://localhost:8673/asxz23', timeout: int = 10):
        self.endpoint_url = endpoint_url
        self.timeout = timeout
        self.headers = {'Content-Type': 'application/json'}
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

    def generate_questions(self, text: str, is_open: bool) -> Dict[str, str]:
        try:
            prompt = self.open_prompt_template.format(text) if is_open else self.mc_prompt_template.format(text)
            payload = json.dumps({"text": text, "prompt": prompt})
            
            response = requests.post(self.endpoint_url, headers=self.headers, data=payload, timeout=self.timeout)
            response.raise_for_status()

            response_data = response.json()
            logger.info(f"Response received: {response_data}")
            
            question = response_data.get("Вопрос")
            correct_answer = response_data.get("Правильный_ответ")
            distractors = response_data.get("Варианты", []) if not is_open else []

            if not question or correct_answer is None or correct_answer == "":
                logger.warning("Response missing question or correct answer, skipping.")
                return {}
            if not is_open and len(distractors) < 3:
                logger.warning("Response lacks enough distractors for MC question, skipping.")
                return {}

            return response_data
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out after {self.timeout} seconds, skipping.")
            return {}
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed: {e}, skipping.")
            return {}
        except ValueError as e:
            logger.warning(f"Invalid response format: {e}, skipping.")
            return {}

    def generate_qa(self, text: str, keyword: str, is_open: bool) -> dict:
        return self.generate_questions(text, is_open)