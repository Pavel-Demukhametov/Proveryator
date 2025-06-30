import json
import logging
import random
import re
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_access_token():
    """Получение токена доступа для аутентификации в GigaChat API"""
    oauth_url = os.getenv("GIGACHAT_OAUTH_URL")
    uid = os.getenv("GIGACHAT_UID")
    client_base64 = os.getenv("GIGACHAT_CLIENT_BASE64")
    cert_path = os.getenv("GIGACHAT_CERT_PATH")

    if not all([oauth_url, uid, client_base64, cert_path]):
        logger.error("Missing required environment variables for GigaChat authentication")
        raise ValueError("One or more environment variables (GIGACHAT_OAUTH_URL, GIGACHAT_UID, GIGACHAT_CLIENT_BASE64, GIGACHAT_CERT_PATH) are not set")

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json',
        'RqUID': uid,
        'Authorization': 'Basic ' + client_base64
    }
    payload = 'scope=GIGACHAT_API_PERS'
    try:
        response = requests.post(oauth_url, headers=headers, data=payload, verify=cert_path)
        response.raise_for_status()
        return response.json().get('access_token')
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get access token: {e}")
        raise

def parse_distractors(response_content: str) -> List[str]:
    """Парсинг неправильных вариантов ответа (дистракторов) из ответа GigaChat"""
    distractors = []
    response_content = re.sub(r'Неправильные варианты ответа[:\-]?\s*', '', response_content, flags=re.IGNORECASE)
    response_content = re.sub(r'3 неправильных варианта ответа[:\-]?\s*', '', response_content, flags=re.IGNORECASE)
    numbered = re.findall(r'\d+[.)]\s*([^;]+)', response_content)
    if numbered and len(numbered) >= 3:
        distractors = [d.strip() for d in numbered[:3]]
        return distractors
    split_distractors = [d.strip() for d in response_content.split(';') if d.strip()]
    if len(split_distractors) >= 3:
        distractors = split_distractors[:3]
        return distractors
    additional = re.split(r'\d+[.)]\s*', response_content)
    additional = [d.strip() for d in additional if d.strip()]
    if len(additional) >= 3:
        distractors = additional[:3]
        return distractors
    sentences = re.split(r'[.!?]\s*', response_content)
    sentences = [s.strip() for s in sentences if s.strip()]
    if len(sentences) >= 3:
        distractors = sentences[:3]
        logger.warning("Distractors parsed by sentences, other methods did not work.")
        return distractors
    logger.error("Failed to parse distractors from response.")
    return distractors

def generate_distractors(access_token: str, question: str, correct_answer: str) -> List[str]:
    """Генерация трех неправильных вариантов ответа для вопроса на основе вопроса и правильного ответа"""
    prompt = (
        '''
        Вы ассистент в области информационных технологий для создания тестов, задача которого — сгенерировать три неправильных варианта ответа (дистрактора) для вопроса с множественным выбором на основе предоставленного вопроса и правильного ответа, связанных с IT. Ваш ответ должен содержать ровно три неправильных варианта, разделенных точкой с запятой, без дополнительного текста, нумерации или объяснений. Дистракторы должны быть правдоподобными, но неверными в контексте вопроса и относиться к IT-тематике.
        Примеры:
            Вопрос: "Какой порт по умолчанию используется для HTTPS?" Правильный ответ: "443" Вывод: 80;22;12
            Вопрос: "Какой язык программирования выполняется в браузере и отвечает за динамическое поведение веб‑страниц?" Правильный ответ: "JavaScript" Вывод: Python;Java;C#
            Вопрос: "Какой менеджер пакетов применяется в проектах на Node.js?" Правильный ответ: "NPM" Вывод: pip;gem;Maven
        Задача: Вопрос: {question} Правильный ответ: {correct_answer} Вывод: [три неправильных варианта, разделенных точкой с запятой]
        '''
    )
    formatted_prompt = prompt.format(question=question, correct_answer=correct_answer)
    payload = json.dumps({
        "model": "GigaChat",
        "messages": [{"role": "user", "content": formatted_prompt}],
        "stream": False,
        "repetition_penalty": 1
    })
    completions_url = os.getenv("GIGACHAT_COMPLETIONS_URL")
    if not completions_url:
        logger.error("Missing GIGACHAT_COMPLETIONS_URL environment variable")
        raise ValueError("GIGACHAT_COMPLETIONS_URL is not set")

    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': 'Bearer ' + access_token
    }
    cert_path = os.getenv("GIGACHAT_CERT_PATH")
    if not cert_path:
        logger.error("Missing GIGACHAT_CERT_PATH environment variable")
        raise ValueError("GIGACHAT_CERT_PATH is not set")

    try:
        response = requests.post(completions_url, headers=headers, data=payload, verify=cert_path)
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error in GigaChat API request: {e}")
        return []
    if "choices" not in response_json or not response_json["choices"]:
        logger.error("GigaChat response does not contain needed data.")
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
        logger.error("Not enough distractors generated.")
        return []
    return distractors

class QAGenerator:
    def __init__(self,
                 qa_model_path: str = os.getenv("QA_MODEL_PATH"),
                 tokenizer_model: str = os.getenv("TOKENIZER_MODEL_PATH")):
        """Инициализация генератора вопросов и ответов с использованием модели и токенизатора"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_path).to(self.device)
        self.access_token = get_access_token()

    def gen_qa_pairs(self, text: str, num_pairs: int = 1) -> List[str]:
        """Генерация пар вопрос-ответ на основе предоставленного текста"""
        input_ids = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids.to(self.device)
        qa_ids = self.qa_model.generate(
            input_ids,
            max_length=128,
            num_beams=6,
            num_return_sequences=num_pairs,
        )
        qa_pairs = [self.tokenizer.decode(q, skip_special_tokens=True) for q in qa_ids]
        return qa_pairs

    def parse_qa_pairs(self, qa_pairs: List[str]) -> List[Dict[str, str]]:
        """Парсинг сгенерированных пар вопрос-ответ в структурированный формат"""
        processed_pairs = []
        seen_pairs = set()
        for qa in qa_pairs:
            if "Вопрос:" in qa and "Ответ:" in qa:
                parts = qa.split("Ответ:")
                if len(parts) == 2:
                    question = parts[0].replace("Вопрос:", "").strip()
                    answer = parts[1].strip()
                    pair_tuple = (question.lower(), answer.lower())
                    if pair_tuple not in seen_pairs:
                        seen_pairs.add(pair_tuple)
                        processed_pairs.append({
                            'question': question,
                            'answer': answer
                        })
        return processed_pairs


    def generate_qa(self, text: str, is_open: bool = False) -> List[Dict]:
        """Генерация вопросов с правильными ответами и, при необходимости, с вариантами ответа"""
        qa_strings = self.gen_qa_pairs(text)
        parsed_pairs = self.parse_qa_pairs(qa_strings)
        result = []
        for pair in parsed_pairs:
            question = pair['question']
            correct_answer = pair['answer']
            if is_open:
                result.append({
                    "Вопрос": question,
                    "Правильный_ответ": correct_answer
                })
            else:
                distractors = generate_distractors(self.access_token, question, correct_answer)
                if len(distractors) < 3:
                    logger.warning("Not enough distractors for question, skipped.")
                    continue
                options = distractors.copy()
                correct_index = random.randint(0, len(options))
                options.insert(correct_index, correct_answer)
                result.append({
                    "Вопрос": question,
                    "Варианты": options,
                    "Правильный_ответ": correct_index
                })
        return result