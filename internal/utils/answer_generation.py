# internal/utils/answer_generation.py

import random
from typing import List

def generate_incorrect_answers(correct_answer: str, num_incorrect: int = 3) -> List[str]:
    """
    Генерирует список неверных вариантов ответов для множественного выбора.
    
    В текущей реализации неверные ответы генерируются как "Неверный ответ 1", "Неверный ответ 2", и т.д.
    
    :param correct_answer: Правильный ответ.
    :param num_incorrect: Количество неверных вариантов.
    :return: Список вариантов ответов, включая правильный и неверные ответы, перемешанные случайным образом.
    """
    incorrect_answers = [f"Неверный ответ {i}" for i in range(1, num_incorrect + 1)]
    all_answers = [correct_answer] + incorrect_answers
    random.shuffle(all_answers)
    return all_answers
