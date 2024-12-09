import json

# Функция для преобразования JSON в формат GIFT
def convert_to_gift(data):
    gift_questions = []

    for item in data:
        question_type = item.get("type")
        question_text = item.get("question")
        correct_answer = item.get("answer")
        
        if question_type == "open":
            # Формат для открытого вопроса
            gift_questions.append(f"::{question_text}:: {question_text} {{{correct_answer}}}")
        
        elif question_type == "mc":
            # Формат для вопроса с множественным выбором
            options = item.get("options")
            gift_answer = " ~".join([f"~{option}" for option in options if option != correct_answer])
            gift_answer = f"={correct_answer} {gift_answer}"
            gift_questions.append(f"::{question_text}:: {question_text} {{{gift_answer}}}")

    return "\n".join(gift_questions)