import json

def escape_gift(text):
    """
    Экранирует специальные символы в строке для корректного использования в формате GIFT.
    """
    replacements = {
        '\\': '\\\\',
        '{': '\\{',
        '}': '\\}',
        '=': '\\=',
        '~': '\\~',
        '#': '\\#',
        ':': '\\:',
        '*': '\\*',
    }
    for char, escape in replacements.items():
        text = text.replace(char, escape)
    return text

def convert_to_gift(data):
    """
    Преобразует список вопросов в строку формата GIFT для импорта в Moodle.

    Поддерживаемые типы вопросов:
      - "open": открытый вопрос с одним правильным ответом и опциональным обратным связью.
      - "mc": вопрос с несколькими вариантами, где один правильный.
    """
    gift_questions = []
    question_number = 1

    for item in data:
        question_type = item.get("type")
        question_text = escape_gift(item.get("question", ""))
        correct_answer = escape_gift(item.get("answer", ""))

        label = f"Q{question_number}"
        question_number += 1 

        if question_type == "open":
            feedback = escape_gift(item.get("feedback", ""))
            if feedback:
                gift_questions.append(f"::{label}:: {question_text} {{={correct_answer}#{feedback}}}")
            else:
                gift_questions.append(f"::{label}:: {question_text} {{={correct_answer}}}")

        elif question_type == "mc":
            options = item.get("options", [])
            gift_answers = []

            for option in options:
                if isinstance(option, dict):
                    option_text = escape_gift(option.get("text", ""))
                    option_feedback = escape_gift(option.get("feedback", ""))
                elif isinstance(option, str):
                    option_text = escape_gift(option)
                    option_feedback = ""
                else:
                    print(f"Неизвестный формат варианта ответа: {option}")
                    continue

                if option_text == correct_answer:
                    if option_feedback:
                        gift_answers.append(f"={option_text}#{option_feedback}")
                    else:
                        gift_answers.append(f"={option_text}")
                else:
                    if option_feedback:
                        gift_answers.append(f"~{option_text}#{option_feedback}")
                    else:
                        gift_answers.append(f"~{option_text}")

            gift_answer_str = " ".join(gift_answers)
            gift_questions.append(f"::{label}:: {question_text} {{{gift_answer_str}}}")

        else:
            print(f"Неизвестный тип вопроса: {question_type}")

    return "\n\n".join(gift_questions)