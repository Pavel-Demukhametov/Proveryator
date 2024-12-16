import json

def escape_gift(text):
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
    gift_questions = []
    question_number = 1

    for item in data:
        question_type = item.get("type")
        question_text = escape_gift(item.get("question", ""))
        correct_answer = escape_gift(item.get("answer", ""))

        label = f"Q{question_number}"
        question_number += 1  # Увеличиваем счетчик для следующего вопроса

        if question_type == "open":
            # Предполагается, что для открытых вопросов также может быть обратная связь
            feedback = escape_gift(item.get("feedback", ""))
            if feedback:
                gift_questions.append(f"::{label}:: {question_text} {{={correct_answer}#{feedback}}}")
            else:
                gift_questions.append(f"::{label}:: {question_text} {{={correct_answer}}}")

        elif question_type == "mc":
            # Формат для вопроса с множественным выбором с обратной связью (если доступна)
            options = item.get("options", [])
            gift_answers = []

            for option in options:
                if isinstance(option, dict):
                    # Вариант ответа с обратной связью
                    option_text = escape_gift(option.get("text", ""))
                    option_feedback = escape_gift(option.get("feedback", ""))
                elif isinstance(option, str):
                    # Вариант ответа без обратной связи
                    option_text = escape_gift(option)
                    option_feedback = ""
                else:
                    print(f"Неизвестный формат варианта ответа: {option}")
                    continue

                if option_text == correct_answer:
                    # Правильный ответ
                    if option_feedback:
                        gift_answers.append(f"={option_text}#{option_feedback}")
                    else:
                        gift_answers.append(f"={option_text}")
                else:
                    # Неправильный ответ
                    if option_feedback:
                        gift_answers.append(f"~{option_text}#{option_feedback}")
                    else:
                        gift_answers.append(f"~{option_text}")

            # Объединяем все ответы в одну строку
            gift_answer_str = " ".join(gift_answers)
            gift_questions.append(f"::{label}:: {question_text} {{{gift_answer_str}}}")

        else:
            print(f"Неизвестный тип вопроса: {question_type}")

    return "\n\n".join(gift_questions)