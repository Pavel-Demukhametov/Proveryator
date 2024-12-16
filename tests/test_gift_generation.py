# tests/test_gift_generation.py

import pytest
from internal.utils.gift_generation import escape_gift, convert_to_gift

class TestEscapeGift:
    def test_escape_empty_string(self):
        assert escape_gift("") == ""
    
    def test_escape_no_special_chars(self):
        input_text = "This is a test string."
        expected = "This is a test string."
        assert escape_gift(input_text) == expected


class TestConvertToGift:
    def test_convert_empty_data(self):
        data = []
        expected = ""
        assert convert_to_gift(data) == expected
    
    def test_convert_single_open_question_without_feedback(self):
        data = [
            {
                "type": "open",
                "question": "What is Python?",
                "answer": "A programming language."
            }
        ]
        expected = "::Q1:: What is Python? {=A programming language.}"
        assert convert_to_gift(data) == expected
    
    def test_convert_single_open_question_with_feedback(self):
        data = [
            {
                "type": "open",
                "question": "What is Python?",
                "answer": "A programming language.",
                "feedback": "Correct! Python is widely used."
            }
        ]
        expected = "::Q1:: What is Python? {=A programming language.#Correct! Python is widely used.}"
        assert convert_to_gift(data) == expected
    
    def test_convert_single_mc_question_with_string_options(self):
        data = [
            {
                "type": "mc",
                "question": "What is Python?",
                "answer": "A programming language.",
                "options": [
                    "A snake.",
                    "A programming language.",
                    "A type of coffee."
                ]
            }
        ]
        expected = "::Q1:: What is Python? {~A snake. =A programming language. ~A type of coffee.}"
        assert convert_to_gift(data) == expected
    
    def test_convert_single_mc_question_with_dict_options(self):
        data = [
            {
                "type": "mc",
                "question": "What is Python?",
                "answer": "A programming language.",
                "options": [
                    {"text": "A snake.", "feedback": "Incorrect."},
                    {"text": "A programming language.", "feedback": "Correct!"},
                    {"text": "A type of coffee.", "feedback": "Incorrect."}
                ]
            }
        ]
        expected = "::Q1:: What is Python? {~A snake.#Incorrect. =A programming language.#Correct! ~A type of coffee.#Incorrect.}"
        assert convert_to_gift(data) == expected
    
    def test_convert_mc_question_mixed_options(self):
        data = [
            {
                "type": "mc",
                "question": "Select the colors in the Russian flag.",
                "answer": "Blue",
                "options": [
                    {"text": "Blue", "feedback": "Correct!"},
                    "Red",
                    {"text": "Green", "feedback": "Incorrect."},
                    "White"
                ]
            }
        ]
        expected = "::Q1:: Select the colors in the Russian flag. {=Blue#Correct! ~Red ~Green#Incorrect. ~White}"
        assert convert_to_gift(data) == expected
    
    def test_convert_multiple_questions(self):
        data = [
            {
                "type": "open",
                "question": "What is Python?",
                "answer": "A programming language."
            },
            {
                "type": "mc",
                "question": "What is the capital of France?",
                "answer": "Paris",
                "options": [
                    "London",
                    "Berlin",
                    "Paris",
                    "Madrid"
                ]
            }
        ]
        expected = (
            "::Q1:: What is Python? {=A programming language.}\n\n"
            "::Q2:: What is the capital of France? {~London ~Berlin =Paris ~Madrid}"
        )
        assert convert_to_gift(data) == expected
    
    def test_convert_unknown_question_type(self):
        data = [
            {
                "type": "essay",
                "question": "Describe Python.",
                "answer": "A versatile programming language."
            }
        ]
        expected = ""
        assert convert_to_gift(data) == expected
    
    def test_convert_invalid_option_format(self):
        data = [
            {
                "type": "mc",
                "question": "What is Python?",
                "answer": "A programming language.",
                "options": [
                    {"text": "A snake.", "feedback": "Incorrect."},
                    123,
                    "A programming language."
                ]
            }
        ]
        expected = "::Q1:: What is Python? {~A snake.#Incorrect. =A programming language.}"
        assert convert_to_gift(data) == expected
    
