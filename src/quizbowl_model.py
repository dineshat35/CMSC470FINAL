from typing import List, Tuple
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
kTOY_DATA = {"tiny": [{"text": "capital England", "page": "London"},
                      {"text": "capital Russia", "page": "Moscow"},
                      {"text": "currency England", "page": "Pound"},
                      {"text": "currency Russia", "page": "Rouble"}],
             "train": [{'page': 'Maine', 'text': 'For 10 points, name this New England state with capital at Augusta.'},
                       {'page': 'Massachusetts', 'text': 'For ten points, identify this New England state with capital at Boston.'},
                       {'page': 'Boston', 'text': 'For 10 points, name this city in New England, the capital of Massachusetts.'},
                       {'page': 'Jane_Austen', 'text': 'For 10 points, name this author of Pride and Prejudice.'},
                       {'page': 'Jane_Austen', 'text': 'For 10 points, name this author of Emma and Pride and Prejudice.'},
                       {'page': 'Wolfgang_Amadeus_Mozart', 'text': 'For 10 points, name this composer of Magic Flute and Don Giovanni.'},
                       {'page': 'Wolfgang_Amadeus_Mozart', 'text': 'Name this composer who wrote a famous requiem and The Magic Flute.'},
                       {'page': "Gresham's_law", 'text': 'For 10 points, name this economic principle which states that bad money drives good money out of circulation.'},
                       {'page': "Gresham's_law", 'text': "This is an example -- for 10 points \\-- of what Scotsman's economic law, which states that bad money drives out good?"},
                       {'page': "Gresham's_law", 'text': 'FTP name this economic law which, in simplest terms, states that bad money drives out the good.'},
                       {'page': 'Rhode_Island', 'text': "This colony's Touro Synagogue is the oldest in the United States."},
                       {'page': 'Lima', 'text': 'It is the site of the National University of San Marcos, the oldest university in South America.'},
                       {'page': 'College_of_William_&_Mary', 'text': 'For 10 points, identify this oldest public university in the United States, a college in Virginia named for two monarchs.'}],
              "dev": [{'text': "This capital of England", "top": 'Maine', "second": 'Boston'},
                      {'text': "The author of Pride and Prejudice", "top": 'Jane_Austen',
                           "second": 'Jane_Austen'},
                      {'text': "The composer of the Magic Flute", "top": 'Wolfgang_Amadeus_Mozart',
                           "second": 'Wolfgang_Amadeus_Mozart'},
                      {'text': "The economic law that says 'good money drives out bad'",
                           "top": "Gresham's_law", "second": "Gresham's_law"},
                      {'text': "located outside Boston, the oldest University in the United States",
                           "top": 'College_of_William_&_Mary', "second": 'Rhode_Island'}]
                }
class QuizBowlModel:
    def __init__(self):
        """
        Load your model(s) and whatever else you need in this function.
        """
        # Loading the FLAN-T5-large and FLAN-T5-small models to check
        self.tokenizer_flan_t5 = AutoTokenizer.from_pretrained('google/flan-t5-base')
        self.model_flan_t5 = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
        
        self.tokenizer_t5 = AutoTokenizer.from_pretrained('google/flan-t5-small')
        self.model_t5 = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small')
        
    def guess_and_buzz(self, question_texts):
        """
        This function accepts a list of question strings and returns a list of tuples containing
        the guess and a boolean indicating whether to buzz.
        """
        inputs_flan_t5 = self.tokenizer_flan_t5(question_texts, return_tensors="pt", padding=True, truncation=True)
        inputs_t5 = self.tokenizer_t5(question_texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            answers_flan_t5 = self.model_flan_t5.generate(**inputs_flan_t5, max_new_tokens=50)
            answers_t5 = self.model_t5.generate(**inputs_t5, max_new_tokens=50)
        
        decoded_answers_flan_t5 = [self.tokenizer_flan_t5.decode(ans, skip_special_tokens=True) for ans in answers_flan_t5]
        decoded_answers_t5 = [self.tokenizer_t5.decode(ans, skip_special_tokens=True) for ans in answers_t5]
        # print(decoded_answers_flan_t5)
        # print(decoded_answers_t5)
        
        # Combine answers by simple averaging
        final_answers = [(a1, a2) for a1, a2 in zip(decoded_answers_flan_t5, decoded_answers_t5)]
        
        return final_answers

if __name__ == "__main__":
    model = QuizBowlModel()
    questions = ["Who wrote 'Pride and Prejudice'?", "What is the capital of France?"]
    for key in kTOY_DATA:
        for item in kTOY_DATA[key]:
            questions.append(item['text'])
    answers = model.guess_and_buzz(questions)
    for quest, ans in zip(questions, answers):
        print(str(quest) + "   :   " + str(ans) )

