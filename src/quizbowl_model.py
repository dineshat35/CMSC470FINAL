from typing import List, Tuple
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer, ElectraModel, ElectraForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch
from eval import normalize_answer
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gzip

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
        model_configs = {
            'flan-t5-large': {'model': 'google/flan-t5-large', 'tokenizer': 'google/flan-t5-large'},
            'flan-t5-small': {'model': 'google/flan-t5-small', 'tokenizer': 'google/flan-t5-small'},
            'flan-t5-base': {'model': 'google/flan-t5-base', 'tokenizer': 'google/flan-t5-base'},
            'flan-t5-finetuned': {'model': 'dbalasub/finetuned-t5-qanta', 'tokenizer': 'dbalasub/finetuned-t5-qanta'},
            'flan-t5-adv-finetuned': {'model': 'dbalasub/finetuned-t5-adv-qanta', 'tokenizer': 'dbalasub/finetuned-t5-adv-qanta'}
        }
        self.models = {}
        self.tokenizers = {}
        self.load_models(model_configs)

    def load_models(self, model_configs):
        """Load multiple models based on configuration."""
        for model_name, config in model_configs.items():
            tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'])
            model = AutoModelForSeq2SeqLM.from_pretrained(config['model'])
            model.eval()
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer

    def guess_and_buzz(self, question_texts):
        total_answers = [self.generate_answers(question) for question in question_texts]
        # here to check all models responses if needed
        # for question, model_answers in zip(question_texts, total_answers):
        #     print(f"{question}\nModel Guesses: {model_answers}\n")
        return [self.ensemble_tfidf_voting(answers) for answers in total_answers]

    def generate_answers(self, question):
        raw_answers = []
        for model_name, model in self.models.items():
            tokenizer = self.tokenizers[model_name]
            input_ids = tokenizer(question, return_tensors="pt", padding=True, truncation=True).input_ids
            with torch.no_grad():
                outputs = model.generate(input_ids, max_new_tokens=5, output_scores=True, return_dict_in_generate=True)
            decoded_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            confidence_score = self.calculate_confidence(outputs.scores)
            raw_answers.append((decoded_text, confidence_score))

        total_scores = sum([score for _, score in raw_answers])
        answers = [(text, score / total_scores if total_scores > 0 else 0) for text, score in raw_answers]
        
        return answers

    def calculate_confidence(self, scores):
        if scores:
            log_probs = [torch.nn.functional.log_softmax(score, dim=-1) for score in scores]
            selected_log_probs = [log_probs[i][0, scores[i].argmax()].item() for i in range(len(log_probs))]
            confidence_score = np.exp(np.mean(selected_log_probs))
        else:
            confidence_score = None
        return confidence_score

    def ensemble_tfidf_voting(self, answers):
        return max(answers, key=lambda x: x[1]) if answers else (None, 0)

if __name__ == "__main__":
    # Initialize the QuizBowlModel
    model = QuizBowlModel()
    hardcoded_questions = ["Who wrote 'Pride and Prejudice'?", "What is the capital of France?"]
    # Load questions from a compressed JSON file
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'qanta.guessdev.json.gz')
    with gzip.open(file_path, 'rt', encoding='utf-8') as file:
        data = json.load(file)
    true_answers = ["Jane Austen", "Paris"]

    loaded_questions = [item['text'] for item in data]  
    true_answers += [item['answer'] for item in data]
    questions = hardcoded_questions + loaded_questions[:8]  # Increase if needed
    total_answers = model.guess_and_buzz(questions)
    
    print("Final Answers with Voting Mechanism:")
    # Display the model's after entire ensemble approach
    for question, model_answers, true_answer in zip(questions, total_answers, true_answers):
        print(f"{question}\nModel Guesses: {model_answers}\nCorrect Answer: {true_answer}\n\n")
