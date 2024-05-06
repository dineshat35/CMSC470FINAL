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
        self.load_models()

    def load_models(self):
        """Load all models"""
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 't5-model-params')
        self.load_seq2seq_model(model_dir)
        self.load_flan_models('google/flan-t5-large', 'google/flan-t5-small')

    def load_seq2seq_model(self, model_dir):
        """Load saved models"""
        self.test_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.test_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        self.test_model.eval()

    def load_flan_models(self, large_model_id, small_model_id):
        """Load hugging face models."""        
        self.tokenizer_flan_t5 = AutoTokenizer.from_pretrained(large_model_id)
        self.model_flan_t5 = AutoModelForSeq2SeqLM.from_pretrained(large_model_id)
        self.tokenizer_t5 = AutoTokenizer.from_pretrained(small_model_id)
        self.model_t5 = AutoModelForSeq2SeqLM.from_pretrained(small_model_id)

    def guess_and_buzz(self, question_texts):
        """Generate answers from all models for given questions"""
        total_answers = self.generate_answers(question_texts)
        # Display the model's guesses before voting
        # print("Answers Before Voting Mechanism:")

        # for question, model_answers in zip(question_texts, total_answers):
        #     print(f"{question}\nModel Guesses: {model_answers}\n\n")
        return self.ensemble_tfidf_voting(total_answers)

    def generate_answers(self, question_texts):
        """Generate answers from each model."""
        # Tokenize and generate answers using each model
        return [(self.decode_answer(self.model_flan_t5, self.tokenizer_flan_t5, question),
                 self.decode_answer(self.model_t5, self.tokenizer_t5, question),
                 self.decode_answer(self.test_model,self.test_tokenizer , question))
                for question in question_texts]

    def decode_answer(self, model, tokenizer, input_text):
        input_ids = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**input_ids, max_new_tokens=5, output_scores=True, return_dict_in_generate=True)
        
        decoded_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        
        if outputs.scores:
            log_probs = [torch.nn.functional.log_softmax(score, dim=-1) for score in outputs.scores]
            scores = []
            for i in range(len(outputs.sequences[0]) - 1):
                selected_log_prob = log_probs[i][0, outputs.sequences[0][i + 1]].item()
                scores.append(selected_log_prob)
            confidence_score = np.exp(np.mean(scores))
        else:
            confidence_score = None

        return decoded_text, confidence_score

    def ensemble_tfidf_voting(self, all_answers):
        """Find answer with highest confidence"""
        for answers in all_answers:
            highest_confidence_answer = max(answers, key=lambda x: x[1])
            yield {'guess': highest_confidence_answer[0], 'confidence': highest_confidence_answer[1]}

        # for answers in all_answers:
        #     texts = [answer[0] for answer in answers]
            
        #     vectorizer = TfidfVectorizer()
        #     tfidf_matrix = vectorizer.fit_transform(texts)
        #     cosine_scores = cosine_similarity(tfidf_matrix)
        #     most_similar_index = np.argmax(np.mean(cosine_scores, axis=0))
        #     yield {'guess': answers[most_similar_index][0], 'confidence': answers[most_similar_index][1]}

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
