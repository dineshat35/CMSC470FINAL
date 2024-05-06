from typing import List, Tuple
import numpy as np
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer, ElectraModel, ElectraForCausalLM, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
import torch
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gzip
from transformers import Text2TextGenerationPipeline

class TestEnsembleQAPipeline(Text2TextGenerationPipeline):
    def __init__(self, model=None, tokenizer=None, framework="pt", **kwargs):  
        super().__init__(model=model, tokenizer=tokenizer, framework=framework)
        self.quiz_bowl_model = QuizBowlModel()  # Initializes your QuizBowl model

    def _forward(self, model_inputs, **generate_kwargs):
        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in model_inputs["input_ids"]]
        results = self.quiz_bowl_model.guess_and_buzz(questions)
        return results

    def postprocess(self, model_outputs):
        results = {}
        for output in model_outputs:
            guess_text = output['guess']
            confidence = output['confidence']
            results = {'guess': guess_text, 'confidence': confidence}
        return results

# From class eval.py code
def normalize_answer(answer):
    """
    Remove superflous components to create a normalized form of an answer that
    can be more easily compared.
    """
    from unidecode import unidecode
    
    if answer is None:
        return ''
    reduced = unidecode(answer)
    reduced = reduced.replace("_", " ")
    if "(" in reduced:
        reduced = reduced.split("(")[0]
    reduced = "".join(x for x in reduced.lower() if x not in string.punctuation)
    reduced = reduced.strip()

    for bad_start in ["the ", "a ", "an "]:
        if reduced.startswith(bad_start):
            reduced = reduced[len(bad_start):]
    return reduced.strip()
class QuizBowlModel:
    def __init__(self):
        self.load_models()

    def load_models(self):
        """Load all models"""
        # model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 't5-model-params')
        # self.load_seq2seq_model(model_dir)
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
                 self.decode_answer(self.model_t5, self.tokenizer_t5, question))
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
