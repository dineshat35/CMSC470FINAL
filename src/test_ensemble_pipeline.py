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
            guess_text = output[0]
            confidence = output[1]
            results = {'guess': guess_text, 'confidence': confidence}
        return results

# # From class eval.py code
# def normalize_answer(answer):
#     """
#     Remove superflous components to create a normalized form of an answer that
#     can be more easily compared.
#     """
#     from unidecode import unidecode
    
#     if answer is None:
#         return ''
#     reduced = unidecode(answer)
#     reduced = reduced.replace("_", " ")
#     if "(" in reduced:
#         reduced = reduced.split("(")[0]
#     reduced = "".join(x for x in reduced.lower() if x not in string.punctuation)
#     reduced = reduced.strip()

#     for bad_start in ["the ", "a ", "an "]:
#         if reduced.startswith(bad_start):
#             reduced = reduced[len(bad_start):]
#     return reduced.strip()
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
            input_ids = tokenizer(question, return_tensors="pt", padding=True, truncation=False).input_ids
            with torch.no_grad():
                outputs = model.generate(input_ids, max_new_tokens=5, output_scores=True, return_dict_in_generate=True)
            decoded_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            confidence_score = self.calculate_confidence(outputs.scores)
            raw_answers.append((decoded_text, confidence_score))

        # normalization if needed
        # total_scores = sum([score for _, score in raw_answers])
        # answers = [(text, score / total_scores if total_scores > 0 else 0) for text, score in raw_answers]
        
        return raw_answers

    def calculate_confidence(self, scores):
        if scores:
            log_probs = [torch.nn.functional.log_softmax(score, dim=-1) for score in scores]
            selected_log_probs = [log_probs[i][0, scores[i].argmax()].item() for i in range(len(log_probs))]
            confidence_score = np.exp(np.mean(selected_log_probs))
        else:
            confidence_score = None
        return confidence_score

    def ensemble_tfidf_voting(self, all_answers):
        return max(all_answers, key=lambda x: x[1]) if all_answers else (None, 0)


# from transformers.pipelines import Pipeline, PIPELINE_REGISTRY
# from transformers import AutoModelForSeq2SeqLM, TFAutoModelForSeq2SeqLM
# from test_ensemble import TestEnsembleQAPipeline
# from transformers import pipeline

# # Register your custom pipeline for PyTorch and TensorFlow models
# PIPELINE_REGISTRY.register_pipeline("test-qa", 
#                                     pipeline_class=TestEnsembleQAPipeline, 
#                                     pt_model=AutoModelForSeq2SeqLM, 
#                                     tf_model=TFAutoModelForSeq2SeqLM)
# qa_pipe = pipeline("test-qa", model="google/flan-t5-small", tokenizer="google/flan-t5-small")

# qa_pipe.push_to_hub("test-qa")