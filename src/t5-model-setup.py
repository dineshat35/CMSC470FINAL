import os
import torch
from datasets import load_dataset, load_metric
import evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np

# Clear CUDA cache
torch.cuda.empty_cache()

# Set environment variables for Hugging Face configurations
os.environ["HF_HOME"] = "/root/.cache/huggingface/"
os.environ["TRANSFORMERS_CACHE"] = "/root/.cache/huggingface/transformers"
os.environ["HF_DATASETS_CACHE"] = "/root/.cache/huggingface/datasets"
os.environ["HF_METRICS_CACHE"] = "/root/.cache/huggingface/metrics"
os.environ["HF_TOKEN"] = "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"

data = load_dataset('qanta', 'mode=first,char_skip=25', trust_remote_code=True)
tokenizer = T5Tokenizer.from_pretrained('consciousAI/question-answering-generative-t5-v1-base-s-q-c', legacy=False)

def preprocess_data(examples):
  input_text = ["question: " + q + " context: " + c for q, c in zip(examples['first_sentence'], examples['text'])]
  targets = examples['answer']

  # Tokenize input texts
  model_inputs = tokenizer(input_text, max_length=512, truncation=True, padding="max_length", return_tensors='pt')

  # Tokenize target texts
  labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length", return_tensors='pt')
  """ example tensor:{'input_ids': tensor([[ 822,   10,   37,  ...,    0,    0,    0],
        [ 822,   10,  100,  ...,    0,    0,    0],
        [ 822,   10,   86,  ...,    0,    0,    0],
        ...,
        [ 822,   10, 8529,  ...,    0,    0,    0],
        [ 822,   10,  100,  ...,    0,    0,    0],
        [ 822,   10,   37,  ...,    0,    0,    0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        ...,
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0],
        [1, 1, 1,  ..., 0, 0, 0]])}
  """
  model_inputs['labels'] = labels['input_ids']
  return model_inputs


# f1_metric = evaluate.load("f1")
# exact_match_metric = evaluate.load("exact_match")

# def compute_metrics(eval_pred):
#   logits, labels = eval_pred
#   print("Logits shape:", logits.shape)

#   predictions = np.argmax(logits, axis=-1)

#   # Assuming you have a way to decode your predictions and labels from token IDs to text
#   decoded_preds = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]
#   decoded_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

#   # Compute F1 and Exact Match scores
#   f1_result = f1_metric.compute(predictions=decoded_preds, references=decoded_labels)
#   exact_match_result = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)

#   return {
#       "f1": f1_result["f1"],
#       "exact_match": exact_match_result["exact_match"]
#   }


tokenized_datasets = data.map(preprocess_data, batched=True, load_from_cache_file=False,
                              remove_columns=['id', 'qanta_id', 'proto_id', 'qdb_id', 'dataset', 'full_question',
                                              'first_sentence', 'char_idx', 'sentence_idx', 'tokenizations', 'page',
                                              'raw_answer', 'fold', 'gameplay', 'category', 'subcategory', 'tournament', 'year'])

small_train_dataset = tokenized_datasets['buzztrain'].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets['buzzdev'].shuffle(seed=42).select(range(1000))
model = T5ForConditionalGeneration.from_pretrained('consciousAI/question-answering-generative-t5-v1-base-s-q-c')

training_args = Seq2SeqTrainingArguments(
  output_dir="./models",
  evaluation_strategy="epoch",
  learning_rate=0.0003,
  gradient_accumulation_steps=3,
  per_device_train_batch_size=3,
  num_train_epochs=5,
  weight_decay=0.01,
  seed=42,
)

trainer = Seq2SeqTrainer(
  model=model,
  args=training_args,
  train_dataset=small_train_dataset,
  eval_dataset=small_eval_dataset,
  tokenizer=tokenizer,
  # compute_metrics=compute_metrics
)

trainer.train()
