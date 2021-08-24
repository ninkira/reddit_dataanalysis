from transformers import pipeline
import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

class TransformerTest:
    classifier = pipeline("sentiment-analysis")
    classifier("I've been waiting for a HuggingFace course my whole life.")


#https://huggingface.co/transformers/task_summary.html concept
#https://huggingface.co/transformers/training.html
class QuestionAnswering:
    def tokenize_function(self):
        # to be filled
        print("tokenize")

    def train(self):
        print("train")
