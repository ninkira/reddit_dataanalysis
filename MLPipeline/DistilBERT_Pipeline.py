import json
import os


import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric, metric
from numpy import exp
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from transformers import DistilBertTokenizerFast
import torch
from DataScripts.DataProcessing.ProcessingBase import ProcessingBase

from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW

from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
"""ML Pipeline für Klassizifierung von AITA Posts in YTA oder NTA. 
 Basierend auf: https://huggingface.co/transformers/custom_datasets.html#seq-imdb."""


class AITADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class SequenceClassification():
    """
    DistilBert Pipeline basierend auf dem oben genannten Beispiel zum Trainieren auf dem IMDB Datenset. Hier wird
    die HuggingFace Trainer-Klasse zum Trainieren, Evaluieren und Testen des DistilBert Models verwendet.
    """
    def load_datasets(self):  # https: // huggingface.co / docs / datasets / loading_datasets.html
        dataset = load_dataset("json", data_files={'train': 'DataFiles/DataSets/DataSet-L/dataset_l_train.json',
                                                   'test': 'DataFiles/DataSets/DataSet-L/dataset_l_test.json',
                                                   'validation': 'DataFiles/DataSets/DataSet-L/dataset_l_val.json'},
                               field="data")

        print("dataset info", dataset['train'].info)
        print("dataset columns", dataset['train'].column_names)

        # get the actual dataset for each part of training and finetuning for later
        train_ds = dataset['train']
        test_ds = dataset['test']
        val_ds = dataset['validation']

        return train_ds, test_ds, val_ds

    def split_dataset_attributes(self, train_ds, test_ds, val_ds):

        train_texts = train_ds['text']
        train_labels = train_ds['label']

        test_texts = test_ds['text']
        test_labels = test_ds['label']

        val_texts = val_ds['text']
        val_labels = val_ds['label']

        return train_texts, train_labels, test_texts, test_labels, val_texts, val_labels

    def preprocessing(self, train_texts, train_labels, test_texts, test_labels, val_texts, val_labels):
        """
        Preprocessing, daher enconding der Texte und Labels aus den Datensets.
        :param train_texts:
        :param train_labels:
        :param test_texts:
        :param test_labels:
        :param val_texts:
        :param val_labels:
        :return: Fertige Datensets mithilfe der AITADataset Klasse, welche zum Trainieren des Modells verwendet werden können.
        """
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        train_dataset = AITADataset(train_encodings, train_labels)
        val_dataset = AITADataset(val_encodings, val_labels)
        test_dataset = AITADataset(test_encodings, test_labels)

        return train_dataset, test_dataset, val_dataset,

    def load_pretrained_model(self, directory_path):

        if os.path.exists(directory_path) == True:
            print("Lade aus lokaler Datei")
            model = DistilBertForSequenceClassification.from_pretrained(directory_path)
        else:
            print("Lade Model von HuggingFace")
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
            model.save_pretrained(directory_path)
        return model

    def train_model(self, model, train_dataset, val_dataset):
        print("=== Training startet ===")
        torch.cuda.empty_cache()
        training_args = TrainingArguments(
            output_dir='./results',  # output directory
            num_train_epochs=7,  # total number of training epochs
            per_device_train_batch_size=6,  # batch size per device during training
            per_device_eval_batch_size=6,  # batch size for evaluation
            warmup_steps=500,  # number of warmup steps for learning rate scheduler
            weight_decay=0.01,  # strength of weight decay
            logging_dir='./logs',  # directory for storing logs
            logging_steps=30,
        )

        trainer = Trainer(
            model=model,  # the instantiated 🤗 Transformers model to be trained
            args=training_args,  # training arguments, defined above
            train_dataset=train_dataset,  # training dataset
            eval_dataset=val_dataset,  # evaluation dataset
        )

        trainer.train()
        # TODO: Implement EvaluationSet
        trainer.save_model(output_dir="Models/Trained/LargeTrainedModel/")
        print("=== Training finished ===")



    def predict(self, test_dataset, model):
        """
        Lässt das Model Vorhersagen machen mithilfe der Trainer.predict()-Methode. Außerdem werden Auswertungsmetriken wie z.B. die Accuracy des
        Modells berechnet. Die Methode speichert sowohl die Metriken als auch eine Auswertung der Texte mit dem vorhergesagten Label in jeweils eine
        eigene JSON-Datei.
        :param test_dataset: Test-Datenset zudem die vorhersagen gemacht werden.
        :param model: das finegetunte, d.h. trainierte DistilBert-Modell
        """
        print("=== Starte Prediction ===")
        trainer = Trainer(
            model=model,
        )
        predictions = trainer.predict(test_dataset)

        result_frame = pd.DataFrame()
        label = ""
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        for idx, prediction in enumerate(predictions.predictions):
            og_dataset_item =  test_dataset.__getitem__(idx)
            decoded_text = tokenizer.decode(og_dataset_item['input_ids'])
            # Berechne Wahrscheinlichkeiten der Vorhersagen aus dem Predictions Objekt
            # Lade Softmax Funktion
            sm = torch.nn.Softmax(dim=-1)

            # Berechne Wahrscheinlichkeit nur mit Prediction Logits
            x = torch.from_numpy(prediction) # Wandelt in Tensor
            probabilities = sm(x) # gibt wegen hoher Logits-Werte unverständliche Werte aus
            # Bereinigte Wahrscheinlichkeiten
            converted_prediction = prediction
            converted_prediction /= 10 # Wandelt die Logit-Werte in Kommazahlen
            y = torch.from_numpy(prediction)
            converted_probabilities = sm(y)

            og_label = og_dataset_item['labels'].numpy()
            if prediction[0] > prediction[1]:
                label = "Label 0 - NTA"
            else:
                label = "Label 1 - YTA"

            result_frame = result_frame.append({
                'Original_Text': decoded_text,
                'Original_Label': str(og_label),
                'Prediction': str(prediction),
                'Predicted_Label_Argmax': str(np.argmax(prediction)),
                'Predicted_Label_Softmax':str(probabilities) + " Summe: " + str(sum(probabilities)),
                'Predicted_Label_Softmax_Converted': str(converted_probabilities) + " Summe: " + str(sum(converted_probabilities)),
                'Predicted_Label': label,
                }, ignore_index=True)
            label = ""
            x = ''


        metrics = self.compute_metrics(predictions)
        # Back-Up: Speicher Metriken in Cache
        trainer.save_metrics(metrics=metrics, split="all")
        trainer.save_metrics(metrics=metrics, split="train")
        trainer.save_metrics(metrics=metrics, split="test")
        trainer.save_metrics(metrics=metrics, split="eval")

        print("=== Speicher Ergebnisse ===")

        result = result_frame.to_json(orient="records")
        parsed = json.loads(result)
        ProcessingBase().save_json_to_file(parsed, "Results/ML_Results/LargeSizedTraining", "LargeSizedPredictions8.json")

        ProcessingBase().save_json_to_file(metrics, "Results/ML_Results/LargeSizedTraining",
                                           "LargeSizedMetrics8.json")

    def compute_metrics(self, pred):
        print("pred", pred)
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def run_pipeline(self):
        """
        __main__ methode für dieses Skript. Hier werden alle nötigen Funktionen vom Instanzieren bis zur Auswertung des Models aufgerufen.
        """
        # create dataset from files - returns train, test, validation split as dictionary
        model = self.load_pretrained_model("Models/distilbert-base-uncased")
        train_ds, test_ds, val_ds = self.load_datasets()
        train_texts, train_labels, test_texts, test_labels, val_texts, val_labels = self.split_dataset_attributes(
            train_ds, test_ds, val_ds)

        train_dataset, val_dataset, test_dataset = self.preprocessing(train_texts, train_labels, test_texts,
                                                                      test_labels, val_texts, val_labels)

        #self.train_model(model, train_dataset, val_dataset)

        trained_model = DistilBertForSequenceClassification.from_pretrained("Models/Trained/LargeTrainedModel")

        self.predict(test_dataset, trained_model)
        # https://huggingface.co/transformers/usage.html for visualisation / plotting of results
        # Source https://huggingface.co/transformers/main_classes/trainer.html
