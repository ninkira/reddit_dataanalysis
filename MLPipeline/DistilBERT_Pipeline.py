import json
import os


import numpy as np
import pandas as pd
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

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

    def __init__(self):
        self.model_name = 'distilbert-base-uncased'
        self.target_directory_results = "Results/ML_Results/LargeSizedTraining"
        self.target_directory_models = "Models/Trained/LargeTrainedModel"

    def load_datasets(self):
        """
        Lädt die verschiedenen JSON-Datensets als Dataset Objs.
        :return: trains_ds, test_ds, val_ds -> Dataset Objs
        """
        dataset = load_dataset("json", data_files={'train': 'DataFiles/DataSets/DataSet-L/dataset_l_train.json',
                                                   'test': 'DataFiles/DataSets/DataSet-L/dataset_l_test.json',
                                                   'validation': 'DataFiles/DataSets/DataSet-L/dataset_l_val.json'},
                               field="data")
        # Aufteilung aus dem dataset Objekt.
        train_ds = dataset['train']
        test_ds = dataset['test']
        val_ds = dataset['validation']

        return train_ds, test_ds, val_ds

    def split_dataset_attributes(self, train_ds, test_ds, val_ds):
        """
        Splittet die Datensets in ihre Texte und Labels nach Columns bzw. Spaltennamen.
        :param train_ds:
        :param test_ds:
        :param val_ds:
        :return:
        """
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
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)

        train_encodings = tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = tokenizer(val_texts, truncation=True, padding=True)
        test_encodings = tokenizer(test_texts, truncation=True, padding=True)

        # Visualisiere Econding Attributes - Hier sieht man sehr gut, dass die Posts durch Truncation und Padding auf die selbe Länge bearbeitet wurden.

        print("=== Tokenized Data ===")
        print("=> Training_Econding: Num of items", len(train_encodings.encodings))
        self.count_words(train_texts)
        self.count_tokens(train_encodings)

        print("=> Test_Econding: Num of items", len(test_encodings.encodings))
        self.count_words(test_texts)
        self.count_tokens(test_encodings)

        print("=> Val_Econding: Num of items", len(val_encodings.encodings))
        self.count_words(val_texts)
        self.count_tokens(val_encodings)

        train_dataset = AITADataset(train_encodings, train_labels)
        val_dataset = AITADataset(val_encodings, val_labels)
        test_dataset = AITADataset(test_encodings, test_labels)

        return train_dataset, test_dataset, val_dataset,

    def load_pretrained_model(self, directory_path):
        """
        Eigene Methode um das vortrainierte Modell zu laden, hier DistilBERT. Lädt entweder aus lokaler Datei oder von HuggingFace in den Cache.
        :param directory_path:
        :return:
        """
        if os.path.exists(directory_path) == True:
            print("Lade aus lokaler Datei")
            model = DistilBertForSequenceClassification.from_pretrained(directory_path)
        else:
            print("Lade Model von HuggingFace")
            model = DistilBertForSequenceClassification.from_pretrained(self.model_name)
            model.save_pretrained(directory_path)
        return model

    def train_model(self, model, train_dataset, val_dataset):
        """
         Trainingsvorgang des Modells.
        Für Details zu den Trainingarguments und genauer Funktionsweise des Trainer siehe. https://huggingface.co/transformers/main_classes/trainer.html

        :rtype: object
        """
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
        trainer.evaluate(eval_dataset=val_dataset)
        trainer.save_model(output_dir=self.target_directory_models)
        print("=== Training finished ===")



    def predict(self, test_dataset, model):
        """
        Lässt das Model Vorhersagen machen mithilfe der Trainer.predict()-Methode.
        :param test_dataset: Test-Datenset zudem die vorhersagen gemacht werden.
        :param model: das finegetunte, d.h. trainierte DistilBert-Modell
        :return predictions: predictions-Objekt des Modells.
        """
        print("=== Starte Prediction ===")
        trainer = Trainer(
            model=model,
        )
        predictions = trainer.predict(test_dataset)

        return predictions

    def analyse_predictions(self, predictions, test_dataset):
        """
        Analysiere zuvor errechnete Vorhersagen. Besonders interessant: predictions.predictions gibt Vorhersagen inform eines np.arrays
        für jeden Datenpunkt inform von Logits zurück. Die Logit-Werte werden mit argmax() in absolute Werte (d.h. das Label 0 oder 1)
        und mit Softmax in die Wahrscheinlichkeiten für jedes Label gewandelt. Zudem wird durch einen Abgleich geschaut, welche Label berechnet wurde.

        Die Ergebnisse werden aus einem Pandas Dateframe inform einer JSON-Datei gespeichert.
        :param predictions:
        :param test_dataset:
        """
        result_frame = pd.DataFrame()
        label = ""
        tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)


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
                'Predicted_Label_Softmax':str(round(float(probabilities.data[0]), 2)) + " (NTA), "+  str(round(float(probabilities.data[1]), 2)) + " (YTA) "  + " - Summe: " + str(int(sum(probabilities))),
                'Predicted_Label_Softmax_Converted': str(round(float(converted_probabilities.data[0]), 2)) + " (NTA), " + str(round(float(converted_probabilities.data[1]), 2))+  " (YTA) " + " - Summe: " + str(round(float(sum(converted_probabilities)))),
                'Predicted_Label': label,
                }, ignore_index=True)
            label = ""
            x = ''

        print("=== Speicher Ergebnisse ===")

        result = result_frame.to_json(orient="records")
        parsed = json.loads(result)
        ProcessingBase().save_json_to_file(parsed, self.target_directory_results, "LargeSizedPredictions.json")


    def compute_metrics(self, pred):
        """
        Berechnet die Metriken zu den Predictions bzw. Vorhersagen des finegetunten Models.

        Basierend auf sklearn Metriken:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html
        Die Metriken zum Predictions-Vorgang (z.B. Loss) und die Performance-Metriken werden abschließend ebenfalls in JSON-Dateien gespeichert.
        """

        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        process_metric = pred.metrics

        metrics = {
            'prediction_process_metrics': process_metric,
            'prediction_result_metrics': {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        }

        ProcessingBase().save_json_to_file(metrics, self.target_directory_results,
                                           "LargeSizedMetrics.json")

    def count_tokens(self, encoding):
        """
        Ermittelt die durschnittliche Länge der encodeten Posts im Datenset. Basierend auf https://www.youtube.com/watch?v=_eSGWNqKeeY&t=1368s (17:30f)
        :param encoding:
        """
        lengths = []
        token_length = 0
        for idx, encoded_sent in enumerate(encoding.encodings):
            tokens = encoding.encodings.__getitem__(idx).tokens
            token_length = len(tokens)
            lengths.append(token_length)
            token_length = 0

        print("Token min length", min(lengths))
        print("Token max length", max(lengths))
        print("Token average length", np.median(lengths))

    def confusion_matrix(self, prediction):
        """
        Visualisiert die Ergebnisse der Predictions als Confusion Matrix.
        Basierend auf sklearn: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html#sklearn.metrics.ConfusionMatrixDisplay.from_predictions

        :param prediction: Ergebnisse aus trainer.predict()
        """

        true_labels = prediction.label_ids
        preds = prediction.predictions.argmax(-1)

        confusion_matrix_base =  confusion_matrix(labels=[0, 1], y_true=true_labels, y_pred=preds)

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_base)
        disp.plot()
        plt.title("Ergebnisse der Vorhersagen für Test-Datenset")
        plt.show()

    def count_words(self, texts):
        print("Dataset Anzahl an Texten", len(texts))
        lengths = []
        num_words = 0

        for text in texts:
            num_words = len(text.split())
            lengths.append(num_words)
            num_words = 0

        print("Text min length", min(lengths))
        print("Text max length", max(lengths))
        print("Text average length", np.median(lengths))



    def run_pipeline(self):
        """
        __main__ methode für dieses Skript. Hier werden alle nötigen Funktionen vom Instanzieren bis zur Auswertung des Models aufgerufen.
        """
        # Lade Model
        model_path = os.path.join("Models", self.model_name)
        model = self.load_pretrained_model(model_path)
        # Lade Datensets
        train_ds, test_ds, val_ds = self.load_datasets()
        train_texts, train_labels, test_texts, test_labels, val_texts, val_labels = self.split_dataset_attributes(
            train_ds, test_ds, val_ds)

        train_dataset, val_dataset, test_dataset = self.preprocessing(train_texts, train_labels, test_texts,
                                                                      test_labels, val_texts, val_labels)

        # Training bzw. Finetuning des Modells (hier: DistilBERT)
        #self.train_model(model, train_dataset, val_dataset)

        trained_model = DistilBertForSequenceClassification.from_pretrained(self.target_directory_models)

        # Predictions bzw. Vorhersagen des fertig trainierten Models
        predictions = self.predict(test_dataset, trained_model)
        # Analyse der Vorgersagen
        self.analyse_predictions(predictions, test_dataset)
        self.compute_metrics(predictions)
        self.confusion_matrix(predictions)

