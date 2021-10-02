import json
import math
import ntpath
import os

import pandas as pd
from DataScripts.DataProcessing.ProcessingBase import ProcessingBase


class DataSet():
    def __init__(self):
        self.dataframe = pd.DataFrame()

    def create_labels(self) -> pd.DataFrame:
        """
        Iteriert durch das Rohdatenset und erstellt einen DataFrame mit den Posts und den dazugehörigen Labels.
        :return: Dataframe mit Labels und Texten.
        """
        print("=== Erstelle Datenset ===")

        aita_data = ProcessingBase().load_datafile("DataFiles", "posts_json_large.json")
        print("AITA Rohdaten - Anzahl der Datenpunkte", len(aita_data))

        dataset_dataframe = pd.DataFrame()

        nta_count = 0  # not the asshole
        nah_count = 0  # no assholes here
        yta_count = 0  # you're the asshole
        esh_count = 0  # everybody sucks here
        neutral_count = 0  # no rating in comment
        total_count_postings = 0
        official_rating = {}
        label = 0

        for batch_idx, post_batch in enumerate(aita_data):

            for idx, post in enumerate(post_batch):
                verdict = post["verdict"]
                content = post["content"]
                if verdict != None and content != "[removed]" and content != "[deleted]": # hier werden Datenpunkte ohne Relevanz aussortiert
                    if verdict == "Not the A-hole":
                        nta_count += 1
                        label = int(0)
                    elif verdict == "Asshole":
                        yta_count += 1
                        label = int(1)
                    elif verdict == "Everyone Sucks":
                        esh_count += 1
                        label = int(2)
                    elif verdict == "No A-holes here":
                        nah_count += 1
                    else:
                        neutral_count += 1
                        label = int(3)
                    dataset_dataframe = dataset_dataframe.append({
                        "label": label,
                        "text": content
                    }, ignore_index=True)
                    label = 0
                    content = ''
                else:
                    continue

        print("total rating", len(aita_data), nta_count, yta_count, esh_count,
              neutral_count, nah_count)
        official_rating['total_postings'] = total_count_postings
        official_rating['total_nta'] = nta_count
        official_rating['total_yta'] = yta_count
        official_rating['total_esh'] = esh_count
        official_rating['total_nah'] = nah_count
        official_rating['total_neutral'] = neutral_count
        # Konvertiere Label in int - sonst Probleme mit DistilBERT
        dataset_dataframe = dataset_dataframe.astype({'label': int, 'text': str})
        return dataset_dataframe

    def sort_dataframe(self, dataframe: pd.DataFrame):
        """
        Sortiert die DataFrame Zeilen nach Spalten, in diesem Fall nach den Werten in der Spalte "Label".
        :param dataframe: Zu bearbeitender DataFrame
        """
        dataframe = dataframe.sort_values(dataframe.columns[0], ascending=False)
        self.dataframe = dataframe

    def create_dataset(self):
        """
        Erstelle Datensets aus dem zuvor erstellten und sortierten Dataframe. Hier werden alle Datenpunkte mit NTA oder YTA Label bearbeitet für die binäre Klassizifierzung mit DistilBERT.
        Ziel ist es, Datensets mit gleicher Anzahl an Datenpunkten bereit zu stellen, da weniger YTA-Daten als NTA-Daten vorhanden sind.
        """
        # Erstelle Sub-Dataframe mit YTA und NTA Daten.
        grouped = self.dataframe.groupby(self.dataframe.columns[0])
        group_nta = grouped.get_group(0)  # NTA
        group_yta = grouped.get_group(1)

        # Ermittle Größe für die Datensets
        print("len_label_yta", len(group_yta.index))
        print("len_label_nta before", len(group_nta.index))
        yta_size = len(group_yta.index)
        group_nta = group_nta.reset_index()
        group_yta = group_yta.reset_index()

        for idx, element in group_nta.iterrows():
            print("index", idx)
            if idx >= yta_size:
                print("Delete row")
                group_nta = group_nta.drop(idx)

        print("len_label_nta Final", len(group_nta.index))
        print("len_label_nya Final", len(group_yta.index))

        nta_train, nta_test, nta_val = self.sort_into_dataset(group_nta)

        yta_train, yta_test, yta_val = self.sort_into_dataset(group_yta)

        joined_train = nta_train + yta_train
        joined_test = nta_test + yta_test
        joined_val = nta_val + yta_val

        print("joined", len(joined_train), len(joined_test), len(joined_val))
        train_dataframe = {
            'data': joined_train
        }
        test_dataframe = {
            'data': joined_test
        }
        val_dataframe = {
            'data': joined_val
        }

       # Speicher in Dateien
        with open(ntpath.join("DataFiles/DataSets/DataSet-L/", 'dataset_l_train.json'), 'w') as fp:
            json.dump(train_dataframe, fp, sort_keys=True, indent=4)

        with open(ntpath.join("DataFiles/DataSets/DataSet-L/", 'dataset_l_test.json'), 'w') as fp:
            json.dump(test_dataframe, fp, sort_keys=True, indent=4)

        with open(ntpath.join("DataFiles/DataSets/DataSet-L/", 'dataset_l_val.json'), 'w') as fp:
            json.dump(val_dataframe, fp, sort_keys=True, indent=4)

    def sort_into_dataset(self, category_dataset):
        """
        Sortiere die Datenpunkte in das jeweilige Sub-Datenset. Hier werden für entweder NTA oder YTA jeweils train_data, test_data, validation_data erstellt.
        :param category_dataset: Die Kategorie für die die Datensets erstellt werden sollen, daher entweder NTA oder YTA.
        :return: train_data, test_data, validation_data - fertigen Datensets.
        """
        # Bestimme Länge der train, test, val Datsets

        train_set_length = len(category_dataset.index) / 100 * 80
        train_set_length = int(math.ceil(train_set_length))

        test_set_length = len(category_dataset.index) / 100 * 10
        test_set_length = int(math.ceil(test_set_length))

        validation_set_length = len(category_dataset.index) / 100 * 10
        validation_set_length = int(math.ceil(validation_set_length))

        train_data = []
        tmp_dict = {}
        test_data = []
        test_dict = {}
        validation_data = []
        val_dict = {}
        stop_num = train_set_length - 1
        stop_num_2 = train_set_length + test_set_length - 1

        # Einfügen in die Datensets
        for idx, point in category_dataset.iterrows():
            print("index new", idx)
            if idx <= train_set_length - 1:
                print("fill train data")
                tmp_dict['label'] = point['label']
                tmp_dict['text'] = point['text']
                train_data.append(tmp_dict)
                tmp_dict = {}
            if idx > stop_num and idx <= stop_num_2:
                print("fill test data")
                test_dict['label'] = point['label']
                test_dict['text'] = point['text']
                test_data.append(test_dict)
                test_dict = {}
            if idx >= stop_num_2:
                print("fill validation data")
                val_dict['label'] = point['label']
                val_dict['text'] = point['text']
                validation_data.append(val_dict)
                val_dict = {}

        print("train data,", train_data)
        print("train data length", len(train_data))

        print("test data,", test_data)
        print("test data length", len(test_data))

        print("val data,", validation_data)
        print("val data length", len(validation_data))

        return train_data, test_data, validation_data
