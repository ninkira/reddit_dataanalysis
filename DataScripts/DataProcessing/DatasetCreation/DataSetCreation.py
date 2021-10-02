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
        print("create Dataset starts")

        aita_data = ProcessingBase().load_datafile("DataFiles", "posts_json_large.json")
        print("aita_data", aita_data)

        dataset_dataframe = pd.DataFrame()

        nta_count = 0  # not the asshole
        nah_count = 0  # no assholes here
        yta_count = 0  # you're the asshole
        esh_count = 0  # everybody sucks here
        neutral_count = 0  # no rating in comment
        total_count_postings = 0
        official_rating = {}
        label = 0
        print("aita_data length", len(aita_data))
        for batch_idx, post_batch in enumerate(aita_data):


            for idx, post in enumerate(post_batch):
                print("idx, post", idx)
                verdict = post["verdict"]
                print("verdict", verdict)
                content = post["content"]
                if verdict != None and content != "[removed]" and content != "[deleted]":
                    print("verdict", verdict)
                    if verdict == "Not the A-hole":
                        nta_count += 1
                        label = int(0)
                        print("official nta", nta_count)
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
                    print("label", label)
                    dataset_dataframe = dataset_dataframe.append({
                        "label": label,
                        "text": content
                    }, ignore_index=True)
                    print(dataset_dataframe)
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
        print("official rating", official_rating)

        # convert labels to int - otherwise problems later on with BERT
        # source https://www.linkedin.com/pulse/change-data-type-columns-pandas-mohit-sharma/
        dataset_dataframe = dataset_dataframe.astype({'label': int, 'text': str})
        return dataset_dataframe

    def sort_dataframe(self, dataframe: pd.DataFrame):
        # https://stackoverflow.com/questions/40462675/sort-dataframe-by-first-column-pandas
        dataframe = dataframe.sort_values(dataframe.columns[0], ascending=False)
        print("sorted", dataframe)
        self.dataframe = dataframe

    def create_dataset(self):
        print("Test test", self.dataframe)
        # get yta and nta
        # https://www.kite.com/python/answers/how-to-split-a-pandas-dataframe-into-multiple-dataframes-by-column-value-in-python#:~:text=Splitting%20a%20pandas%20Dataframe%20into,with%20the%20Dataframe%20was%20found.
        grouped = self.dataframe.groupby(self.dataframe.columns[0])
        print("grouped", grouped)
        group_nta = grouped.get_group(0)  # NTA
        group_yta = grouped.get_group(1)

        # create equal length
        print("label_nta", group_nta)
        print("len_label_yta", len(group_yta.index))
        print("len_label_nta before", len(group_nta.index))
        yta_size = len(group_yta.index)
        # https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
        print("label_yta", group_yta)
        group_nta = group_nta.reset_index()
        group_yta = group_yta.reset_index()
        # https://pandas.pydata.org/pandas-docs/dev/reference/api/pandas.DataFrame.reset_index.html
        for idx, element in group_nta.iterrows():
            print("index", idx)
            if idx >= yta_size:
                print("Delete row")
                # https://www.google.com/search?q=remove+rows+from+dataframe&client=opera-gx&hs=UQ4&sxsrf=AOaemvKIzwsninH46-zdimM0bDf9SFl7hA%3A1630960593753&ei=0Xs2Ycq4LaaVxc8Pgua9kAE&oq=remove+rows&gs_lcp=Cgdnd3Mtd2l6EAMYAjIFCAAQywEyBQgAEMsBMgoIABCABBCHAhAUMgUIABDLATIFCAAQywEyBQgAEIAEMgUIABDLATIFCAAQgAQyBQgAEMsBMgUIABDLAToHCCMQsAMQJzoHCAAQRxCwA0oECEEYAFDMWVjeXmDHcGgBcAJ4AIABXogB5AKSAQE0mAEAoAEByAEJwAEB&sclient=gws-wiz
                group_nta = group_nta.drop(idx)

        print("len_label_nta final", len(group_nta.index))
        print("len_label_nya final", len(group_yta.index))

        nta_train, nta_test, nta_val = self.sort_into_dataset(group_nta)
        # print("nta_train", len(nta_train))
        # print("nta_test",len(nta_test))
        # print("nta_val", len(nta_val))

        yta_train, yta_test, yta_val = self.sort_into_dataset(group_yta)
        print("yta_train", len(yta_train))
        print("yta_test", len(yta_test))
        print("yta_val", len(yta_val))

        joined_train = nta_train + yta_train
        joined_test = nta_test + yta_test
        joined_val = nta_val + yta_val

        print("joined", len(joined_train), len(joined_test), len(joined_val))

        # final_dataframe = pd.DataFrame()
        # https://stackoverflow.com/questions/36526282/append-multiple-pandas-data-frames-at-once

        train_dataframe = {
            'data': joined_train
        }

        test_dataframe = {
            'data': joined_test
        }
        val_dataframe = {
            'data': joined_val
        }

        print("final dataframe", type(train_dataframe))

        with open(ntpath.join("DataFiles/DataSets/DataSet-L/", 'dataset_l_train.json'), 'w') as fp:
            json.dump(train_dataframe, fp, sort_keys=True, indent=4)

        with open(ntpath.join("DataFiles/DataSets/DataSet-L/", 'dataset_l_test.json'), 'w') as fp:
            json.dump(test_dataframe, fp, sort_keys=True, indent=4)

        with open(ntpath.join("DataFiles/DataSets/DataSet-L/", 'dataset_l_val.json'), 'w') as fp:
            json.dump(val_dataframe, fp, sort_keys=True, indent=4)



    def sort_into_dataset(self, category_dataset):

        # get length for train, test, val
        print("final dataframe length", len(category_dataset.index))
        train_set_length = len(category_dataset.index) / 100 * 80
        # runden in python https://qastack.com.de/programming/2356501/how-do-you-round-up-a-number-in-python
        train_set_length = int(math.ceil(train_set_length))

        print("train set length", train_set_length)
        test_set_length = len(category_dataset.index) / 100 * 10
        test_set_length = int(math.ceil(test_set_length))
        print("test set length", test_set_length)

        validation_set_length = len(category_dataset.index) / 100 * 10
        validation_set_length = int(math.ceil(validation_set_length))
        print("validation set length", validation_set_length)

        train_data = []
        tmp_dict = {}
        test_data = []
        test_dict = {}
        validation_data = []
        val_dict = {}
        stop_num = train_set_length - 1
        print("stop", stop_num)
        stop_num_2 = train_set_length + test_set_length - 1
        print("stop2", stop_num_2)

        for idx, point in category_dataset.iterrows():
            print("index new", idx)
            if idx <= train_set_length - 1:
                print("fill train data")
                tmp_dict['label'] = point['label']
                tmp_dict['text'] = point['text']
                train_data.append(tmp_dict)
                tmp_dict =  {}
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
