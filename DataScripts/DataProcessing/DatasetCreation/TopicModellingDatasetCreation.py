import json

import ntpath

import pandas as pd
from DataScripts.DataProcessing.ProcessingBase import ProcessingBase


class TopicModellingDataset:

    def __init__(self):
        print("TopicModelling Dataset Creation")

    def create_labels(self, base_data) -> pd.DataFrame:
        """
        Erstellt aus dem "verdict"- und "content-"Attributen des Basedatenset einen Pandas Dataframe mit den Labels.
        :param base_data: Zu verarbeitendes Datenset.
        :return: Pandas Dateframe mit Post-Texten und dem dazugehörigen Label
        """
        dataset_dataframe = pd.DataFrame()
        nta_count = 0  # not the asshole
        nah_count = 0  # no assholes here
        yta_count = 0  # you're the asshole
        esh_count = 0  # everybody sucks here
        neutral_count = 0  # no rating in comment
        total_count_postings = 0
        official_rating = {}
        label = 0
        for post in base_data:
            total_count_postings += 1
            verdict = post["verdict"]
            content = post["content"]
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
            dataset_dataframe = dataset_dataframe.append({
                "label": label,
                "content": content
            }, ignore_index=True)
            label = 0
        print("official Verdicts (Total/NTA/YTA):", total_count_postings, nta_count, yta_count)
        official_rating['total_postings'] = total_count_postings
        official_rating['total_nta'] = nta_count
        official_rating['total_yta'] = yta_count
        official_rating['total_esh'] = esh_count
        official_rating['total_nah'] = nah_count
        official_rating['total_neutral'] = neutral_count
        dataset_dataframe = dataset_dataframe.astype({'label': int, 'content': str})
        return dataset_dataframe

    def sort_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Sortiert den Dataframe nach den Labels. Source: https://stackoverflow.com/questions/40462675/sort-dataframe-by-first-column-pandas
        :rtype: dataframe: Sortierter Dataframe
        """
        dataframe = dataframe.sort_values(dataframe.columns[0], ascending=False)
        return dataframe

    def create_datasets(self, base_data):
        """
        Erstellt aus dem Base-Datenset zwei Subdatensets für je die NTA- und YTA-Kategorie.
        :param base_data: das zu bearbeitende Datenset, z.B. das Top-Datenset.
        """
        # Preprocessing - Erstelle Dataframe mit den Texten und Labels und sortiere ihn anschließend.
        label_df = self.create_labels(base_data)
        sorted_label_df = self.sort_dataframe(label_df)

        # Hole alle Texte mit NTA oder YTA Label
        # https://www.kite.com/python/answers/how-to-split-a-pandas-dataframe-into-multiple-dataframes-by-column-value-in-python#:~:text=Splitting%20a%20pandas%20Dataframe%20into,with%20the%20Dataframe%20was%20found.
        grouped = sorted_label_df.groupby(sorted_label_df.columns[0])
        group_nta = grouped.get_group(0)
        group_yta = grouped.get_group(1)
        print("group", group_yta)

        print("Anzahl an YTA Datenpunkten:", len(group_yta.index))
        print("Anzahl an NTA Datenpunkten:", len(group_nta.index))

        # Resette Index um Verfälschungen zu vermeiden
        group_nta = group_nta.reset_index()
        group_yta = group_yta.reset_index()

        # Konvertiere in JSON und speicher in Verzeichnis
        nta_data = group_nta.to_json(orient="records")
        nta_data = json.loads(nta_data)
        yta_data = group_yta.to_json(orient="records")
        yta_data = json.loads(yta_data)


        with open(ntpath.join("DataFiles/TopicModellingData", 'tm_nta_data.json'), 'w') as fp:
            json.dump(nta_data, fp, indent=4)

        with open(ntpath.join("DataFiles/TopicModellingData", 'tm_yta_data.json'), 'w') as fp:
            json.dump(yta_data, fp, indent=4)
