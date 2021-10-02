import json
import os
from os import listdir
from os.path import isfile, join


class ProcessingBase:
    """Diese Klasse stellt einige Funktionen für den Basis Umgang mit JSON Dateien bereit, z.B. das Laden oder Speichern."""

    def load_datafile(self, directory_path: str,  filename: str):
        complete_path = os.path.join(directory_path, filename)
        with open(complete_path) as json_file:
            data = json.load(json_file)
            # print("Data:", data)
            return data

    def load_datafile_utf8(self, directory_path: str, filname: str) -> json:
        file = os.path.join(directory_path, filname)
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def save_json_to_file(self, json_object, directory_path: str, file_title: str):
        complete_path = os.path.join(directory_path, file_title)
        with open(complete_path, "w+") as outfile:
            json.dump(json_object, outfile)

    def connect_files(self):
        """
        Diese Methode fügt mehrere JSON-Dateien zusammen. Wurde aufgrund des Crawlings entwickelt - falls der Crawler
        abstürzt können mit dieser Methode die Crawler-Ergebnisse zusammengefügt werden.
        """
        my_path = "DataFiles/CrawlDirectory/Pushshift_Metadata"
        files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
        final_array = []
        for idx, file_name in enumerate(files):
            file_content = self.load_datafile(my_path, file_name)
            final_array.append(file_content)
        target_path = "DataFiles/"
        self.save_json_to_file(final_array, target_path, "posts_json_large.json")
        print("final_array", final_array)