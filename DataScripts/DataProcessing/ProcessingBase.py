import json
import os


class ProcessingBase:

    def load_datafile(self, directory_path: str,  filename: str):
        complete_path = os.path.join(directory_path, filename)
        with open(complete_path) as json_file:
            data = json.load(json_file)
            # print("Data:", data)
            return data