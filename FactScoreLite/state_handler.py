import json


class StateHandler:
    def __init__(self, path):
        self.db_path = path

    def save(self, data):
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=4)

    def load(self):
        try:
            with open(self.db_path, "r") as f:
                data = json.load(f)

            return data

        except FileNotFoundError:
            return []
