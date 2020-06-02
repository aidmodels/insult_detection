from mlpm.solver import Solver
from deeppavlov import build_model
import json


class InsultDetectionSolver(Solver):
    def __init__(self, toml_file=None):
        super().__init__(toml_file)
        # Do you Init Work here
        with open('./insults_kaggle_conv_bert.json') as f:
            self.configs = json.load(f)
        
        self.model = build_model(self.configs)
        self.ready()
    def infer(self, data):
        # if you need to get file uploaded, get the path from input_file_path in data
        output = self.model([data['input']])
        result = {
            "is_insult": output[0]
        }
        return result # return a dict
