from data_collect import DataCollector
from preprocess import Preprocessor
from train import GPT2Trainer
from bert import BERTTrainer

class GPyT:
    def __init__(self):
        self.data_collector = DataCollector()
        self.preprocessor = Preprocessor()
        self.trainer = GPT2Trainer() # to initialize a GPT-2 model 
        # self.trainer = BERTTrainer() # to initialize a BERT model

    def run(self):
        # Step 1: Data Collection
        self.data_collector.collect_and_clean_repos()

        # Step 2: Preprocessing
        self.preprocessor.process_repos()

        # Step 3: Training
        self.trainer.train()

if __name__ == "__main__":
    processor = GPyT()
    processor.run()
