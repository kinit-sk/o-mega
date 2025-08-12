from datasets import load_dataset 
from dataset.dataset import Dataset
import pandas as pd

class HuggingfaceDataset(Dataset):
    def __init__(self, path, columns,task, **kwargs):
        """
        Args:
            path (str): Hugging Face dataset path (e.g., "username/dataset_name").
            columns (list): List mapping two columns, which will be used for cosine similarity task.
                        - If it is text_classification task list must be in this form: ['text_column', 'label_column']
            task (str): 'post_claim_matching' or 'text_classification'
            **kwargs: Additional arguments passed to the parent class.
        """
        super().__init__(**kwargs)
        self.path = path
        self.columns = columns
        tasks = ['post_claim_matching', 'text_classification']
        assert task in tasks, \
            f"Available tasks {tasks}"
        self.task=task
        self.split = 'Full'
        self.names_columns=[]
        self.dataset=None
        self.id_to_post_vocab= None
        self.id_to_fact_check_vocab=None
        self.fact_check_post_mapping = None  # Will store tuples of (claim_idx, post_idx)
        self.load()


    def load(self):
        self._load_and_process_dataset()
        self.column_names()
        self.rename_columns()
        self.define_tables()

    def column_names(self):
        if self.task == 'post_claim_matching':
            self.names_columns= ['post','claim']
        if self.task == 'text_classification':
            self.names_columns= ['text','label']
    
    def _load_and_process_dataset(self):
        """Load the dataset from Hugging Face and process columns."""
        dataset = load_dataset(self.path)
        available_splits = list(dataset.keys())
        if self.split == "Full":
            full_dataset = pd.concat(
                [dataset[split].to_pandas() for split in dataset.keys()],
                ignore_index=True
            )            
            self.dataset=full_dataset
        elif self.split in available_splits:
            self.dataset= dataset[self.split]
        else:
            raise ValueError(
                f"Invalid split: '{self.split}'. Available splits: {available_splits} or 'Full'")


    def rename_columns(self):
        self.dataset = self.dataset.rename(columns={self.columns[0]: self.names_columns[0], self.columns[1]: self.names_columns[1]})
        self.dataset = self.dataset.filter([self.names_columns[0],self.names_columns[1]])


    def define_tables(self):
        self.id_to_fact_check_vocab = self.dataset[self.names_columns[1]].to_dict()
        self.id_to_post_vocab = self.dataset[self.names_columns[0]].to_dict()
        self.fact_check_post_mapping = [[i, i] for i in range(len(self.dataset))]
 
    def __len__(self):
        """Returns the number of claim-post pairs."""
        return len(self.fact_check_post_mapping)
