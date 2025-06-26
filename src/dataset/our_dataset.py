from __future__ import annotations

import numpy as np
import torch
import ast
import os
import random
from os.path import join as join_path
from typing import Iterable, Optional
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from dataset.custom_types import Language, is_in_distribution, combine_distributions
from dataset.dataset import Dataset
import utils


class OurDataset(Dataset):            
    def __init__(
        self,
        csv_dirpath: str,
        split: str = None,
        crosslingual: bool = False,
        fact_check_fields: Iterable[str] = ('claim', ),
        fact_check_language: Optional[Language] = None,
        language: Optional[Language] = "eng",
        post_language: Optional[Language] = None,
        initial_cleaning: bool = False,
        save_dirpath: str = None,
        text_version: str = 'english',
        ocr_version: str = 'english',
        use_original_split: bool = False, # Added parameter to choose split method NEW!
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        
        assert all(field in ('claim', 'title') for field in fact_check_fields)
        assert split in (None, 'test', 'train')
        assert text_version in ('english', 'original')
        assert ocr_version in ('english', 'original')
        
        self.csv_dirpath = csv_dirpath
        self.crosslingual = crosslingual
        self.fact_check_fields = fact_check_fields
        self.fact_check_language = fact_check_language
        self.language = language
        self.post_language = post_language
        self.split = split
        self.text_version = text_version
        self.ocr_version = ocr_version
        self.save_dirpath = save_dirpath

        if initial_cleaning:
            self.perform_initial_cleaning()
        else:
            self.load()

    def _load_csv(self):
        posts_path = join_path(self.csv_dirpath, 'posts.csv')
        fact_checks_path = join_path(self.csv_dirpath, 'fact_checks.csv')
        fact_check_post_mapping_path = join_path(self.csv_dirpath, 'fact_check_post_mapping.csv')

        for path in [posts_path, fact_checks_path, fact_check_post_mapping_path]:
            assert os.path.isfile(path)

        parse_col = lambda s: ast.literal_eval(s.replace('\n', '\\n')) if s else s
        
        all_df_fact_checks = pd.read_csv(fact_checks_path).fillna('').set_index('fact_check_id')
        for col in ['claim', 'instances', 'title']:
            all_df_fact_checks[col] = all_df_fact_checks[col].apply(parse_col)
            
        all_df_posts = pd.read_csv(posts_path).fillna('').set_index('post_id')
        for col in ['instances', 'ocr', 'verdicts', 'text']:
            all_df_posts[col] = all_df_posts[col].apply(parse_col)

        mapping = pd.read_csv(fact_check_post_mapping_path)[["fact_check_id", "post_id"]]
        
        return all_df_fact_checks, all_df_posts, mapping

    def perform_initial_cleaning(self) -> None:
        all_df_fact_checks, all_df_posts, _map = self._load_csv()

        all_df_posts["content"] = self.build_post_vocab(all_df_posts).values()
        all_df_posts = all_df_posts[all_df_posts["content"] != ""]

        grouped_posts = all_df_posts.groupby(by="content")
        content_value_counts = all_df_posts["content"].value_counts()
        unique_contents = content_value_counts[content_value_counts > 1].index.values

        for uniq_cont in tqdm(unique_contents):
            # reflect changes in the post-claim mapping
            post_ids = grouped_posts.get_group(uniq_cont).index.values
            _map.loc[_map["post_id"].isin(post_ids), "post_id"] = post_ids[0]
            all_df_posts = all_df_posts.drop(index=post_ids[1:])
        _map = _map.drop_duplicates()

        ####### cleaning fact checks

        all_df_fact_checks["content"] = self.build_fact_check_vocab(all_df_fact_checks).values()
        all_df_fact_checks = all_df_fact_checks[all_df_fact_checks["content"] != ""]

        grouped_claims = all_df_fact_checks.groupby(by="content")
        content_value_counts = all_df_fact_checks["content"].value_counts()
        unique_contents = content_value_counts[content_value_counts > 1].index.values

        for uniq_cont in tqdm(unique_contents):
            # reflect changes in the post-claim mapping
            claim_ids = grouped_claims.get_group(uniq_cont).index.values
            _map.loc[_map["fact_check_id"].isin(claim_ids), "fact_check_id"] = claim_ids[0]
            all_df_fact_checks = all_df_fact_checks.drop(index=claim_ids[1:])
        _map = _map.drop_duplicates()
        
        ####### update mapping

        _map = _map[
            (_map["fact_check_id"].isin(all_df_fact_checks.index)) &
            (_map["post_id"].isin(all_df_posts.index))
        ]
        
        all_df_posts = all_df_posts[all_df_posts.index.isin(_map["post_id"])]
        all_df_fact_check_post_mapping = _map

        os.makedirs(self.save_dirpath, exist_ok=True)
        all_df_posts.to_csv(join_path(self.save_dirpath, "posts.csv"))
        all_df_fact_checks.to_csv(join_path(self.save_dirpath, "fact_checks.csv"))
        all_df_fact_check_post_mapping.to_csv(join_path(self.save_dirpath, 'fact_check_post_mapping.csv'))
        
     
    def load(self) -> None:
        self.all_df_fact_checks, self.all_df_posts, self.all_df_fact_check_post_mapping = self._load_csv()
        self.df_fact_checks, self.df_posts = self.filter_data()

        fc_ids = [_map[0] for _map in self.fact_check_post_mapping]
        post_ids = [_map[1] for _map in self.fact_check_post_mapping]
    
        self.df_fact_checks_w_existing_posts = self.df_fact_checks[
            self.df_fact_checks.index.isin(fc_ids)
        ]
        self.df_posts_w_existing_fact_checks = self.df_posts[
            self.df_posts.index.isin(post_ids)
        ]
        
        self.id_to_fact_check_vocab = {
            id: content 
            for id, content 
            in zip(self.df_fact_checks.index, self.df_fact_checks["content"])
        }
        self.id_to_post_vocab = {
            id: content 
            for id, content 
            in zip(self.df_posts.index, self.df_posts["content"])
        }

    def filter_data(self) -> tuple[pd.DataFrame]:
        df_posts = self.all_df_posts.copy()
        df_fact_checks = self.all_df_fact_checks.copy()
        df_fact_check_post_mapping = self.all_df_fact_check_post_mapping.copy()
  
        # Filter fact-checks by the language detected in claim
        if self.language or self.fact_check_language:
            df_fact_checks = df_fact_checks[df_fact_checks['claim'].apply(
                lambda claim: is_in_distribution(self.language or self.fact_check_language, claim[2])  # claim[2] is the language distribution 
            )]
            
        # Filter posts by the language detected in the combined distribution.
        # There was a slight bug in the paper version of post language filtering and in effect we have slightly more posts per language
        # in the paper. The original version did not take into account that the sum of percentages in a distribution does not have to be equal to 1.
        if self.language or self.post_language:
            def post_language_filter(row):
                texts = [
                    text
                    for text in [row['text']] + row['ocr']
                    if text  # Filter empty texts
                ]
                distribution = combine_distributions(texts)
                return is_in_distribution(self.language or self.post_language, distribution)
                
            df_posts = df_posts[df_posts.apply(post_language_filter, axis=1)]

        df_fact_check_post_mapping = df_fact_check_post_mapping[
            (df_fact_check_post_mapping["fact_check_id"].isin(df_fact_checks.index)) &
            (df_fact_check_post_mapping["post_id"].isin(df_posts.index))
        ]   

        # choose a specific data split, a subset
        if self.split:
            if self.use_original_split: ### NEW because of added original split
                # Use the original split method from the second file
                split_post_ids = set(self.split_post_ids(self.split))
                df_posts = df_posts[df_posts.index.isin(split_post_ids)]
                
                # Update the mapping to reflect the filtered posts
                df_fact_check_post_mapping = df_fact_check_post_mapping[
                    df_fact_check_post_mapping["post_id"].isin(df_posts.index)
                ]
            else:    
                df_fact_check_post_mapping = self.split_data_based_on_claims(
                    df_fact_check_post_mapping, 
                    split_name=self.split
                )
                post_ids = df_fact_check_post_mapping["post_id"].unique()
                df_posts = df_posts.loc[post_ids]

            # SANITY CHECK
            df_fact_check_post_mapping = df_fact_check_post_mapping[
                (df_fact_check_post_mapping["fact_check_id"].isin(df_fact_checks.index)) &
                (df_fact_check_post_mapping["post_id"].isin(df_posts.index))
            ]   

        # Create mapping variable
        post_ids = set(df_posts.index)
        fact_check_ids = set(df_fact_checks.index)
        fact_check_post_mapping = set(
            (fact_check_id, post_id)
            for fact_check_id, post_id in df_fact_check_post_mapping.itertuples(index=False, name=None)
            if fact_check_id in fact_check_ids and post_id in post_ids
        )
        self.fact_check_post_mapping = list(fact_check_post_mapping)
        return df_fact_checks, df_posts

    def build_fact_check_vocab(self, df_fact_checks: pd.DataFrame) -> dict:    
        return {
            fact_check_id: Dataset().clean_text(claim[self.text_version == 'english'])
            for fact_check_id, claim in zip(df_fact_checks.index, df_fact_checks['claim'])
        }
    
    def build_post_vocab(self, df_posts: pd.DataFrame) -> dict:
        id_to_post_vocab = dict()
        for post_id, post_text, ocr in zip(df_posts.index, df_posts['text'], df_posts['ocr']):
            texts = list()
            if post_text:
                texts.append(Dataset().clean_text(post_text[self.text_version == 'english']))
            for ocr_text in ocr:
                texts.append(Dataset().clean_text(ocr_text[self.ocr_version == 'english']))
            id_to_post_vocab[post_id] = ' '.join(texts)

        return id_to_post_vocab
        
    def split_data_based_on_claims(
            self, df_mapping: pd.DataFrame, split_name: str, 
            train_size: float = 0.8, rng_seed: int = 0
    ) -> tuple[list[int], pd.DataFrame]:
        unique_fc_ids = df_mapping["fact_check_id"].unique().tolist()
        rnd = random.Random(rng_seed) 
        rnd.shuffle(unique_fc_ids)

        groups_sizes = (
            df_mapping
                .groupby(by="fact_check_id")
                .size()[unique_fc_ids]
                .values
        )
        offsets = np.cumsum(groups_sizes)
            
        # find a train/test split
        offsets = np.array(offsets)
        max_index = np.where(offsets / len(df_mapping) < train_size)[0].max()
        
        train_fc_ids = unique_fc_ids[: max_index + 1]
        test_fc_ids = unique_fc_ids[max_index + 1:]
        fc_ids = train_fc_ids if split_name == "train" else test_fc_ids

        df_mapping = df_mapping[(df_mapping["fact_check_id"].isin(fc_ids))]
        return df_mapping
    
    @staticmethod ### NEW based on original multiclaim split
    def split_post_ids(split):
        """
        Returns post IDs for a specific split based on the original method.
        This split only works for the particular dataset version with 28092 posts.
        
        Args:
            split (str): One of 'train', 'dev', or 'test'
            
        Returns:
            list: Post IDs for the specified split
        """
        assert split in ('train', 'dev', 'test'), "Split must be one of 'train', 'dev', or 'test'"
        
        rnd = random.Random(1)  # Fixed seed for reproducibility
        post_ids = list(range(28092))  # This split only works for the particular dataset version with this number of posts
        rnd.shuffle(post_ids)
        
        train_size = 0.8
        dev_size = 0.1
        
        train_end = int(len(post_ids) * train_size)
        dev_end = train_end + int(len(post_ids) * dev_size)
        
        if split == 'train':
            return post_ids[:train_end]
        elif split == 'dev':
            return post_ids[train_end:dev_end]
        else:  # test
            return post_ids[dev_end:]

    def compute_embeddings(self, model: SentenceTransformer, 
                           savepath: str = "./data/embeddings/temp", 
                           compute_claims: bool = True, 
                           loader_kwargs: dict = None,
                           verbose: bool = True
    ) -> None:
        """Compute the embeddings of the entire database (either all posts or all claims)
        
        We want to have the embeddings precomputed for quick comparison of all posts to a specific claim 
        """
        os.makedirs(savepath, exist_ok=True)    
        df_database = self.df_fact_checks if compute_claims else self.df_posts
        dataset = list(zip(df_database.index.values, df_database["content"].values))
    
        if loader_kwargs is None:
            loader_kwargs = {
                "batch_size": 64,
                "num_workers": 1
            }
        loader = DataLoader(dataset, **loader_kwargs)
    
        if verbose:
            print(f"Computing embeddings for {'claims' if compute_claims else 'posts'}")
        for batch in tqdm(loader, disable=verbose is False):
            ids, texts = batch
    
            with torch.no_grad():
                embeddings = model.encode(
                    texts,
                    batch_size = len(texts),
                    device=utils.get_device()
                )
            for id, emb in zip(ids, embeddings):
                full_path = os.path.join(savepath, f"{id}.npy")
                np.save(full_path, emb)
    def clone(self):
        import copy
        return copy.deepcopy(self)


if __name__ == "__main__":
    ds = OurDataset(split="train", csv_dirpath="./data", language="eng", version='original')