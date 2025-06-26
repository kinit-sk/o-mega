import os
import re
from tqdm import tqdm
import json
import numpy as np
from transformers import PreTrainedTokenizer
from typing import Union
from datetime import datetime
import pandas as pd
from collections import defaultdict
import torch
from torch.nn.functional import normalize
import ast
import os
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from dataset import OurDataset
from functools import partial
from xai import GAE_Explain, ConservativeLRP, semantic_search_forward_function,Occlusion_word_level
from explain import STS_ExplainWrapper, ExplanationExecuter, compare_multiple_explanation_methods,SentenceTransformerToHF
import utils
from annotations import Annotations, TokenConversion
from captum._utils.models.linear_model import SkLearnLasso
from captum.attr._core.lime import get_exp_kernel_similarity_function
import optuna
from optuna.samplers import TPESampler, GridSampler,NSGAIISampler, NSGAIIISampler, GPSampler
from torch.utils.data import DataLoader
from evaluate import EvaluateExplanation
import captum.attr as a
from optuna.trial import TrialState,Trial
import seaborn as sns
import time 
from dataset import OurDataset
import copy
import importlib
import itertools
from architecture import get_tokenizer
import scipy
import warnings
import tracemalloc
import gc
from collections import Counter
from itertools import product


class Check_docano_XAI:
        
    def __init__(self,rationale_dataset_path: str,xai_dataset: OurDataset,tuple_indexes: tuple[tuple]) -> None:
        with open(rationale_dataset_path, "r", encoding="utf-8") as f:
                self.rationale_dataset = json.load(f)
        self.tuple_indexes=tuple_indexes
        self.xai_dataset=xai_dataset

    def check_xaiposts_docanoposts(self) -> list:
            """
            Function which match posts and fact_checks by indexes
            output: lists of indexes which match rationale masks and Ourdataset
            """
            xai=[]
            doc=[]
            for  number,entry in enumerate(self.rationale_dataset):
                for number_1,tuple_index in enumerate(self.tuple_indexes):
                    claim_id,post_id = tuple_index
                    if entry['post_id'] == post_id and entry['fact_check_id'] == claim_id:
                        xai.append(number_1)
                        doc.append(number)
            return doc,xai
    
    def get_post(self,dataset: str,index: int) : #dataset need to be set "data" or "doc" index is index in particular dataset
        """
        function get specific post by index in dataset with rationale masks or in Ourdataset. 
        Input: 'doc',index
        Output: string (Ourdataset) or list (rationale mask) of post
        """
        if dataset == 'data':
            l=self.xai_dataset[index]
            return l[1]
        if dataset == 'doc':
            l=self.rationale_dataset[index]
            return l['post_rationale']['tokens']
        else: 
            raise TypeError('Write in format: "data" or "doc", index(int)')
        
    def get_matched_doccano(self,):
        """
            Function which match posts and fact_checks by indexes
            output: lists of indexes which match rationale masks and dataset
        """
        indexes_doc,indexes_xai=self.check_xaiposts_docanoposts()
        doc_data=[self.rationale_dataset[i] for i in indexes_doc]
        return indexes_doc,indexes_xai,doc_data
    
    @staticmethod
    def exist_maps(path: str, list_methods: list) -> list:
        """
            Load explanations from text format and transform  
            Check methods in text file and transform all methods in explanation
            Take into account small and capital letters 

            Input: path with explanations; list of all methods from text file 
            Output: lists with dictionaries {"post": [['...']],"claim":[['...']],'explanation':[...]}
        """
        with open(path, 'r',encoding='utf-8') as file:
            content = file.read()
            content = content.replace('\n        ','').replace('\n       ','')
            # content = content
            content = re.sub('\\\\','',content)


        # counter=0
        explanations=[]
        exp={}
        maps_dict={}
        regex = re.compile(r'''\[{'explanation':''')
        regex2 = re.compile(r''' '</s>']]}]''')
        matches = [(match.start()) for match in regex.finditer(content)]
        matches2 = [(match.end()) for match in regex2.finditer(content)]
        combined_list = [[matches[i], matches2[i]] for i in range(len(matches))]

        for list_id in combined_list:
            expl_post=content[list_id[0]:list_id[1]]
            index = expl_post.find(", 'post': ")
            explanation = expl_post[:index]  # Part before the delimiter
            text = expl_post[index:]
            explanation=explanation.replace("]],device='cuda:0'))", "]], device='cuda:0'))")
            exp_list_method=['Lime', 'GradientShap', 'Input X Gradient', 'Saliency', 'Guided Backprop', 'Deconvolution', 'Occlusion', 'Kernel Shap']
            parts = {}
            current_pos = 0
            for i in range(len(exp_list_method) - 1):
                word_start = explanation.find(exp_list_method[i], current_pos)
                next_word_start = explanation.find(exp_list_method[i + 1], word_start)
                if word_start != -1 and next_word_start != -1:
                    part = explanation[word_start:next_word_start]
                    parts[exp_list_method[i]]=part
                    current_pos = next_word_start 
            final_word_start = explanation.find(exp_list_method[-1], current_pos)
            if final_word_start != -1:
                part=explanation[final_word_start:]
                parts[exp_list_method[-1]]=part


            for method, part in parts.items(): 
                if method in list_methods:
                    part=part.replace(f"{method}': (tensor(", '').replace('tensor(', '').replace(" device='cuda:0'), ", '').replace("device='cuda:0'), ", '').replace(" device='cuda:0')), ", '').replace(",'", '').replace(", device='cuda:0'))}",'')
                    list_of_lists_of_lists = ast.literal_eval(part)
                    post_map = torch.tensor(list_of_lists_of_lists[0]).squeeze()
                    claim_map= torch.tensor(list_of_lists_of_lists[1]).squeeze()
                    maps_dict[method]=(post_map,claim_map)


            exp['explanation']=maps_dict
            maps_dict={}
            text = expl_post[index:] 
            index_c = text.find(", 'claim': ")
            claim=text[index_c:]
            claim=claim.replace(", 'claim': ", '').replace('}]','').replace(", '</s>'","")
            claim=ast.literal_eval(claim)
            # claim= tokenizer.convert_tokens_to_string(claim[0])

            post=text[:index_c]
            post=post.replace(", 'post': ", '').replace(", '</s>'","")
            post=ast.literal_eval(post)
            # post= tokenizer.convert_tokens_to_string(post[0])

            exp['post']=post
            exp['claim']=claim
            explanations.append(exp)
            exp={}
        return explanations
    
    @staticmethod
    def exist_maps_json(path: str) -> list:
        '''
        load explanations from json format 
        '''
        with open(path, 'r',encoding='utf-8') as file:
            explanations =  json.load(file)

        return explanations
    
    def get_stats_data(self,length_words:bool=False ,count_words:bool=False ,length_characters:bool=False ,under_char:int=None ,under_words:int=None )-> pd.DataFrame:
        """
            Create plots and dataframe for statistical purposes.

            length_words; count_words; length_characters: If true than gives us plots based on name

            under_char; under_words: number of posts which has lower characters or words based on given number
        """
        len_words=[]
        c_w=[]
        len_char=[]
        for lol in self.rationale_dataset:
            if not lol['ocr_rationale']:
                whole_post=lol['post_rationale']['tokens']
            elif not lol['post_rationale']: 
                whole_post=lol['ocr_rationale'][0]['tokens']
            else:
                whole_post=lol['post_rationale']['tokens'] + lol['ocr_rationale'][0]['tokens']
            len_words_sen=[]
            for post_word in whole_post:
                len_words_sen.append(len(post_word))
            len_words.append(sum(len_words_sen) / len(len_words_sen))
            c_w.append(len(len_words_sen))
            len_char.append(sum(len_words_sen))
            q3, q1 = np.percentile(c_w, [75 ,25])
            iqr = q3 - q1
        df = pd.DataFrame({
        'Length of dataset': [len(c_w)],
        'Average length of words/tokens': [sum(len_words) / len(len_words)],
        'Count words: iqr':[iqr],
        'Count words: q3': q3,
        'Count words: q1': q1,})
        if under_char is not None:         
            df[f'Number of sentences under {under_char} characters']=sum(x < under_char for x in len_char)
        elif under_words is not None:
            df[f'Number of sentences under {under_words} words']=sum(x < under_words for x in c_w)
        df = pd.melt(df, var_name='index', value_name='values')
        df=df.round(2)
        if length_words == True:
            len_words.remove(max(len_words))
            plt.figure(figsize=(8, 5))
            sns.histplot(len_words)
            plt.tight_layout()
            plt.title('Average length of words in Post+OCR')
            plt.xlabel('Count of characters')
            plt.ylabel('Frequency')
            plt.show()
        if count_words == True:
            plt.figure(figsize=(8, 5))
            sns.histplot(c_w)
            plt.tight_layout()
            plt.title('Count of words in Posts+OCR')
            plt.xlabel('Count of words')
            plt.ylabel('Frequency')
            plt.show()
        if length_characters == True:
            plt.figure(figsize=(8, 5))
            sns.histplot(len_char)
            plt.tight_layout()
            plt.title('Length of words in Posts+OCR')
            plt.xlabel('Length of words')
            plt.ylabel('Frequency')
            plt.show()
        return df
    
    def delete_data(self,way:str,number_of_tokens:int) -> None:
        """
        function to delete data which are not suitable based on size of text 

        Input: way: the way delete to big 'over' or to small 'under' posts from dataset
        Output: change self.rationale_dataset
        """
        try:
            assert way == "under" or way == "over", f'You need to set variable way to "over" for cutting words/tokens above limit or "under" for cutting words/tokens under limit'
            filtered_dataset=[]
            for data in self.rationale_dataset:
                if not data['ocr_rationale']:
                    whole_post=data['post_rationale']['tokens']
                elif not data['post_rationale']: 
                    whole_post=data['ocr_rationale'][0]['tokens']
                else:
                    whole_post=data['post_rationale']['tokens'] + data['ocr_rationale'][0]['tokens']
                if way == "under": 
                    if number_of_tokens>=len(whole_post):
                        filtered_dataset.append(data)
                else: 
                    if number_of_tokens<=len(whole_post):
                        filtered_dataset.append(data)
                        
            self.rationale_dataset = filtered_dataset
        except Exception as e:
            print(e)
        print(len(self.rationale_dataset))

    def check_dupl_post_claim_combination(self)-> list:
        """
        Check if we have unique combinations of posts+ocr and claims in dataset

        Output: list of indexes in dataset where are duplicates.
        Warning: delete from dataset with rationale masks and from Ourdataset
        """
        seen = set()
        duplicates = set()
        for i in self.rationale_dataset:
            post=self.xai_dataset.all_df_posts.loc[i['post_id']]['content']
            claim=self.xai_dataset.all_df_fact_checks.loc[i['fact_check_id']]['claim'][0]
            text = post + claim
            if text in seen:
                duplicates.add(text)
            else:
                seen.add(text)
        return duplicates

         

class Compare_docano_XAI:

    def __init__(self,rationale_dataset,importance_map: list[dict]) -> None:
        if isinstance(rationale_dataset,list):
            self.rationale_dataset=rationale_dataset
        elif isinstance(rationale_dataset, str):
            try:
                self.rationale_dataset=None
                with open(rationale_dataset, "r", encoding="utf-8") as f:
                    self.rationale_dataset = json.load(f)
            except:
                raise ('Wrong path or wrong file')  
        else:
            self.rationale_dataset= None
                      
        self.importance_map=importance_map

        columns = ['Post','Number of positives']
        self.df_score = pd.DataFrame(columns=columns)

    """
    Main function of this class is compare explanations with rationale masks for new implemented metrics (besides ERASER) or additional_metrics
    If are not rationales this will be not able to compute

    rationale_dataset:  
    """
    
    ###
    #  Normalizations
    ###
    @staticmethod
    def min_max_normalize(expl: torch.Tensor) -> torch.Tensor:
        for i in range(len(expl)):
            max_r = torch.nanquantile(expl, 0.95)
            min_r = torch.nanquantile(expl, 0.05)
            expl[i] = (expl[i] - min_r) / (max_r - min_r + 1e-9)
            expl[i] = torch.clip(expl[i], min=0, max=1)
        return expl
    
    @staticmethod
    def log_normalize(explanation_tensor: torch.Tensor) -> torch.Tensor:
        explanation_tensor = torch.log1p(explanation_tensor) 
        return explanation_tensor
    
    @staticmethod
    def log_scale_normalize(explanation_tensor: torch.Tensor) -> torch.Tensor:
        max_r=torch.nanquantile(explanation_tensor, 0.95)
        explanation_tensor=torch.log1p(explanation_tensor) / torch.log1p(max_r)
        return explanation_tensor
    
    @staticmethod
    def log_min_max_normalize(explanation_tensor: torch.Tensor) -> torch.Tensor:
        log_tensor = torch.log1p(explanation_tensor)
        max_log = torch.nanquantile(explanation_tensor, 0.95)
        min_log = torch.nanquantile(explanation_tensor, 0.05)
        explanation_tensor = (log_tensor - min_log) / (max_log - min_log + 1e-9)
        explanation_tensor = torch.clip(explanation_tensor, min=0, max=1)
        return explanation_tensor
    
    @staticmethod
    def log_mean_normalize(explanation_tensor: torch.Tensor, epsilon=1e-8) -> torch.Tensor:
        log_tensor = torch.log1p(explanation_tensor + epsilon)
        mean_log = torch.mean(explanation_tensor)
        q1_log = torch.nanquantile(log_tensor, 0.25)
        q3_log = torch.nanquantile(log_tensor, 0.75)
        iqr_log = q3_log - q1_log
        explanation_tensor = (log_tensor - mean_log) / iqr_log
        return explanation_tensor
    
    @staticmethod
    def log_softplus_normalize(explanation_tensor: torch.Tensor) -> torch.Tensor:
        explanation_tensor=torch.log1p(torch.exp(explanation_tensor))
        return explanation_tensor
    @staticmethod
    def log_sigmoid_normalize(explanation_tensor: torch.Tensor) -> torch.Tensor:
        explanation_tensor=1 / (1 + torch.exp(-explanation_tensor))
        return explanation_tensor
    
    @staticmethod 
    def tanh_normalize(explanation_tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.mean(explanation_tensor)
        q1_log = torch.nanquantile(explanation_tensor, 0.25)
        q3_log = torch.nanquantile(explanation_tensor, 0.75)
        iqr_log = q3_log - q1_log
        explanation_tensor = torch.tanh((explanation_tensor - mean) / iqr_log)
        return explanation_tensor
    @staticmethod 
    def without_normalize(explanation_tensor: torch.Tensor) -> torch.Tensor:
        return explanation_tensor

    @staticmethod
    def mean_var_normalize(explanation_tensor: torch.Tensor, eps=1e-25)-> torch.Tensor:
        # explanation_tensor = explanation_tensor.squeeze(0)
        if explanation_tensor.numel() <= 1:
            return explanation_tensor
        mean = torch.mean(explanation_tensor)
        var = torch.var(explanation_tensor)
        return (explanation_tensor - mean) / torch.sqrt(var + eps) 
    
    @staticmethod
    def second_moment_normalize(explanation_tensor: torch.Tensor, eps=1e-25) -> torch.Tensor:
        q75 = torch.nanquantile(explanation_tensor, 0.75, dim=0, keepdim=True)
        q25 = torch.nanquantile(explanation_tensor, 0.25, dim=0, keepdim=True)
        iqr = q75 - q25
        mean_iqr = torch.mean(iqr, dim=-1, keepdim=True)
        normalized_tensor = explanation_tensor / (mean_iqr + eps)
        return normalized_tensor
    
        
    ###
    #  Metrics
    ###

    @staticmethod
    def dice_coe(xai_tensor: torch.Tensor,mask: torch.Tensor) -> int:
        """Compute Dice Coefficient."""
        intersection = torch.sum(xai_tensor * mask)
        union = torch.sum(xai_tensor) + torch.sum(mask)
        result=(2.0 * intersection) / union
        return result
    
    @staticmethod
    def jac_coe(xai_tensor,mask) -> int:
        """Compute Jaccard Coefficient."""
        intersection = torch.sum(xai_tensor * mask)
        union = torch.sum(xai_tensor) + torch.sum(mask) - intersection 
        result=intersection / union
        return result
    
    @staticmethod
    def TP(xai_tensor,mask):
        """Compute True Positives."""
        return torch.sum(xai_tensor * mask)
    @staticmethod
    def TN(xai_tensor, mask):
        """Compute True Negatives."""
        return torch.sum((1 - xai_tensor) * (1 - mask))
    
    @staticmethod
    def FP(xai_tensor, mask):
        """Compute False Positives."""
        return torch.sum(xai_tensor * (1 - mask))
    
    @staticmethod
    def FN(xai_tensor, mask):
        """Compute False Negatives."""
        return torch.sum((1 - xai_tensor) * mask)
    
    @staticmethod
    def precision(xai_tensor, mask):
        """Compute Precision."""
        tp = torch.sum(xai_tensor * mask)
        fp = torch.sum(xai_tensor * (1 - mask))
        return tp / (tp + fp) if (tp + fp) > 0 else torch.zeros(1)
    
    @staticmethod
    def recall(xai_tensor, mask):
        """Compute Recall."""
        tp = torch.sum(xai_tensor * mask)
        fn = torch.sum((1 - xai_tensor) * mask)
        return tp / (tp + fn) if (tp + fn) > 0 else torch.zeros(1)
    
    @staticmethod
    def accuracy(xai_tensor, mask):
        """Compute Accuracy."""
        tp = torch.sum(xai_tensor * mask)
        fn = torch.sum((1 - xai_tensor) * mask)
        fp = torch.sum(xai_tensor * (1 - mask))
        tn = torch.sum((1 - xai_tensor) * (1 - mask))
        return (tp+tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else torch.zeros(1)
    
    @staticmethod
    def F1(xai_tensor, mask):
        """Compute F1 Score."""
        fp = torch.sum(xai_tensor * (1 - mask))
        tp = torch.sum(xai_tensor * mask)
        fn = torch.sum((1 - xai_tensor) * mask)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else torch.zeros(1)
    
    def overall_percentage(xai_tensor,mask):
            total = mask.numel() 
            matches = torch.sum(mask == xai_tensor)
            pos_over = (matches / total) * 100
            return pos_over
    
    def positive_percentage(xai_tensor,mask):
            overlap = mask & xai_tensor
            overlap_sum = torch.sum(overlap)
            mask_sum = torch.sum(mask)
            per_overlap = (overlap_sum / mask_sum) * 100
            return per_overlap
    @staticmethod
    def gini_f(xai_tensor:torch.tensor):
        sorted_tensor, indices = torch.sort(xai_tensor, descending=False)
        n = sorted_tensor.shape[0]
        # try:
        indices = torch.arange(1, n + 1, dtype=torch.float32, device='cpu')
        gini = float(torch.sum((2 * indices - n - 1) * sorted_tensor) / (n * torch.sum(sorted_tensor)))
        return gini
    @staticmethod
    def point_game_f(xai_tensor:torch.tensor,mask:torch.tensor)->int:
        count_ones = torch.sum(mask == 1).item()
        total_elements = mask.numel()
        percentage = (count_ones / total_elements) * 100
        num_elements = int((percentage / 100) * xai_tensor.numel())
        top_values, top_indices = torch.topk(xai_tensor, num_elements)
        mask_indices = torch.nonzero(mask, as_tuple=False).squeeze()
        matches = torch.isin(top_indices, mask_indices).sum().item()
        if num_elements == 0:
            return 0
        final_percentage = matches / num_elements
        return final_percentage
    @staticmethod    
    def localization(xai_tensor:torch.tensor,mask:torch.tensor)->int:
        gini=Compare_docano_XAI.gini_f(xai_tensor)
        point_game=Compare_docano_XAI.point_game_f(xai_tensor,mask)
        if point_game == 0 or gini ==0:
            return 0
        return (gini*0.3 + point_game*0.7)/1
    
    @staticmethod
    def count_chunks_of_ones(binary_mask):   
        transitions = (binary_mask[:-1] == 0) & (binary_mask[1:] == 1)
        num_chunks = transitions.sum().item() + (binary_mask[0] == 1).item()
        return num_chunks
        # c=[]
        # a=0
        # for word in post:
        #     tokenized_word=tokenizer(word)
        #     list_tok=tokenizer.convert_ids_to_tokens(tokenized_word['input_ids'])
        #     list_tok = [tok for tok in list_tok if tok not in tokenizer.all_special_tokens]
        #     a+=len(list_tok)
        #     c.append(list_tok)
        #     print(list_tok)

    def connect_rationale(post_claim_pair):
        ocr={}
        post_dic={}

        if post_claim_pair['post_rationale']:
            post_dic=post_claim_pair['post_rationale']
        if post_claim_pair['ocr_rationale']:
            ocr=post_claim_pair['ocr_rationale'][0]
            if isinstance(ocr,list):
                ocr=ocr[0]
        if 'tokens' in post_dic and 'tokens' in ocr:
            post_doccano=post_dic['tokens']+ocr['tokens']
            mask=post_dic['mask']+ocr['mask']
        elif 'tokens' not in ocr:
            post_doccano=post_dic['tokens']
            mask=post_dic['mask']
        elif 'tokens' not in post_dic:
            post_doccano=ocr['tokens']
            mask=ocr['mask']
        return post_doccano,mask



    def concise_tensor_word(model,xai_tensor:torch.tensor,post:list)->torch.tensor:
        """
        Computing of explanation is based on significance of tokens, problem comes when we want explanations based on words for better visualization.
        Function change mean explanations if word has more than one token.

        Args: 
                xai_tensor: 1D tensor
                post: list with ordinary tokens (words)
        Returns: explanations with reduced size
        """
        tokenizer=get_tokenizer(model)
        adjusted_tensor=[]
        counter=0
        for word in post:
            tokenized_word=tokenizer(word)
            list_tok=tokenizer.convert_ids_to_tokens(tokenized_word['input_ids'])
            list_tok = [tok for tok in list_tok if tok not in tokenizer.all_special_tokens]
            #xai_tensor from rationale to normal
            counter+=len(list_tok)
            word_tensor=xai_tensor[0:len(list_tok)]
            try:
                word_mean_tensor = torch.max(word_tensor)
                word_mean_tensor.item()
            except:
                word_mean_tensor = torch.tensor(float('nan'))
            adjusted_tensor.append(word_mean_tensor)
            xai_tensor=xai_tensor[len(list_tok):]
        adjusted_tensor=torch.Tensor(adjusted_tensor) 
        len_tens=list(adjusted_tensor.size())
        if len_tens[0] > 512:
           adjusted_tensor=adjusted_tensor[:512]
        return adjusted_tensor
        


    def concise_tensor_sentence(xai_tensor:torch.tensor,post:list,mask:torch.tensor=None )->torch.tensor:
        start=0
        adjusted_mask=[]
        adjusted_tensor=[]
        adjusted_sentence=[]
        try:
            for number,word in enumerate(post):
                if any(p in word for p in ".,!?;:…") or number == list(xai_tensor.size())[0]-1 : #If punctuation is in string or it is last string
                    adjusted_sentence.append(' '.join(post[start:number+1]))
                    word_tensor= xai_tensor[start:number+1]
                    if mask is not None:
                        tensor_mask= mask[start:number+1]
                        mask_part= torch.tensor(1) if torch.sum(tensor_mask) > tensor_mask.numel() / 2 else torch.tensor(0)
                        adjusted_mask.append(mask_part.item())
                    try:
                        word_mean_tensor = torch.sum(word_tensor)
                        word_mean_tensor.item()
                    except:
                        word_mean_tensor = torch.tensor(float('nan'))
                    adjusted_tensor.append(word_mean_tensor)
                    start=number+1
            adjusted_tensor=torch.Tensor(adjusted_tensor) 
            if mask is not None:
                adjusted_mask=torch.Tensor(adjusted_mask) 
                return adjusted_tensor,adjusted_sentence,adjusted_mask
            else: 
                return adjusted_tensor,adjusted_sentence
        except:
            print('')


    def change_token_explanation(self,model:STS_ExplainWrapper,dataset:OurDataset,sentence_exp:bool=False,word_exp:bool=False) -> pd.DataFrame:    
        assert sentence_exp != word_exp, "Error: You cannot compute sentence explanation and word explanations at the same time." 
        new_importance_map=[]
        for index_doccano,post_claim_pair in enumerate(self.rationale_dataset):
            # self.importance_map[index_doccano]
            expl_dict={}
            post_doccano,mask=Compare_docano_XAI.connect_rationale(post_claim_pair)
            claim_id=post_claim_pair['fact_check_id']
            claim=dataset.id_to_fact_check_vocab[claim_id]
            claim=claim.split(' ')

            if isinstance(self.importance_map[index_doccano],list):
                expl=self.importance_map[index_doccano][0]
            else: 
                expl=self.importance_map[index_doccano]

            for method,list_tensors in expl['explanation'].items():
                expl_dict['explanation']=[]
                for index,tensor in enumerate(list_tensors):
                    if tensor.shape[0] == 1:
                        tensor=torch.squeeze(tensor)
                    if index==0: 
                        mask = torch.tensor(mask).clone().detach()
                        if tensor.numel()==1:
                            pass
                        elif mask.shape[0] < tensor.shape[0]:
                            if sentence_exp:
                                tensor=Compare_docano_XAI.concise_tensor_word(model,tensor,post_doccano)
                                tensor,post_doccano,mask=Compare_docano_XAI.concise_tensor_sentence(tensor,post_doccano,mask)
                            if word_exp:
                                tensor=Compare_docano_XAI.concise_tensor_word(model,tensor,post_doccano)
                                
                            if mask.shape[0] < tensor.shape[0]:  
                                tensor=tensor[:mask.shape[0]] 
                            elif mask.shape[0] > tensor.shape[0]:
                                mask=mask[:tensor.shape[0]] 
                        else:
                            tensor = tensor
                            mask = mask[:tensor.shape[0]] 
                    if index == 1:
                        if tensor.size()[0] > len(claim):
                            if sentence_exp:
                                tensor=Compare_docano_XAI.concise_tensor_word(model,tensor,claim)
                                tensor,claim=Compare_docano_XAI.concise_tensor_sentence(tensor,claim)
                            if word_exp:
                                tensor=Compare_docano_XAI.concise_tensor_word(model,tensor,claim)
                               

                    tensor=torch.unsqueeze(tensor,dim=0)
                    expl_dict['explanation'].append(tensor)
            expl_dict['explanation']={method:tuple(expl_dict['explanation'])}
            expl_dict['post']=[post_doccano]
            expl_dict['claim']=[claim]
            expl_dict['mask']=mask
            new_importance_map.append(expl_dict)
        return new_importance_map
    

    #set mask for tokens 
    def concise_word_mask(post,mask,tokenizer):
        post_list=post.split(' ')
        counter=0
        new_mask=[]
        for word,binary in zip(post_list,mask):
            tokenized_word=tokenizer(word)
            list_tok=tokenizer.convert_ids_to_tokens(tokenized_word['input_ids'])
            list_tok = [tok for tok in list_tok if tok not in tokenizer.all_special_tokens]
            new_mask.extend([binary]*len(list_tok))
        new_mask=torch.tensor([t.item() for t in new_mask])
        return new_mask 


    @staticmethod
    def find_mask(annotations,explanations,dataset=None,tokenizer=None,method=None):
        for i,annot in enumerate(annotations): 
            post_doccano,mask=Compare_docano_XAI.connect_rationale(annot)
            if isinstance(mask,list):
                mask=torch.Tensor(mask)
            if dataset and tokenizer and method:
                if list(mask.size())[0] != list(explanations[i]['explanation'][method][0].size())[1]:
                    post=dataset[i][1]
                    mask=Compare_docano_XAI.concise_word_mask(post,mask,tokenizer)
            if list(mask.size())[0] > 510: 
                mask = mask[:510]
            if explanations[i]['explanation'][method][0].squeeze().size()[0] != mask.size()[0]:
                # if mask.size()[0] < explanations[i]['explanation'][method][0].squeeze().size()[0]
                print(f"{explanations[i]['explanation'][method][0].squeeze().size()[0]}:{mask.size()[0]}")
            explanations[i]['mask']=mask
        return explanations



    def compare(self,index_doccano: int,model,      
                sig_number=None,best_percent=None,         
                percentage_metrics=None,
                evaluation_metrics=None,

                normalization=None,
                ) -> pd.DataFrame:
        """
        Added metrics outside the eraser metrics file

        Input: index_doccano: index number in dataset with rational mask and Ourdataset 
        Output: dataframe with computed metrics 
        """    
        try:    
            mask= self.importance_map[index_doccano]['mask']
            post_doccano=self.importance_map[index_doccano]['post'][0]
            # ocr={}
            # post_dic={}
            # if self.rationale_dataset[index_doccano]['post_rationale']:
            #     post_dic=self.rationale_dataset[index_doccano]['post_rationale']
            # if self.rationale_dataset[index_doccano]['ocr_rationale']:
            #     ocr=self.rationale_dataset[index_doccano]['ocr_rationale'][0]
            #     if isinstance(ocr,list):
            #         ocr=ocr[0]
            # if 'tokens' in post_dic and 'tokens' in ocr:
            #     post_doccano=post_dic['tokens']+ocr['tokens']
            #     mask=post_dic['mask']+ocr['mask']

            # elif 'tokens' not in ocr:
            #     post_doccano=post_dic['tokens']
            #     mask=post_dic['mask']

            # elif 'tokens' not in post_dic:
            #     post_doccano=ocr['tokens']
            #     mask=ocr['mask']


            row = {'Post': post_doccano}
            if isinstance(self.importance_map[index_doccano],list):
                expl=self.importance_map[index_doccano][0]
            else: 
                expl=self.importance_map[index_doccano]

            for method,list_tensors in expl['explanation'].items():
                xai_tensor=list_tensors[0]
                if xai_tensor.shape[0] == 1:
                    xai_tensor=torch.squeeze(xai_tensor)
                # if 'e5' in model.hf_transformer.config._name_or_path: 
                #     xai_tensor=xai_tensor[1:-1]
                # if 'T5' in model.hf_transformer.config._name_or_path:
                #     xai_tensor = xai_tensor[:-1]
                # mask = torch.tensor(mask).clone().detach()
                # if mask.shape[0] < xai_tensor.shape[0]:
                #     # mask=Compare_docano_XAI.adjust_spread_mask(model,mask,post_doccano)
                #     # print(len(xai_tensor),len(post_doccano))
                #     xai_tensor=Compare_docano_XAI.concise_tensor(model,xai_tensor,post_doccano)
                #     if mask.shape[0] < xai_tensor.shape[0]: # sometimes in computation map dont have full verion of post 
                #         xai_tensor=xai_tensor[:mask.shape[0]] 
                #     elif mask.shape[0] > xai_tensor.shape[0]: # sometimes in computation map dont have full verion of post 
                #         mask=mask[:xai_tensor.shape[0]] 
                # else:
                #     xai_tensor = xai_tensor
                #     mask = mask[:xai_tensor.shape[0]] 
                # #normalizations
                if normalization is not None:
                    if isinstance(normalization,str):
                            norm=getattr(Compare_docano_XAI, normalization)
                            xai_tensor=norm(xai_tensor)
                            xai_tensor=torch.nan_to_num(xai_tensor)
                if float(xai_tensor.min()) <0:
                    xai_tensor=xai_tensor+abs(float(xai_tensor.min()))

                # pos=(xai_tensor * mask).sum()
                # all=xai_tensor.sum()
                # perc=(pos/all).item()
                # row[f'perc-{method}'] = perc #check deconvolution
                # variance=torch.var(xai_tensor)
                # row[f'var-{method}'] = variance.item()
                # xai_tensor.to('cpu')
                try: 
                    mask.to(utils.get_device())
                    row[f'avg_prec_score-{method}']= average_precision_score(mask, xai_tensor)
                except Exception as e :
                    # row[f'avg_prec_score-{method}']=0
                    print(e)
                    continue
                #Gini index


                # print('---')
                if min(xai_tensor) < 0:
                    adjust_xai_tensor=xai_tensor + abs(min(xai_tensor))
                else:
                    adjust_xai_tensor=xai_tensor
                adjust_xai_tensor += 0.0000001
                sorted_tensor, indices = torch.sort(adjust_xai_tensor, descending=False)
                n = sorted_tensor.shape[0]
                # try:
                indices = torch.arange(1, n + 1, dtype=torch.float32)
                indices=indices.to(utils.get_device())
                sorted_tensor=sorted_tensor.to(utils.get_device())
                try:
                    gini = float(torch.sum((2 * indices - n - 1) * sorted_tensor) / (n * torch.sum(sorted_tensor)))
                    row[f'gini-{method}']=gini
                except Exception as e:
                    gini=0
                    row[f'gini-{method}']=gini
                    print(e)
                spare=Compare_docano_XAI.gini_f(xai_tensor)
                row[f'spareness-{method}']=spare
                local=Compare_docano_XAI.localization(xai_tensor,mask)
                row[f'localization-{method}']=local
                points=Compare_docano_XAI.point_game_f(xai_tensor,mask)
                row[f'point_game-{method}']=points
                # print(f"Quantum implemented spareness {gini}")

                # frac_con_dis= xai_tensor/torch.abs(xai_tensor).sum()             # https://arxiv.org/pdf/2005.00631
                # entropy=scipy.stats.entropy(pk=frac_con_dis)
                # print(f'Quantum entropy {entropy}')

                # print(f'Spareness Gini index frac_con_dis{1 - torch.sum(frac_con_dis**2).item()}')


                # xai_tensor_smooth=xai_tensor+1e-10
                # mask_smooth=mask+1-10
                # e=scipy.stats.entropy(pk=xai_tensor_smooth)
                # e=scipy.stats.entropy(pk=mask_smooth,qk=xai_tensor_smooth)
                # print(f'Kullback-Leibler divergence {e}')
                # num_elements = frac_con_dis.numel()
                # num_top_5_percent = max(1, int(num_elements * 0.05))
                # sorted_tensor, _ = torch.sort(frac_con_dis, descending=True)
                # av_top_5=torch.mean(sorted_tensor[:num_top_5_percent].float()).item()
                # print(f'Top 5% of contribution {av_top_5}')
                # print('---')



                #word level
                # im_words=[]
                # word_level = defaultdict(dict)
                # for num,word in enumerate(post_doccano):    
                #     # word_level[word]['length']=len(word)
                #     # word_level[word]['bin']=mask[num]
                #     # word_level[word]['importance']=xai_tensor[num]
                #     if mask[num+1].item() == 1:
                #         im_words.append(len(word))
                # row['len_words'] = sum(im_words) / len(im_words) if im_words else 0




                # chunk level commented because 
                # chunk_level= defaultdict(dict)
                # transitions = (mask[:-1] == 0) & (mask[1:] == 1)
                # num_chunks = transitions.sum().item() + (mask[0] == 1).item()
                # # chunk_level[post_doccano]['number of chunks']=num_chunks
                # row['number of chunks'] = num_chunks

                # ones_positions = torch.where(mask == 1)[0]
                # diff = ones_positions[1:] - ones_positions[:-1]
                # chunk_ends = torch.where(diff > 1)[0]
                # lengths = torch.diff(torch.cat((torch.tensor([-1]), chunk_ends, torch.tensor([len(ones_positions)-1]))))
                # chunk_lengths = lengths
                # avg_length = chunk_lengths.float().mean().item()
                # # chunk_level[post_doccano]['avg length of chunks']= avg_length
                # row['avg length of chunks'] = avg_length



                # if not torch.isnan(xai_tensor).any() and len(post_doccano) > 4:
                #     xai_unsqueeze=xai_tensor.unsqueeze(1)
                #     kmeans = KMeans(n_clusters=4,random_state=0).fit(xai_unsqueeze)
                #     centers = kmeans.cluster_centers_
                #     sorted_indices = np.argsort(centers, axis=0)[:, 0]
                #     label_mapping = {sorted_indices[i]: i for i in range(4)}
                #     new_labels = np.array([label_mapping[label] for label in kmeans.labels_])
                #     new_labels_tensor = torch.tensor(new_labels)

                    # bin = torch.where(new_labels_tensor > 1, torch.tensor(1), torch.tensor(0))
                    # tensor=(bin*mask)/mask
                    # valid_tensor = tensor[~torch.isnan(tensor)]
                    # total_valid = valid_tensor.numel()
                    # num_ones = (valid_tensor == 1).sum().item()
                    # percent_ones = (num_ones / total_valid) * 100
                    # row[f'Ones in mask(m3)-{method}'] = percent_ones
                    # if len(np.unique(kmeans.labels_)) > 1:
                    #     silhouette_avg = silhouette_score(xai_unsqueeze, new_labels)
                    #     row[f'Silhouette_avg-{method}'] = silhouette_avg
                    #     row[f'Sparsity-{method}'] = (bin == 0).sum().item()/bin.size()[0] 

                    #compare centers
                    # or_centers=centers.squeeze(1)
                    # or_centers=np.sort(or_centers)[::-1]
                    # cluster_diff=np.diff(np.flip(or_centers))
                    # row[f'Cluster_diff-{method}']=np.mean(cluster_diff)           

                #     distances=[]
                #     last_index = -1
                #     for i in range(len(new_labels_tensor)):
                #         if new_labels_tensor[i] == 2:
                #             if last_index != -1:
                #                 distances.append(i - last_index)
                #             last_index = i
                #     if len(distances) != 0:
                #         mean_dist=sum(distances)/len(distances)
                #         row[f'distance-{method}']=mean_dist
                #     else:
                #         row[f'distance-{method}']=0
                # else:
                #     row[f'distance-{method}']=None





                    # plt.figure(figsize=(10, 6))
                    # plt.hist(xai_tensor[mask.type(torch.bool)], bins=30, density=True, alpha=0.6, color='g')
                    # plt.title(normalization+ '-positive-'+method)
                    # plt.xlabel('Value')
                    # plt.ylabel('Frequency')
                    # plt.grid(True)
                    # plt.show()

                    # plt.figure(figsize=(10, 6))
                    # plt.hist(xai_tensor[~mask.type(torch.bool)], bins=30, density=True, alpha=0.6, color='g')
                    # plt.title(normalization+ '-negative-'+method)
                    # plt.xlabel('Value')
                    # plt.ylabel('Frequency')
                    # plt.grid(True)
                    # plt.show()
                    # print(post_doccano)




                #infer singnificance
                # if isinstance(sig_number,float):
                #     xai_tensor=(xai_tensor > sig_number).int()
                # if isinstance(best_percent,int):
                    # max=torch.quantile(xai_tensor, 0.80).item()
                    # threshold = (1 - (best_percent / 100)) * max
                    # xai_tensor=(xai_tensor > threshold).int()


                    # threshold_value = torch.quantile(xai_tensor.float(), 1 - (best_percent / 100))
                    # xai_tensor=(xai_tensor > threshold_value).int()


                #evaluation metrics and percentage
                # if isinstance(evaluation_metrics,list):
                #     if len(evaluation_metrics) >= 0:
                #         for metric in evaluation_metrics:
                #             obj_metric=getattr(Compare_docano_XAI, metric)
                #             calculated_metric=obj_metric(xai_tensor,mask)
                #             row[method+f'-{metric}'] = calculated_metric.item()
            row['len_sentence'] = mask.size(dim=0) 
            self.df_score = pd.concat([self.df_score, pd.DataFrame([row])], ignore_index=True)   
        except Exception as e :
            print (e)
        return self.df_score

    # def vizualisations(self,explanations,model,path_to_save,evaluation_metrics,sig_numbers=None,list_of_percentage=None,normalization=None):
    #     loop_eval=sig_numbers
    #     if isinstance(list_of_percentage,list):
    #         loop_eval=list_of_percentage

    #     for perc in loop_eval:
    #         columns = ['Post','Number of positives',"Perc"]
    #         self.df_score = pd.DataFrame(columns=columns)
    #         for number in range(0,len(explanations)):
    #             self.df_score=self.compare(number,model=model,     
    #                         best_percent=perc,
    #                         normalization=normalization,
    #                         evaluation_metrics=evaluation_metrics)
    #         df=self.df_score           
    #         df["Perc"]=perc
    #         # df.to_csv(f'{path_to_save}/{normalization}_{perc}.csv',index=False)
    #         df=df.drop('Post',axis='columns')
    #         df=df.mean()
    #         df=df.to_frame().T
    #         try:
    #             final_df= pd.concat([final_df, df], ignore_index=True)
    #         except:
    #             final_df=df
    #     # Creation of graph
    #     a=list(df.columns[2:])
    #     methods=set([s.split('-')[0] for s in a])
    #     d = defaultdict(list,{ k:[] for k in methods })

    #     for col in df.columns[2:]:
    #         col_spl=col.split('-')[0]
    #         d[col_spl].append(col)

    #     palette = sns.color_palette("husl", len(d))
    #     plt.figure(figsize=(10, 6))

    #     y_col = 'Perc'
    #     for group, cols in d.items():
    #         for col in cols:
    #             if col != y_col:  # Skip the y-column
    #                 plt.plot(final_df[y_col],final_df[col], label=col, color=palette[list(d.keys()).index(group)])
        
    #     plt.xlabel('Percentage threshold for Xai')
    #     plt.ylabel(evaluation_metrics[0])
    #     plt.title(f'Methods with various tresholds with {normalization}')
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.tight_layout()
    #     if normalization == None:
    #         plt.savefig(f'{path_to_save}/{normalization}.png', format='png', dpi=300)

    #     plt.savefig(f'{path_to_save}/{normalization}.png', format='png', dpi=300)
        # plt.show()



class Hyper_optimalization:
        def __init__(self,
                    dataset:OurDataset,
                    model_path:str,
                    embeddings_module_name:str,
                    methods:list,
                    normalizations:list,
                    rationale_path:str = None,
                    method_param:dict = None,
                    model_param:dict = None,
                    perc: int =None,
                    explanations_path: str = None,
                    explanation_maps_word: bool = False,
                    explanation_maps_token: bool = False,
                    explanation_maps_sentence: bool = False,
                    faithfulness_weight:int=0.5,
                    plausability_weight:int=0.5,
                    additional_metric_weight:int=0.3,
                    multiple_object:bool=False,
                    additional_metric:list=None) -> None:
            self.methods= methods
            self.normalizations= normalizations
            self.embeddings_module_name=embeddings_module_name
            self.perc=perc
            self.model_path=model_path
            self.rationale_path=rationale_path
            self.model_param=model_param
            self.method_param=method_param
            self.explanations_path=explanations_path
            self.multiple_object=multiple_object
            self.additional_metric=additional_metric
            self.additional_metric_weight=additional_metric_weight
            self.explanation_maps_word= explanation_maps_word
            self.explanation_maps_token = explanation_maps_token
            self.explanation_maps_sentence = explanation_maps_sentence
            self.counter_methods=None
            self.counter_methods_dup=None
            self.rationale_data,self.dataset=self.process_input_dataset(dataset)
            self.counter=0
            self.counter_normalizations= None
            self.counter_normalizations_dup= None
            self.plausability_weight=plausability_weight
            self.faithfulness_weight=faithfulness_weight
            self.visualizations_metric=[]
            self.additional_results=[]
            # self.plausability_word_evaluation=plausability_word_evaluation
            # self.faithfulness_word_evaluation=faithfulness_word_evaluation
            # faithfulness_word_evaluation:bool=False,plausability_word_evaluation:bool=False,
            """
            run_optimization() : Main algorithm of optimalization. Function take specific sampler with parameters.
                            - sampler: The sampler in string format (e.g., TPESampler).
                            - **kwargs: Additional keyword arguments to be passed with the sampler class.

            rationale_path: Path to rationale_dataset

            methods: Explanation methods in string form from captum or implemented methods (name of classes in ./src/xai folder).  

            normalizations: List of strings with name of functions. Given string must be same as a name of function 

            perc: Percentage of dataset with posts and claims (Ourdataset), which will be used for hyperoptimalization.
                  Perc change length of self.dataset.fact_check_post_mapping. 

            model_path: Path of model this can be:
                        Personal model with path to the folder 

            embeddings_module_name: Name of embedding layer and first layer in embedding layer
                                    Example: model 'intfloat/multilingual-e5-large' has name of embedding layer: 'embeddings.word_embeddings'

            model and method param: dictionaries which set parameters for model and method
                model_param: All methods which need to set model with specific architecture or model need specific wrapper to use specific method
                             Here also put implemented_method= Trueexplanation_maps_token

                method_param: Contain all methods and values can be "parameters": change computation of method
                                                                    "token_groups_for_feature_mask": Feature_mask defines a mask for the input, grouping features which correspond to the same interpretable feature.
                                                                                                     Feature_mask contain the same number of tensors as inputs. 
                                                                                                     Each tensor should be the same size as the corresponding input or broadcastable to match the input tensor. 
                                                                                                     Values across all tensors should be integers in the range 0 to num_interp_features - 1, 
                                                                                                     And indices corresponding to the same feature should have the same value.
                                                                                                     Note that features are grouped across tensors (unlike feature ablation and occlusion),
                                                                                                     So if the same index is used in different tensors, those features are still grouped and added simultaneously.
                                                                    "compute_baseline": masks for grouping features of the same token together
                                                                                        used for attribution methods like feature permutation or ablation
                                                                                        we want to permutate/ablate the entire tokens, not specific float values of their embeddings

            explanations_path: Path to json with explanations.
                               Explanations will be saved here for possible second usage 
                               Firstly if file not exist you need to create empty json file with empty list [] inside
                               (Hyper_optimalization.load_explanations: function for loading, Hyper_optimalization.save_json:function for saving explanations )

            multiple_object: Setting Optuna evaluation into the multi-objective or single-objective 
                             For evaluation of explanaitons we utilize two types of evaluation metrics: faithfullness and plausability
                             Multiple_object= True  Optuna make average between used types of metrics and result will be 1 integer 
                             Multiple_object= False Optuna return 2 integers (faithfulness,plausability)

            faithfulness_weight, plausability_weight: Weights for final results  
                                                      Must be multiple_object=False
                                                      Sum of values must be 1. 

            additional_metric, additional_metric_weight: Additionial computation metrics: ['localization','point_game','spareness']
                                                         Sum of all 3 weights must be 1.

            """

        def process_input_dataset(self,dataset:OurDataset):
            """
            Load datasets with specific perc and order the Ourdataset and dataset with rationale with masks.
            """
            if len(dataset) == 0:
                raise ValueError('Dataset is empty. Please load the dataset.')
            if self.rationale_path:
                check=Check_docano_XAI(rationale_dataset_path=self.rationale_path,xai_dataset=dataset,tuple_indexes=dataset.fact_check_post_mapping) #data=doccano importance_map=xai
                indexes_doc, indexes_xai, doc_data=check.get_matched_doccano()
                if isinstance(self.perc,int) or isinstance(self.perc,float) :
                    length_dataset=(self.perc/100)*len(indexes_xai)
                    length_dataset=round(length_dataset)
                    indexes_xai=indexes_xai[:length_dataset]    
                    dataset.fact_check_post_mapping = [dataset.fact_check_post_mapping[i] for i in indexes_xai]                
                    doc_data=doc_data[:length_dataset]
                return doc_data,dataset

            else: 
                length_dataset=(self.perc/100)*len(dataset.fact_check_post_mapping)
                length_dataset=round(length_dataset)
                dataset.fact_check_post_mapping=dataset.fact_check_post_mapping[:length_dataset]
                return None,dataset 

            # if isinstance(self.explanations_path,str) and os.path.exists(self.explanations_path) and os.stat(self.rationale_path).st_size == 0:
            #     explanations=check.exist_maps_json(self.explanations_path)
                # explanations=check.exist_maps(self.explanations_path,list_methods=self.methods)
                # return doc_data,dataset,indexes_xai,explanations       #Toto sa vymaze lebo sa z jsonu bude loadovat
        




        def compute_combinations(model_param, method_param, methods, normalizations):
            """
            Compute how many combinations of methods, methods with parameters, and normalizations can be created.
            Adjusted to group method-specific and model-specific parameters separately.
            """
            all_combinations = []
        
            for method in methods:
                base_parameters = method_param.get(method, {}).get("parameters", {})
                param_keys = list(base_parameters.keys())
                param_values = list(base_parameters.values())
        
                param_combinations = [dict(zip(param_keys, values)) for values in product(*param_values)] if param_keys else [{}]
        
                for param_comb in param_combinations:
                    for normalization in normalizations:
                        if len(param_comb)==0:
                            method_comb = {
                                "method": method,
                                "method_param": {
                                },
                                "normalization": normalization,
                                "model_param": {}
                                }
                        else:  
                            method_comb = {
                                "method": method,
                                "method_param": {
                                    "parameters": param_comb
                                },
                                "normalization": normalization,
                                "model_param": {}
                                }
        
                        # Add additional method parameters like "token_groups_for_feature_mask"
                        extra_params = {k: v for k, v in method_param.get(method, {}).items() if k != "parameters"}
                        method_comb["method_param"].update(extra_params)
        
                        if method in model_param:
                            model_specific_params = model_param[method]
        
                            if method == 'Lime':
                                similarity_funcs = model_specific_params.get('similarity_func', [])
                                interpretable_models = model_specific_params.get('interpretable_model', [])
        
                                for sim_func, interp_model in product(similarity_funcs, interpretable_models):
                                    temp_comb = method_comb.copy()
                                    temp_comb["model_param"] = {
                                        "similarity_func": sim_func,
                                        "interpretable_model": interp_model
                                    }
                                    all_combinations.append(temp_comb)
                            else:
                                temp_comb = method_comb.copy()
                                temp_comb["model_param"] = model_specific_params
                                all_combinations.append(temp_comb)
                        else:
                            all_combinations.append(method_comb)
        
            return all_combinations
        @staticmethod
        def load_explanations(explanations_path,method_param,model_param,method,explanations):
            """
            Load explanation with specific model and method parameters
            """
            try:    
                with open(explanations_path, 'r', encoding="utf-8") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                os.makedirs(os.path.dirname(explanations_path), exist_ok=True)
                with open(explanations_path, 'w', encoding="utf-8") as file:
                    json.dump([], file, ensure_ascii=False, indent=4)
                existing_data = []
            except Exception as e :
                print(e)
                print('Explanations are not loaded')
            for exp in existing_data:
                # Transfer 'parameters' from list to tuple to be able check explanations
                try:
                    if method in exp['method_param']:
                        if 'parameters' in exp['method_param'][method]:
                            par=exp['method_param'][method]['parameters'].values()
                            if isinstance(list(par)[0],list):
                                p = {k: tuple(v) if isinstance(v, list) else v for k, v in exp['method_param'][method]['parameters'].items()}
                                exp['method_param'][method]['parameters'] = p
                    if exp['method_param']==method_param and exp['model_param']==model_param and list(exp['explanation'].keys())[0]==method:
                        # post_exp=torch.tensor(exp['explanation'][method][0]).to(utils.get_device())
                        # claim_exp=torch.tensor(exp['explanation'][method][1]).to(utils.get_device())
                        exp['explanation'][method]=(torch.tensor(exp['explanation'][method][0]),# , dtype=torch.float64
                                                    torch.tensor(exp['explanation'][method][1])) #, dtype=torch.float64
                        explanations.extend([exp])
                except:
                    print('')
            return explanations
        
        def load_explanations_ids(explanations):
            list_ids=[]
            if explanations:
                for exp in explanations:
                    list_ids.append(exp['ids'])
            return list_ids
        @staticmethod
        def adjust_explan(explanations:list,methods:list,normalization:str):
            """
            Loaded explanations from json need to change format and apply normalization
            """
            if isinstance(methods,str):
                methods=[methods]
            list_exp=[]
            for num in range(len(explanations)):
                expl={}
                expl['explanation'] = {}
                for method in methods:
                    if method in explanations[num]['explanation'].keys():
                        norm=getattr(Compare_docano_XAI, normalization)

                        if isinstance(explanations[num]['explanation'][method][0],list):
                            explanations[num]['explanation'][method][0] = torch.tensor(explanations[num]['explanation'][method][0])
                        norm_pos=norm(explanations[num]['explanation'][method][0])
                        try:
                            min_pos=float(min(norm_pos[~torch.isnan(norm_pos)]))
                        except:
                            min_pos=1e-6
                        post_ten=torch.nan_to_num(norm_pos,nan=min_pos-1e-4)

                        if isinstance(explanations[num]['explanation'][method][1],list):
                            explanations[num]['explanation'][method][1] = torch.tensor(explanations[num]['explanation'][method][1])
                        norm_claim=norm(explanations[num]['explanation'][method][1])
                        try:
                            min_claim=float(min(norm_claim[~torch.isnan(norm_claim)]))
                        except:
                            min_claim=1e-6
                        claim_ten=torch.nan_to_num(norm_claim,nan=min_claim-1e-4)
                        
                        xai_tensor=(post_ten,claim_ten)
                        expl['explanation'][method] = xai_tensor
                expl['claim'] =  explanations[num]['claim']
                expl['post'] =  explanations[num]['post']
                list_exp.append(expl)
            return list_exp
        
        @staticmethod
        def save_json(explanations_path,cr_explanations,method):
            """
            Save created explanations during creation of explanations
            """
            if cr_explanations:
                for one_exp in cr_explanations:
                    one_exp['explanation'][method]=(one_exp['explanation'][method][0].tolist(),one_exp['explanation'][method][1].tolist())
                try:
                    with open(explanations_path, 'r',encoding='utf-8') as file:
                        existing_data = json.load(file)
                except Exception as e:
                    print(e)
                    print('Create or load json with explanations')
                existing_data.extend(cr_explanations)
                with open(explanations_path, 'w',encoding='utf-8') as file:
                    json.dump(existing_data, file, indent=4,ensure_ascii=False)
        
        def set_model(self,method,model_param):
            """
            Took model path and load specific architecture of model based on method needs
            """
            def hook_fn(module, input, output):
                print(f"Layer: {module.__class__.__name__}, Output Shape: {output.shape}")
    
            adjust_model_param={}
            model=None

            # if 'e5' in self.model_path:
            #     setup= STS_ExplainWrapper.setup_t5_transformer()
            # else:
            #     setup= STS_ExplainWrapper.setup_t5_transformer()          
            
            if 'implemented_method' in model_param:
                    if method=='Occlusion_word_level':
                        model_cap= STS_ExplainWrapper.setup_transformer(self.model_path,self.embeddings_module_name)
                    else:
                        model_cap=SentenceTransformerToHF(self.model_path).to(utils.get_device()).eval()

            if 'layer' in model_param:
                    model=STS_ExplainWrapper.setup_transformer(self.model_path,self.embeddings_module_name,interpretable_embeddings=True)
                    # adjust_model_param['layer']=[]
                    for layer in model_param['layer']:
                            _, layer_str = layer.rsplit(".", 1)
                            get_layer=getattr(model,layer_str)
                            if callable(get_layer):
                                get_layer=get_layer()
                            get_layer=model.register_forward_hook(hook_fn)
                            adjust_model_param['layer']=get_layer
                    #         adjust_model_param['layer'].append(get_layer)
                    # del model_param[method]['layer']

            for string_mod_par,values in model_param.items():
                    if isinstance(values,dict):
                        if "function_name" in values.keys():
                            module_name, function_name = values["function_name"].rsplit(".", 1)
                            module = importlib.import_module(module_name)
                            retrieved_function = getattr(module, function_name)
                            if 'parameters' in values:
                                parameters=values['parameters']
                                adjust_model_param[string_mod_par]=retrieved_function(**parameters)
                            else:
                                adjust_model_param[string_mod_par]=retrieved_function()
                    elif string_mod_par != 'layer': 
                        adjust_model_param[string_mod_par]=values


            if model == None and 'implemented_method' not in model_param: 
                    model=STS_ExplainWrapper.setup_transformer(self.model_path,self.embeddings_module_name,interpretable_embeddings=True)
            
            if 'implemented_method' not in model_param:
                    method_w_gaps = method.replace(" ", "")
                    method_att = getattr(a, method_w_gaps) 
                    model_cap = method_att(model,**adjust_model_param)
            return model_cap,model_param

        # def check_json_id(self,number,ids,explanations):
        #     if explanations:
        #         if tuple(explanations[number]['ids']) == ids:
        #             return explanations[number]
        #         else:
        #             for exp in explanations:
        #                 if tuple(exp['ids']) == ids: 
        #                     return exp
        #         return None 
        def exp_implemented_met(self,post:str,claim:str,method:str,model,method_param:dict,model_param:dict) -> list:
                """
                Compute explanations for implemented methods
                """
                def foward_fun(enc):
                    try:
                        if list(enc['input_ids'].size())[1]>512:
                            for key in enc:
                                enc[key]=enc[key][:, :512]
                            with torch.no_grad():
                                emb = model(**enc)[0]
                                forward_function = partial(semantic_search_forward_function, embedding=emb)
                                return forward_function
                    except:
                        print('')
                def com_simil(model, tokenizer, text, method,forward_function): # here need to change 
                    cls_method=globals()[method]
                    if method== 'GAE_Explain':
                        module_paths_to_hook = ["hf_transformer.encoder.layer.*.attention.self.dropout"]
                        explain_class = cls_method(module_paths_to_hook,apply_normalization=False)
                    if method=='ConservativeLRP':
                        store_A_path_expressions = ["hf_transformer.embeddings"]
                        attent_path_expressions = ["hf_transformer.encoder.layer.*.attention.self.dropout"]
                        norm_layer_path_expressions = ["hf_transformer.embeddings.LayerNorm","hf_transformer.encoder.layer.*.attention.output.LayerNorm","hf_transformer.encoder.layer.*.output.LayerNorm"]
                        explain_class = cls_method(store_A_path_expressions, attent_path_expressions, norm_layer_path_expressions,apply_normalization=False)
                    explain_class.prepare_model(model)
                    explanation, predictions = explain_class._explain_batch(model, tokenizer, text,forward_function=forward_function)
                    explain_class.cleanup()
                    return explanation, predictions   
                try:                
                    tokenizer = get_tokenizer(model)
                    claim_enc = tokenizer(claim, return_tensors="pt").to(utils.get_device())
                    post_enc = tokenizer(post, return_tensors="pt").to(utils.get_device())
                    if not method== 'Occlusion_word_level':
                        forward_function=foward_fun(claim_enc)
                        post_explanation, _=com_simil(model, tokenizer, post,method,forward_function,method_param)
                        forward_function=foward_fun(post_enc)
                        claim_explanation, _=com_simil(model, tokenizer, claim,method,forward_function,method_param)
                    else:
                        cls_method=globals()[method]
                        occl_class= cls_method(tokenizer=tokenizer,model=model,forward_func=model.forward_tokens,**method_param[method]['parameters']) 
                        post_explanation,claim_explanation = occl_class.post_claim_occlusion(post,claim)
                    print('\n')
                    print(f'[post]:{post}')
                    print(f'[claim]:{claim}')
                    print('\n')
                    explanation={}
                    explanation['explanation']={method:(post_explanation,claim_explanation)} 
                    explanation['post']=post
                    explanation['claim']=claim
                except Exception as e:
                    print(e)
                return [explanation]
            # importance_map[0]['explanation'][method]
                        
        def compute_explanation(self,method_param:dict,model_param:dict,method:str)->list:
            """
            Computes an explanation based on the given method and model parameters.
            Returns: explanations (list) and specific architecture of model based on method 
            """
            explanations=[]
            created_explanations=[]
            explain_wrappers = []

            unpack_model_param=model_param
            if method in model_param:
                unpack_model_param=model_param[method]
            model_cap,unpack_model_param = self.set_model(method,unpack_model_param)
        # claim,post,ids= self.dataset.getitem_with_ids(i)
        # post_split=post.split()
        # c=[]
        # a=0
        # for word in post_split:
        #     tokenized_word=tokenizer(word)
        #     list_tok=tokenizer.convert_ids_to_tokens(tokenized_word['input_ids'])
        #     list_tok = [tok for tok in list_tok if tok not in tokenizer.all_special_tokens]
        #     a+=len(list_tok)
        #     c.append(list_tok)
        #     print(list_tok)

            #put evaluation object 
            if 'implemented_method' not in unpack_model_param:
                explain_wrappers.append(ExplanationExecuter(model_cap,**method_param[method],visualize_explanation=False,apply_normalization=False))
            if len(self.dataset) == 0: 
                raise ValueError('Dataset is empty. Please load the dataset.')
            # creating explantions
            explanations_already_exist =Hyper_optimalization.load_explanations(self.explanations_path,method_param,model_param,method,explanations)
            id_list=Hyper_optimalization.load_explanations_ids(explanations_already_exist)
            for i in range(0,len(self.dataset)):
                claim,post,ids= self.dataset.getitem_with_ids(i)
                if id_list:
                    if list(ids) in id_list:
                        index_explanation=id_list.index(list(ids))
                        explanations.append(explanations_already_exist[index_explanation])
                        continue
                if 'implemented_method' in unpack_model_param:
                    importance_map=self.exp_implemented_met(post,claim,method,model_cap,method_param,model_param)
                    tokenizer=get_tokenizer(model_cap)
                else: 
                    importance_map=compare_multiple_explanation_methods(explain_wrappers, post, claim, additional_attribution_kwargs= {}, method_names=[method],visualize=False)
                    tokenizer=get_tokenizer(model_cap.forward_func)
                importance_map[0]['ids']=ids
                importance_map[0]['method_param']=method_param
                importance_map[0]['model_param']=model_param
                # Delete special tokens
                new_att=()
                for id_, exp in enumerate(importance_map[0]['explanation'][method]):
                    if id_==0:
                        text=post
                    if id_ ==1:
                        text=claim
                    id_set=set()
                    text=tokenizer(text)
                    for special_token in tokenizer(tokenizer.all_special_tokens)['input_ids']:
                        ids= [i for i, x in enumerate(text['input_ids'] ) if x == special_token[1]]
                        id_set.update(ids)
                    id_special_token=list(id_set)
                    try:
                        if id_special_token:
                            id_special_token.reverse()
                            token_attributions=exp[0].tolist()
                            for d in id_special_token:
                                if d > len(token_attributions):
                                    token_attributions.pop(511)
                                else: 
                                    token_attributions.pop(d)

                            new_att = new_att+ (torch.tensor(token_attributions).unsqueeze(dim=0),)
                    except Exception as e:
                        print(e)
                importance_map[0]['explanation'][method]=new_att
                #adjusting_post
                post=tokenizer(post)
                post=tokenizer.convert_ids_to_tokens(post['input_ids'])
                post = [tok for tok in post if tok not in tokenizer.all_special_tokens]
                importance_map[0]['post']=[post]
                #adjusting_claim
                claim=tokenizer(claim)
                claim=tokenizer.convert_ids_to_tokens(claim['input_ids'])
                claim=[tok for tok in claim if tok not in tokenizer.all_special_tokens]
                importance_map[0]['claim']=[claim]
                # if 'e5' in self.model_path:  # here need to change 
                #     importance_map[0]['explanation'][method]=(importance_map[0]['explanation'][method][0][:,1:-1],importance_map[0]['explanation'][method][1][:,1:-1])
                # if 'T5' in self.model_path:  # here need to change 
                #     importance_map[0]['explanation'][method]=(importance_map[0]['explanation'][method][0][:,:-1],importance_map[0]['explanation'][method][1][:,:-1])
                explanations.extend(importance_map)
                created_explanations.extend(importance_map)
            if created_explanations:
                self.save_json(self.explanations_path,copy.deepcopy(created_explanations),method)
            return explanations,model_cap
        
        def take_param(self,method:str)->dict:
            """
            Function set dictionaries with parameters and if parameters are not given than will be retrieved params in this function
            Returns: dictionaries with parameters 
            """
            if self.model_param==None: 
                function_name = "captum.attr._core.lime.get_exp_kernel_similarity_function"
                parameters = {"distance_mode": "euclidean", "kernel_width": 450}
                exp_eucl_distance = {
                    "function_name": function_name,
                    "parameters": parameters,
                }
                parameters = {"distance_mode": "euclidean", "kernel_width": 750}
                exp_eucl_distance_2 = {
                    "function_name": function_name,
                    "parameters": parameters,
                }

                function_name = "captum._utils.models.linear_model.SkLearnLasso"
                parameters = {"alpha": 1e-19}
                interpre_mod = {
                    "function_name": function_name,
                    "parameters": parameters,
                }

                parameters = {"alpha": 1e-10}
                interpre_mod_2 = {
                    "function_name": function_name,
                    "parameters": parameters,
                }
                model_param={'Lime': {'similarity_func': [exp_eucl_distance,exp_eucl_distance_2], 'interpretable_model':[interpre_mod,interpre_mod_2]}}
            else:
                model_param=copy.deepcopy(self.model_param)
            if self.method_param==None:
                method_param={'Lime':{"parameters":{"n_samples":[80,110,50,100]}, "token_groups_for_feature_mask": True},       ### Lime
                'Saliency':{"parameters":{'abs':[True,False]}},                                                                                                ### Saliency
                'Occlusion':{"parameters":{"sliding_window_shapes":[(3,1024),(5,1024)]},"compute_baseline": True},            ### Occlusion
                'Input X Gradient':{},                                                                            ### Input X Gradient
                'Guided Backprop':{},                                                                            ### Guided Backprop
                'Deconvolution':{},                                                                            ### Deconvolution
                'GradientShap':{ "compute_baseline": True },                                                  ### GradientShap
                'Kernel Shap':{"token_groups_for_feature_mask": True }}                                      ### KernelShap
            else:
                method_param=copy.deepcopy(self.method_param)

            model_param= {method: model_param[method]} if method in model_param else {}
            method_param= {method: method_param[method]} if method in method_param else {}
            return method_param,model_param
        @staticmethod
        def empty_string_baseline(model_path,embeddings_module_name,indexes_xai,dataset):
            print('Computing empty baseline')
            baseline_list=[]
            for i in range(0,len(indexes_xai)):
                claim, _= dataset[i]
                print(claim)
                model=STS_ExplainWrapper.setup_transformer(model_path,embeddings_module_name)
                enc_ctx = model.model.preprocess_input(claim) #
                enc_text = model.model.preprocess_input('')
                with torch.no_grad():
                    emb_ctx = model.model._forward(enc_ctx)
                    baseline = model(enc_text["input_encodings"]["input_ids"], "input_ids", emb_ctx, additional_forward_kwargs=enc_text)
                    baseline = (baseline + 1) / 2
                    baseline_list.append(baseline)

            return sum(baseline_list)/len(baseline_list)

                

        def set_computations(self,model,explanations,comparison:Compare_docano_XAI) -> pd.DataFrame:
            """
            Set computations of explanation thought whole dataset 
            """
            if not len(comparison.importance_map[0]['post'][0]) == len(explanations[0]['post'][0]):
                comparison.importance_map = explanations
            try:
                for number in range(0,len(self.rationale_data)):
                        df=comparison.compare(number,model=model,     
                                # best_percent=perc,
                                # evaluation_metrics=['overall_percentage','positive_percentage','TP','TN','FP','FN','recall','precision','accuracy','F1','dice_coe','jac_coe'],
                                # normalization=norm
                                )
                return df
            except Exception as e:
                print(e)
        @staticmethod
        def filter_evaluation(df:pd.DataFrame,list_ev:list)-> dict:
            """
            Select columns from dataframe with metrics 
            """
            dict_ev_values={}
            for value in list_ev:
                df_acc = df.filter(like=value).copy()
                print(df_acc)
                df_acc.replace(0, pd.NA, inplace=True)
                ev_values = df_acc.mean(skipna=True)
                ev_values=float(ev_values.iloc[0])
                dict_ev_values[value]=ev_values
            return dict_ev_values
        @staticmethod
        def choose_best(trials: list): 
            """
            Choose best trial from hyperoptimalization
            """
            best=0
            best_trial=0
            for trial in trials:
                if best < sum(trial.values):
                    best=sum(trial.values)
                    best_trial=trial
            return best_trial
        
        @staticmethod
        def choose_trials(trial:Trial,method:str,method_param:dict,model_param:dict):
            if method_param:
                if 'parameters' in method_param[method]: 
                    for param,values in method_param[method]['parameters'].items():
                        try:
                            if type(values) == tuple and type(values[0]) != str: 
                                assert len(values)==3,'For tuple you need to select 3 numbers (low,high,{step=int})'
                                if isinstance(values[1], int):
                                    method_param[method]['parameters'][param] = trial.suggest_int(f'{method}-{param}',values[0],values[1],**values[2])
                                elif isinstance(values[1],float):
                                    method_param[method]['parameters'][param] = trial.suggest_float(f'{method}-{param}',values[0],values[1],**values[2])
                                else: 
                                    raise TypeError("Selecting of parameters are not successfull")
                            else:
                                method_param[method]['parameters'][param] = trial.suggest_categorical(f'{method}-{param}',values)
                        except Exception as e:
                            print(e)
                    print(method_param[method]['parameters'])
            if model_param:
                if method == 'Lime':
                    def loop_over_hyperparam(all_params,values):
                        k = list(values.keys())
                        v= list(values.values())
                        counter=0
                        for key,value in zip(k,v):
                            if isinstance(value,str):
                                all_params.append({key:one_par})
                            if isinstance(value,list):
                                for one_par in value:
                                    all_params.append({key:one_par})
                            if isinstance(value,dict):
                                normalized_grid = {k: v if isinstance(v, list) else [v] for k, v in value.items()}
                                keys = normalized_grid.keys()
                                values_product = itertools.product(*normalized_grid.values())
                                combinations = [{'parameters':dict(zip(keys, values))} for values in values_product]
                                all_params.extend(combinations)
                        grouped = defaultdict(list)
                        for d in all_params:
                            # Use a frozenset of keys to define a type
                            key_signature = frozenset(d.keys())
                            grouped[key_signature].append(d)

                        # Step 2: Generate combinations (Cartesian product across groups)
                        groups = list(grouped.values())

                        # Ensure there's more than 1 group, or product won't make sense
                        if len(groups) > 1:
                            combinations = []
                            for combo in product(*groups):
                                # Merge dictionaries in the combination
                                merged = {}
                                for d in combo:
                                    merged.update(d)
                                combinations.append(merged)
                        return combinations


                    for param,values in model_param[method].items():
                        list_all_param=[]
                        list_all_param=loop_over_hyperparam(list_all_param,values)

                        model_param[method][param] =trial.suggest_categorical(f'{method}-{param}',list_all_param)


            return trial,method_param,model_param


        def objective(self,trial):
            """
            Choose method, parameters of method and normalization, compute explanations and evaluate explanations. 
            Returns: result of hyperoptimalization based on self.multi_objective return 1 or more numbers   
            """
            method=trial.suggest_categorical('method', self.methods)
            normalization = trial.suggest_categorical('normalization', self.normalizations)
            # if not isinstance(self.explanations_path, str):  # toto sa vymaze
            method_param,model_param=self.take_param(method)
            trial,method_param,model_param= Hyper_optimalization.choose_trials(trial,method,method_param,model_param)
            if model_param:
                print(model_param[method])
            print(normalization)
            print(method)
            for previous_trial in trial.study.trials:
                if previous_trial.state == TrialState.COMPLETE and trial.params == previous_trial.params:
                    self.counter=self.counter+1
                    self.counter_methods_dup[method] += 1
                    self.counter_normalizations_dup[normalization] += 1
                    if hasattr(trial.study, 'directions'):
                        print(f"Duplicated trial: {trial.params}, return {previous_trial.values}")
                        return previous_trial.values
                    else:
                        print(f"Duplicated trial: {trial.params}, return {previous_trial.value}")
                        return previous_trial.value

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explanations,model_cap=self.compute_explanation(method_param,model_param,method)
            
            self.counter_methods[method] += 1
            self.counter_normalizations[normalization] += 1
            if hasattr(model_cap,'forward_func'):
                model=model_cap.forward_func
            else:
                model=model_cap


            explanations=explanations[:len(self.dataset.fact_check_post_mapping)]
            explanations=Hyper_optimalization.adjust_explan(explanations,method,normalization)


            comparison=Compare_docano_XAI(rationale_dataset=self.rationale_data,importance_map=explanations)
            if self.explanation_maps_sentence:
                explanations=comparison.change_token_explanation(model,self.dataset,sentence_exp=True)
            if self.explanation_maps_word:
                explanations=comparison.change_token_explanation(model,self.dataset,word_exp=True)

            if self.explanation_maps_token and self.rationale_data:
                tokenizer=get_tokenizer(model)
                explanations= Compare_docano_XAI.find_mask(self.rationale_data,explanations,self.dataset,tokenizer,method)
            if self.rationale_path:
                df = self.set_computations(model,explanations,comparison)


            if self.rationale_path:
                if self.additional_metric:
                    add_metrics=['avg_prec_score']+self.additional_metric
                else: 
                    add_metrics=['avg_prec_score']
                add_metrics=add_metrics+['localization','point_game','spareness']
                eval_values=self.filter_evaluation(df,add_metrics)


            loader = DataLoader(self.dataset, batch_size=1)
            evaluation = EvaluateExplanation(verbose=True,rationale_path=self.rationale_path)     
            unpack_model_param=model_param
            if method in model_param:
                unpack_model_param=model_param[method]


            if 'implemented_method' in unpack_model_param:   
                model_cap = STS_ExplainWrapper.setup_transformer(self.model_path,self.embeddings_module_name) # retrieve from function
                final_metric, all_metrics= evaluation.evaluate(loader,
                                                               explanation_maps=explanations,
                                                               explanation_maps_token=self.explanation_maps_token,
                                                               explanation_maps_word=self.explanation_maps_word,
                                                               explanation_maps_sentence=self.explanation_maps_sentence,
                                                               method_name=method,
                                                               visualize=True,
                                                               model_param=model_param,
                                                               method_param=method_param,
                                                               model=model_cap,
                                                            #    plausability_word_evaluation=self.plausability_word_evaluation,
                                                            #    faithfulness_word_evaluation=self.faithfulness_word_evaluation)
                                                                )
            else:
                explain = ExplanationExecuter(model_cap, compute_baseline=False, visualize_explanation=False,apply_normalization=False)
                final_metric, all_metrics= evaluation.evaluate(loader,
                                                               explain,
                                                               explanation_maps=explanations,
                                                               explanation_maps_token=self.explanation_maps_token,
                                                               explanation_maps_word=self.explanation_maps_word,
                                                               explanation_maps_sentence=self.explanation_maps_sentence,
                                                               method_name=method,
                                                               visualize=True,
                                                               model_param=model_param,
                                                               method_param=method_param,
                                                            #    plausability_word_evaluation=self.plausability_word_evaluation,
                                                            #    faithfulness_word_evaluation=self.faithfulness_word_evaluation
                                                               )
            
            add_result={}
            for tr in all_metrics['visualization']:
                tr['normalization']=normalization
                add_result.update({f"{tr['metric']}-probs":tr['probabilities']})
            #save info for plots 
            self.visualizations_metric.extend(all_metrics['visualization'])
            #save info for tables
            try:
                if self.rationale_path:
                    add_result={**add_result,**eval_values,**all_metrics['faithfulness'], **all_metrics['plausibility']}
                    all_metrics['plausibility']['avg_prec_score']=eval_values.pop('avg_prec_score')

                else:
                    add_result={**add_result,**all_metrics['faithfulness'], **all_metrics['plausibility']}

                add_result['params']=trial.params
                self.additional_results.append(add_result)
            except Exception as e:
                print(e)

            faithfullness = list(all_metrics['faithfulness'].values())
            print(all_metrics['faithfulness'])
            if len(faithfullness) == 0:
                faithfullness_mean=0
            else:
                faithfullness_mean=sum(faithfullness)/len(faithfullness)
            plausability = list(all_metrics['plausibility'].values())
            print(all_metrics['plausibility'])
            if len(plausability) == 0:
                plausability_mean=0
            else:
                plausability_mean=sum(plausability)/len(plausability)
            #single-object
            if self.additional_metric:
                final_metric=(self.faithfulness_weight * faithfullness_mean + self.plausability_weight * plausability_mean + self.additional_metric_weight * eval_values[self.additional_metric[0]]) / (self.faithfulness_weight + self.plausability_weight + self.additional_metric_weight)
            if self.multiple_object and not self.additional_metric:
                return faithfullness_mean,plausability_mean
            elif self.multiple_object and self.additional_metric:
                return faithfullness_mean,plausability_mean,eval_values[self.additional_metric[0]]
            return final_metric
        



        def run_optimization(self,sampler,n_trials,**kwargs):
            """
            It's a function which run optimalization based on given sampler from optuna. For more info: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html 
                - sampler: The sampler in string format (e.g., TPESampler).
                - **kwargs: Additional keyword arguments to be passed to the sampler.
    
            Example:
            run_optimization(sampler_cls='TPESampler', n_ei_candidates=15, seed=82)
            """
            assert sampler in ('GridSampler', 'RandomSampler','TPESampler','NSGAIISampler','BruteForceSampler','NSGAIIISampler','GPSampler')
            self.counter=0
            self.tried_params=[]
            self.counter_methods={method: 0 for method in self.methods}
            self.counter_methods_dup={method: 0 for method in self.methods}
            self.counter_normalizations= {normalization: 0 for normalization in self.normalizations}
            self.counter_normalizations_dup={normalization: 0 for normalization in self.normalizations}
            
            start = time.time()
            tracemalloc.start()
            if not self.rationale_path and self.multiple_object:
                raise ValueError("Without rationales you need to select single objective (self.multiple_object == False)")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sampler = getattr(optuna.samplers, sampler)
                sampler = sampler(**kwargs)
                if self.additional_metric and self.multiple_object:
                    study=optuna.create_study(directions=["maximize","maximize","maximize"],sampler=sampler)
                    study.optimize(self.objective, n_trials=n_trials,n_jobs=1) 
                if self.multiple_object and not self.additional_metric:
                    study=optuna.create_study(directions=["maximize","maximize"],sampler=sampler)
                    study.optimize(self.objective, n_trials=n_trials,n_jobs=1) 
                else:
                    study=optuna.create_study(directions=["maximize"],sampler=sampler)
                    study.optimize(self.objective, n_trials=n_trials,n_jobs=1) 
            
            end = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            print(f'current: {current}')
            print(f'peak: {peak}')
            current_mb = current / (1024 ** 2) 
            peak_mb = peak / (1024 ** 2) 
            number_of_trials = len(study.trials)
            time_opt=end - start
            time_opt_h=time_opt/3600


            

            class_vis=Visualization_opt(self.methods,self.normalizations,self.counter,self.counter_normalizations,self.counter_normalizations_dup,self.counter_methods,self.counter_methods_dup,self.visualizations_metric,study,number_of_trials,time_opt_h,sampler,current_mb,peak_mb,self.additional_results)
            
            self.visualizations_metric=[]
            self.additional_results=[]
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return Hyper_optimalization.choose_best(study.trials),study.trials,class_vis
        

class Visualization_opt:
    def __init__(self,methods,normalizations,counter,counter_normalizations,counter_normalizations_dup,counter_methods,counter_methods_dup,visualizations_metric,study,number_of_trials,time_opt,sampler,current,peak,additional_results):
        self.methods=methods
        self.normalizations=normalizations
        self.counter_dup=counter
        self.counter_normalizations=counter_normalizations
        self.counter_normalizations_dup=counter_normalizations_dup
        self.counter_methods=counter_methods
        self.counter_methods_dup=counter_methods_dup
        self.visualizations_metric=visualizations_metric
        self.study=study
        self.number_of_trials=number_of_trials
        self.time_opt=time_opt
        self.sampler=sampler
        self.current=current
        self.peak=peak
        self.additional_results=additional_results
        self.best=Hyper_optimalization.choose_best(study.trials)
        self.save_path=None



        """
        This class will be filled with information from Hyperoptimalization to use 3 functions which plot or create dataframe.
        Input: save_path_plot= path where results will be saved 

        best_norm:
                Compare all normalizations applies to specific method with specific paramters.
            
            Input: method_param and method
            Output: bar plot 

        table_counter:
                Give dataframe with all trials and computed metrics for each trial.

        table_sampler:
                Give dataframe with important info for sampler
            
            Output:
                methods and normalizations: How many times where selected for optimalization
                best trial: method, parameters of method, normalization and results of objective.
                sampler info: time, memory, how much trials and when find this best trial 

        visual_aopc:
                Visualization for aopc metrics.

        """


    @staticmethod    
    def plot_aopc(vis_list: list,save_path_plot:str):
        import textwrap
        # Create a single plot
        fig, ax = plt.subplots(figsize=(10, 6))
        # create lines for empty strings and 
        whole_post_cs=[]
        empty_post_cs=[]
        method_info=''
        for vis in vis_list:
            for dict_probabilities in vis['probabilities']:
                if dict_probabilities['erassing_itterations'] == 100.0:
                   whole_post_cs.append(dict_probabilities['prob'])
                if dict_probabilities['erassing_itterations'] == 0.0:
                    empty_post_cs.append(dict_probabilities['prob'])
        whole_post_cs=sum(whole_post_cs)/len(whole_post_cs)
        empty_post_cs=sum(empty_post_cs)/len(empty_post_cs)

        for vis in vis_list:
            perc_cut = [j['erassing_itterations'] + (1.1 if j['erassing_itterations'] < 0 else 1) for j in vis['probabilities'] if j['erassing_itterations'] not in [0, 100]]
            # perc_cut.insert(0,0)
            # perc_cut = [j['erassing_itterations']+1 for j in vis['probabilities'] if j['erassing_itterations'] not in [0,100]]

            probs = [j['prob'] for j in vis['probabilities'] if j['erassing_itterations'] not in [0,100]] 
            # for item in vis['probabilities']:
            #     if item['erassing_itterations'] == 0 and :
            #         probs.insert(0,item['prob'])     

            # Plotting the data line
            if len(perc_cut) > 1:
                line, = ax.plot(perc_cut, probs, marker='o', linestyle='-', label=f'{vis["method"]}')
            parameters_list=[]
            if vis['normalization']:
                parameters_list.append(f"{vis['method']}-norm: {vis['normalization']}")

            if vis['method_param']:
                allowed_keys = {'token_groups_for_feature_mask','compute_baseline'}
                if not set(vis['method_param'].keys()).issubset(allowed_keys):
                    parameters_list.append(f"{vis['method']}-method_p: {vis['method_param']}")

            if vis['model_param']:
                allowed_keys = {'implemented_method'}
                if not set(vis['model_param'].keys()).issubset(allowed_keys):
                    parameters_list.append(f"{vis['method']}-model_p: {vis['model_param']}")
            wrapped_params = []
            for param in parameters_list:
                wrapped_params.append('\n'.join(textwrap.wrap(param, width=70)))
            method_info += '\n'.join(wrapped_params) + '\n\n'
            # Adding the method parameters as text
        ax.plot(perc_cut, [whole_post_cs]*len(perc_cut),linestyle='-', label=f'CS of whole posts and claims')
        ax.plot(perc_cut, [empty_post_cs]*len(perc_cut),linestyle='-', color='red', label=f'CS of empty string and claims')
        ax.text(1.05, 1, f"{method_info}", fontsize=11, ha='left', va='top', transform=ax.transAxes)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
        ax.set_ylim(empty_post_cs-0.02, whole_post_cs+0.02)
        ax.set_xlabel('Errasing itterations', fontsize=14, labelpad=10)
        ax.set_ylabel('Average of Cosine similarities', fontsize=14)
        ax.set_title(f"All Methods: {vis_list[0]['metric']}", fontsize=16)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=10)
        ax.grid(True)
        plt.subplots_adjust(right=0.8)
        if vis_list[0]['metric']== 'aopc_suff':
            x_axis_comment=['Positive','Negative']
        else:
           x_axis_comment=['Negative','Positive']
        x_min, x_max = ax.get_xlim()
        ax.text(x_min+0.1, empty_post_cs - 0.027, x_axis_comment[0], ha='left', va='top', fontsize=12)
        ax.text(x_max-0.1, empty_post_cs - 0.027, x_axis_comment[1], ha='right', va='top', fontsize=12)
        if save_path_plot:
            plt.savefig(f"{save_path_plot}/{vis_list[0]['metric']}.png")
        plt.show()
        plt.close() 






    def visual_aopc(self,save_path_plot:str):
            list_filtred_vis_suff=[]
            list_filtred_vis_com=[]
            if self.visualizations_metric: 
                for method in self.methods:
                    filtered_trials = [ trial for trial in self.study.trials if trial.params['method'] == method]
                    best_trial=Hyper_optimalization.choose_best(filtered_trials)
                    if best_trial == 0: 
                        print(f'Method {method} was not used in optimalization.')
                        continue 
                    filtered_vis= [ met for met in self.visualizations_metric if met['method'] == method and met['normalization']==best_trial.params['normalization']]
                    best_param=best_trial.params.copy()
                    if len(filtered_vis) > 2:
                        filtered_vis=filtered_vis[:2] 
                    for i,vis in enumerate(filtered_vis):
                        c_filtered_vis=copy.deepcopy(filtered_vis[i])
                        for params in ['method_param','model_param']:
                            if method in c_filtered_vis[params]:
                                c_filtered_vis[params]=c_filtered_vis[params][method]
                                if 'parameters' in c_filtered_vis[params] and not 'function_name' in c_filtered_vis[params]:
                                    c_filtered_vis[params]=c_filtered_vis[params]['parameters']
                            if not c_filtered_vis[params]:
                                c_filtered_vis[params]=''
                        if c_filtered_vis['metric']=='aopc_suff':
                            list_filtred_vis_suff.append(c_filtered_vis)
                        else: 
                            list_filtred_vis_com.append(c_filtered_vis)
                self.plot_aopc(list_filtred_vis_suff,save_path_plot)
                self.plot_aopc(list_filtred_vis_com,save_path_plot)

    def table_counter(self,save_path_plot=None):
        rows=[]
        for id,trial in enumerate(self.study.trials):
            entry = trial.params
            row = {
                'method': entry['method'],
                'normalization': entry['normalization'],
            }
            
            additional_params = {f'additional parameter {i-1}': f'{k.replace(entry["method"],"")}:{v}' for i, (k, v) in enumerate(entry.items()) if k not in ['method', 'normalization']}
            try:
                result={'final_metric':trial.value}
            except:
                ordered_keys = ['faithfulness', 'plausibility']
                result = {ordered_keys[0]: trial.values[0], ordered_keys[1]: trial.values[1]}
            for dict_add in self.additional_results:
                if dict_add['params']==trial.params:
                    add= dict_add.copy()        
                    del add['params']
                    probs = [key for key in add if '-probs' in key]
                    for prob in probs: 
                        add[prob]=[item['prob'] for item in add[prob]]
                    row.update(add)
            row.update(result)        
            row.update(additional_params)
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.applymap(lambda x: tuple(x) if isinstance(x, list) else x)
        df_final=df.groupby(df.columns.tolist(), as_index=False, dropna=False).size()
        if save_path_plot:
            df_final.to_csv(save_path_plot)
        return df_final
    
    def table_sampler(self,save_path_plot=None):
        params= {key + '_best': value for key, value in self.best.params.items()}
        if len(self.best.values)==2:
            params['faithfullness']=self.best.values[0]
            params['plausability']=self.best.values[1]
        elif len(self.best.values)==3:
            params['faithfullness']=self.best.values[0]
            params['plausability']=self.best.values[1]
            params['additional_metric']=self.best.values[2]
        else:
            params['overall_score']=self.best.values[0]
        params['best_find_at']=self.best._trial_id
        params['peak memory usage (mb)']=self.peak
        params['time (hours)']=self.time_opt
        params['number_dup']=self.counter_dup
        params['all_trials']=self.number_of_trials
        a =self.counter_normalizations | self.counter_methods | params
        df_2 = pd.DataFrame(list(a.items()), columns=['Index', self.sampler])
        if save_path_plot:
            df_2.to_csv(save_path_plot)
        return df_2
    
    # def table_dup(self,save_path_plot=None):
    #     params={'number_dup':self.counter_dup}
    #     a =self.counter_normalizations_dup | self.counter_methods_dup | params  
    #     df_2 = pd.DataFrame(list(a.items()), columns=['Index', self.sampler])
    #     if save_path_plot:
    #         df_2.to_csv(f"{save_path_plot}/{self.sampler}.csv'")
    #     return df_2
    
    def best_norm(self,method:str,method_param:dict=None):
            filtered_trials_method = [trial for trial in self.study.trials if trial.params['method'] == method]
            filtered_trials_param = []
            for trial in filtered_trials_method:
                for key, value in trial.params.items():
                    if key == f'{method}-{list(method_param.keys())[0]}' and f'{method}-{value == list(method_param.values())[0]}':
                        filtered_trials_param.append(trial)
            
            vis_dict = {}
            for trial in filtered_trials_param:
                vis_dict[trial.params['normalization']] = trial.values
            
            # Define categories based on the number of values
            if len(filtered_trials_param[0].values) == 1:
                categories = ['final_metric']
            elif len(filtered_trials_param[0].values) == 2:
                categories = ['faithfullness', 'plausability']
            elif len(filtered_trials_param[0].values) == 3:
                categories = ['faithfullness', 'plausability', 'additional_metric']
            
            x = np.arange(len(categories))  # x positions for the bars
            width = 0.2  # Reduce width to avoid overlap
            
            fig, ax = plt.subplots(figsize=(4*len(categories), 2*len(categories)))
            
            # Plot bars with proper shifting
            for i, (variable, vals) in enumerate(vis_dict.items()):
                ax.bar(x + i * width, vals, width, label=variable)  # Adjust spacing with +0.1
            
            # Adjust the x-ticks and labels
            ax.set_xlabel('Evaluation metrics')
            ax.set_ylabel('')
            ax.set_title(f'{method}-{method_param}')
            ax.set_xticks(x + width / len(categories))  # Position the x-ticks at the center of each group
            ax.set_xticklabels(categories)
            
            # Add y-ticks
            ax.y_ticks = np.arange(0, max(max(vis_dict.values())) + 0.05, 0.05)
            ax.legend(loc='upper right',bbox_to_anchor=(1, 1), borderaxespad=0.1)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()




