import json
from typing import Union, Optional
from dataset import OurDataset,HuggingfaceDataset
import re
import numpy as np
from typing import Union
import pandas as pd
import torch
import ast
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
import utils
import seaborn as sns
from architecture import get_tokenizer
from explain import STS_ExplainWrapper,ClassificationWrapper


class Check_docano_XAI:  
    def __init__(self,rationale_dataset_path: str,xai_dataset:Optional[Union[OurDataset, HuggingfaceDataset]],tuple_indexes: tuple[tuple]) -> None:
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
            if len(post_claim_pair['ocr_rationale']) == 1:
                ocr=post_claim_pair['ocr_rationale'][0]
            else:
                ocr = {
                    'interpretable_tokens': post_claim_pair['ocr_rationale'][0]['interpretable_tokens'],
                    'tokens': [],
                    'mask': []
                }
                for item in post_claim_pair['ocr_rationale']:
                    ocr['tokens'].extend(item['tokens'])
                    ocr['mask'].extend(item['mask'])


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
        if len_tens[0] > tokenizer.model_max_length:
           adjusted_tensor=adjusted_tensor[:tokenizer.model_max_length]
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


    def change_token_explanation(self,model:Optional[Union[STS_ExplainWrapper,ClassificationWrapper]],dataset:Optional[Union[HuggingfaceDataset, OurDataset]],sentence_exp:bool=False,word_exp:bool=False) -> pd.DataFrame:    
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
            if list(mask.size())[0] > tokenizer.max_len_single_sentence: 
                mask = mask[:tokenizer.max_len_single_sentence]
            if explanations[i]['explanation'][method][0].squeeze().size()[0] != mask.size()[0]:
                # if mask.size()[0] < explanations[i]['explanation'][method][0].squeeze().size()[0]
                print(f"{explanations[i]['explanation'][method][0].squeeze().size()[0]}:{mask.size()[0]}")
            explanations[i]['mask']=mask
        return explanations



    def compare(self,index_doccano: int,model,      
                sig_number=None,best_percent=None,         
                percentage_metrics=None,
                evaluation_metrics=None,
                additional_metrics=[],
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
                    if torch.isnan(xai_tensor).any():
                        min_val = xai_tensor[~torch.isnan(xai_tensor)].min()
                        xai_tensor = torch.nan_to_num(xai_tensor,nan = min_val-1e-10)
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

                if 'gini' in additional_metrics:
                    try:
                        gini = float(torch.sum((2 * indices - n - 1) * sorted_tensor) / (n * torch.sum(sorted_tensor)))
                        row[f'gini-{method}']=gini
                    except Exception as e:
                        gini=0
                        row[f'gini-{method}']=gini
                        print(e)
                if 'spareness' in additional_metrics:
                    spare=Compare_docano_XAI.gini_f(xai_tensor)
                    row[f'spareness-{method}']=spare
                if 'localization' in additional_metrics:
                    local=Compare_docano_XAI.localization(xai_tensor,mask)
                    row[f'localization-{method}']=local
                if 'point_game' in additional_metrics:
                    points=Compare_docano_XAI.point_game_f(xai_tensor,mask)
                    row[f'point_game-{method}']=points
            row['len_sentence'] = mask.size(dim=0) 
            self.df_score = pd.concat([self.df_score, pd.DataFrame([row])], ignore_index=True)   
        except Exception as e :
            print (e)
        return self.df_score
