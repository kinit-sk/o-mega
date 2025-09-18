from visualization_opt import Visualization_opt
from typing import Union, Optional
import os
import json
from typing import Union
import pandas as pd
from collections import defaultdict
import torch
import os
from functools import partial
from xai import GAE_Explain, ConservativeLRP, semantic_search_forward_function,Occlusion_word_level
from explain import STS_ExplainWrapper, ExplanationExecuter_STS, ExplanationExecuter_CT,compare_multiple_explanation_methods,SentenceTransformerToHF,ClassificationWrapper
import utils
import optuna
from optuna.samplers import TPESampler, GridSampler,NSGAIISampler, NSGAIIISampler, GPSampler
from torch.utils.data import DataLoader
from evaluate import EvaluateExplanation
import captum.attr as a
from optuna.trial import TrialState,Trial
import seaborn as sns
import time 
from dataset import OurDataset,HuggingfaceDataset
import copy
import importlib
import itertools
from architecture import get_tokenizer
import warnings
import tracemalloc
import gc
from itertools import product
from functools import partial
from check_explanations import Check_docano_XAI, Compare_docano_XAI



class Hyper_optimalization:
        def __init__(self,
                    dataset:Optional[Union[OurDataset, HuggingfaceDataset]],
                    task:str,
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
                    additional_metric:list=None,
                    num_classes=None) -> None:
            tasks = ['post_claim_matching', 'text_classification']
            assert task in tasks, \
                f"Available tasks {allowed_metrics}"
            self.task= task
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
            self.num_classes=num_classes
            allowed_metrics = ['localization', 'point_game', 'spareness', 'gini']
            if additional_metric is not None:
                assert all(m in allowed_metrics for m in additional_metric), \
                    f"All additional_metric values must be in {allowed_metrics}"
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
            self.pair_results=[]
            self.final_head=self.define_final_head(task)
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
                             Multiple_object= True Optuna return 2 integers (faithfulness,plausability) 
                             Multiple_object= False Optuna make average between used types of metrics and result will be 1 integer 

            faithfulness_weight, plausability_weight: Weights for final results  
                                                      Must be multiple_object=False
                                                      Sum of values must be 1. 

            additional_metric, additional_metric_weight: Additionial computation metrics: ['localization','point_game','spareness','gini']
                                                         Sum of all 3 weights must be 1.
            """

        def process_input_dataset(self,dataset:Optional[Union[OurDataset, HuggingfaceDataset]]):
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
                if len(dataset) == 0:
                    raise ValueError("Dataset not match rationales")
                return doc_data,dataset

            else: 
                length_dataset=(self.perc/100)*len(dataset.fact_check_post_mapping)
                length_dataset=round(length_dataset)
                dataset.fact_check_post_mapping=dataset.fact_check_post_mapping[:length_dataset]
                return None,dataset 
            
        def get_wrapper_constructor(self,task: str, **kwargs):
            constructors = {
                'post_claim_matching': STS_ExplainWrapper,
                'text_classification': ClassificationWrapper
            }
            return partial(constructors[task], **kwargs)
        def define_final_head(self,task):
            if task == 'post_claim_matching':
                final_head=self.get_wrapper_constructor(task)
            if task == 'text_classification':
                final_head=self.get_wrapper_constructor(task,num_classes=self.num_classes)
            return final_head

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
        def load_explanations(explanations_path,method_param,model_param,method):
            """
            Load explanation with specific model and method parameters
            """
            matched_explanations = []
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
                        exp['explanation'][method] = tuple(
                            torch.tensor(item) for item in exp['explanation'][method]
                        )
                        matched_explanations.extend([exp])
                except Exception as e :
                    print(e)
            return matched_explanations
        
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
                        norm = getattr(Compare_docano_XAI, normalization)
                        xai_tensor = []
                        num_expl = len(explanations[num]['explanation'][method])

                        for i in range(num_expl):
                            if method in explanations[num]['explanation']:
                                # Convert list to tensor if needed
                                if isinstance(explanations[num]['explanation'][method][i], list):
                                    explanations[num]['explanation'][method][i] = torch.tensor(explanations[num]['explanation'][method][i])

                                # Normalize and handle NaN values
                                normalized = norm(explanations[num]['explanation'][method][i])
                                try:
                                    min_val = float(min(normalized[~torch.isnan(normalized)]))
                                except:
                                    min_val = 1e-6
                                processed_tensor = torch.nan_to_num(normalized, nan=min_val-1e-4)
                                xai_tensor.append(processed_tensor)

                        expl['explanation'][method] = tuple(xai_tensor)
                        # expl['explanation'][method] = xai_tensor
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
                    one_exp['explanation'][method] = tuple(x.tolist() for x in one_exp['explanation'][method])
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
            
            if 'implemented_method' in model_param:
                    if method=='Occlusion_word_level':
                        model_cap= self.final_head.func.setup_transformer(self.model_path,self.embeddings_module_name,**self.final_head.keywords)
                    else:
                        model_cap=SentenceTransformerToHF(self.model_path).to(utils.get_device()).eval()

            if 'layer' in model_param:
                    model=self.final_head.func.setup_transformer(self.model_path,self.embeddings_module_name,interpretable_embeddings=True,**self.final_head.keywords)
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
                    model=self.final_head.func.setup_transformer(self.model_path,self.embeddings_module_name,interpretable_embeddings=True,**self.final_head.keywords)
            if 'implemented_method' not in model_param:
                    method_w_gaps = method.replace(" ", "")
                    method_att = getattr(a, method_w_gaps) 
                    model_cap = method_att(model,**adjust_model_param)
            return model_cap,model_param

        def exp_implemented_met(self,post:str,claim:str,method:str,model,method_param:dict,model_param:dict,task:str) -> list:
                """
                Compute explanations for implemented methods
                """
                def tex_cla():
                    raise ValueError("Implemented methods are not able to do in Text Classification task")
                def foward_fun(enc,model_max_length:int):
                    try:
                        if list(enc['input_ids'].size())[1]>model_max_length:
                            for key in enc:
                                enc[key]=enc[key][:, :model_max_length]
                            with torch.no_grad():
                                emb = model(**enc)[0]
                                forward_function = partial(semantic_search_forward_function, embedding=emb)
                                return forward_function
                    except:
                        print('')
                def com_simil(model, tokenizer, text, method,forward_function,model_param): # here need to change 
                    cls_method=globals()[method]
                    if method== 'GAE_Explain':
                        assert 'layers' in model_param[method], "'Layers' parameter is not available"
                        explain_class = cls_method(**model_param[method]['layers'],apply_normalization=False)
                    if method=='ConservativeLRP':
                        assert 'layers' in model_param[method], "'Layers' parameter is not available"
                        # layers= {'store_A_path_expressions':["hf_transformer.embeddings"],'attent_path_expressions':['hf_transformer.encoder.layer.*.attention.self.dropout'],'norm_layer_path_expressions':["hf_transformer.embeddings.LayerNorm","hf_transformer.encoder.layer.*.attention.output.LayerNorm","hf_transformer.encoder.layer.*.output.LayerNorm"]}
                        # store_A_path_expressions = ["hf_transformer.embeddings"]
                        # attent_path_expressions = ["hf_transformer.encoder.layer.*.attention.self.dropout"]
                        # norm_layer_path_expressions = ["hf_transformer.embeddings.LayerNorm","hf_transformer.encoder.layer.*.attention.output.LayerNorm","hf_transformer.encoder.layer.*.output.LayerNorm"]
                        explain_class = cls_method(**model_param[method]['layers'],apply_normalization=False)
                    explain_class.prepare_model(model)
                    explanation, predictions = explain_class._explain_batch(model, tokenizer, text,forward_function=forward_function)
                    explain_class.cleanup()
                    return explanation, predictions   
                tokenizer = get_tokenizer(model)
                claim_enc = tokenizer(claim, return_tensors="pt").to(utils.get_device())
                post_enc = tokenizer(post, return_tensors="pt").to(utils.get_device())
                if not method== 'Occlusion_word_level' and task == 'post_claim_matching':
                    forward_function=foward_fun(claim_enc,tokenizer.model_max_length)
                    post_explanation, _=com_simil(model, tokenizer, post,method,forward_function,model_param)
                    forward_function=foward_fun(post_enc,tokenizer.model_max_length)
                    claim_explanation, _=com_simil(model, tokenizer, claim,method,forward_function,model_param)
                elif task == 'post_claim_matching':
                    cls_method=globals()[method]
                    occl_class= cls_method(tokenizer=tokenizer,model=model,forward_func=model.forward_tokens,**method_param[method]['parameters']) 
                    post_explanation,claim_explanation = occl_class.post_claim_occlusion(post,claim)
                else:
                    tex_cla()
                print('\n')
                print(f'[post]:{post}')
                print(f'[claim]:{claim}')
                print('\n')
                explanation={}
                explanation['explanation']={method:(post_explanation,claim_explanation)} 
                explanation['post']=post
                explanation['claim']=claim
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

            #put evaluation object 
            if  'implemented_method' not in unpack_model_param: 
                if self.task == 'post_claim_matching':
                    explain_wrappers.append(ExplanationExecuter_STS(model_cap,**method_param[method],visualize_explanation=False,apply_normalization=False))
                else: 
                    explain_wrappers.append(ExplanationExecuter_CT(model_cap,**method_param[method],visualize_explanation=False,apply_normalization=False))
            if len(self.dataset) == 0: 
                raise ValueError('Dataset is empty. Please load the dataset.')
            # creating explantions
            explanations_already_exist =Hyper_optimalization.load_explanations(self.explanations_path,method_param,model_param,method)
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
                    importance_map=compare_multiple_explanation_methods(explain_wrappers, 
                                                                        post, 
                                                                        claim, 
                                                                        additional_attribution_kwargs= {}, 
                                                                        method_names=[method],
                                                                        task=self.task,
                                                                        visualize=False)
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
                                    token_attributions.pop(tokenizer.model_max_length-1)
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
                if self.task == 'post_claim_matching':
                    claim=tokenizer(claim)
                    claim=tokenizer.convert_ids_to_tokens(claim['input_ids'])
                    claim=[tok for tok in claim if tok not in tokenizer.all_special_tokens]
                    importance_map[0]['claim']=[claim]
                else:
                    importance_map[0]['claim']=[[claim]]
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
            method_param= {method: method_param[method]} if method in method_param else {method:{}}
            return method_param,model_param
        @staticmethod
        def empty_string_baseline(model_path,embeddings_module_name,indexes_xai,dataset):
            print('Computing empty baseline')
            baseline_list=[]
            for i in range(0,len(indexes_xai)):
                claim, _= dataset[i]
                print(claim)
                model=STS_ExplainWrapper.setup_transformer(model_path,embeddings_module_name)  # HERE add final_head
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
            #select method his parameters by optuna
            method=trial.suggest_categorical('method', self.methods)
            normalization = trial.suggest_categorical('normalization', self.normalizations)
            method_param,model_param=self.take_param(method)
            trial,method_param,model_param= Hyper_optimalization.choose_trials(trial,method,method_param,model_param)
            if model_param:
                print(model_param[method])
            print(normalization)
            print(method)
            # sae duplicates
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
            # Crate explanations
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
                if self.additional_metric:
                    add_metrics=add_metrics+self.additional_metric
                eval_values=self.filter_evaluation(df,add_metrics)


            loader = DataLoader(self.dataset, batch_size=1)
            evaluation = EvaluateExplanation(verbose=True,rationale_path=self.rationale_path)     
            unpack_model_param=model_param
            if method in model_param:
                unpack_model_param=model_param[method]


            if 'implemented_method' in unpack_model_param:   
                model_cap = self.final_head.func.setup_transformer(self.model_path,self.embeddings_module_name,**self.final_head.keywords) # retrieve from function
                final_metric, all_metrics= evaluation.evaluate(loader,
                                                               explanation_maps=explanations,
                                                               explanation_maps_token=self.explanation_maps_token,
                                                               explanation_maps_word=self.explanation_maps_word,
                                                               explanation_maps_sentence=self.explanation_maps_sentence,
                                                               task=self.task,
                                                               method_name=method,
                                                               visualize=True,
                                                               model_param=model_param,
                                                               method_param=method_param,
                                                               model=model_cap,
                                                            #    plausability_word_evaluation=self.plausability_word_evaluation,
                                                            #    faithfulness_word_evaluation=self.faithfulness_word_evaluation)
                                                                )
            else:
                explain = ExplanationExecuter_STS(model_cap, compute_baseline=False, visualize_explanation=False,apply_normalization=False)
                final_metric, all_metrics= evaluation.evaluate(loader,
                                                               explain,
                                                               explanation_maps=explanations,
                                                               explanation_maps_token=self.explanation_maps_token,
                                                               explanation_maps_word=self.explanation_maps_word,
                                                               explanation_maps_sentence=self.explanation_maps_sentence,
                                                               task=self.task,
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
            self.pair_results.extend(all_metrics['results'])
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
            
            best=Hyper_optimalization.choose_best(study.trials)

            class_vis=Visualization_opt(self.methods,
                                        self.normalizations,self.counter,
                                        self.counter_normalizations,
                                        self.counter_normalizations_dup,
                                        self.counter_methods,
                                        self.counter_methods_dup,
                                        self.visualizations_metric,
                                        study,
                                        number_of_trials,
                                        time_opt_h,
                                        sampler,
                                        current_mb,
                                        peak_mb,
                                        self.additional_results,
                                        best,
                                        self.pair_results)
            
            self.visualizations_metric=[]
            self.additional_results=[]
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            return best,study.trials,class_vis
  