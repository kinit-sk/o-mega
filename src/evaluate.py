import json
from typing import Type
import numpy as np
import os
from tqdm import tqdm
from sentence_transformers.util import semantic_search
import torch
from torch.utils.data import DataLoader
from annotations import TokenConversion
from transformers import PreTrainedTokenizer
from sentence_transformers import SentenceTransformer
import pandas as pd
from architecture import SentenceTransformerToHF,get_tokenizer
from dataset import OurDataset
import utils
from explain import ExplanationExecuter_STS, ExplanationExecuter_CT,STS_ExplainWrapper
import measures as m
from measures import ExplanationWithRationale, BaseEvaluator
import warnings
import matplotlib.pyplot as plt 
from collections import defaultdict
import re

class ClaimMatchingEvaluate:
    """Perform claim matching and evaluate model in claim matching task
    
    We distinguish two types of tasks:
    PFCR: Match top claims to a post
    Inv PFCR: Match top posts to a claim

    In both cases either the posts or the claims act as a database that contains
    all of its data and find the most similar objects to the query.
    For example, in PFCR, we use all the claims found in train+test subsets, 
    however we only work with the posts of our current subset (either train or test)
    """
    def __init__(self, verbose: bool = False, perform_pfcr_task: bool = True) -> None:
        self.verbose = verbose
        self.perform_pfcr_task = perform_pfcr_task
        
    def evaluate(
        self, model: SentenceTransformer, query_dataset: OurDataset, 
        database_dataset: OurDataset = None, embedding_loadpath: str = None, 
        embedding_savepath: str = "./data/embeddings/temp",
        loader_kwargs: dict = None, top_k: list[int] = [1,3,5]
    ) -> dict[str: float]:
        if database_dataset is None and embedding_loadpath is None:
            raise ValueError("You need to specify at least one of the following parameters: 'database', 'embedding_loadpath'")
        if embedding_loadpath is None or os.path.exists(embedding_loadpath) == False:
            database_dataset.compute_embeddings(
                model, savepath=embedding_savepath, 
                compute_claims=self.perform_pfcr_task, 
                verbose=self.verbose
            )
            embedding_loadpath = embedding_savepath
            
        #load embeddings from the disk
        emb_files = sorted(os.listdir(embedding_loadpath))
        db_obj_ids = torch.tensor([int(f[:f.rfind(".")]) for f in emb_files])
        
        database_embeddings = []
        if self.verbose:
            print("Loading embeddings from the disk...")
        for file in tqdm(emb_files, disable=self.verbose is False):
                emb = np.load(os.path.join(embedding_loadpath, file))
                database_embeddings.append(emb)
        database_embeddings = torch.from_numpy(np.stack(database_embeddings)).to(utils.get_device())

        if loader_kwargs is None:
            loader_kwargs = {
                "batch_size": 64,
                "num_workers": 1
            }
        df_query_dataset = (
            query_dataset.df_posts_w_existing_fact_checks 
            if self.perform_pfcr_task 
            else query_dataset.df_fact_checks_w_existing_posts
        )
        
        ground_truth_ids = {}
        query_idx = 1 if self.perform_pfcr_task else 0
        _map_query_ids = np.array([
            p[query_idx] for p in query_dataset.fact_check_post_mapping
        ])
        _map_db_ids = np.array([
            p[1 - query_idx] for p in query_dataset.fact_check_post_mapping
        ])
        for query_obj_id in df_query_dataset.index:
            idx = np.where(_map_query_ids == query_obj_id)
            ground_truth_ids[query_obj_id] = _map_db_ids[idx]
            
        ds = list(zip(df_query_dataset.index.values, df_query_dataset["content"].values))
        query_loader = DataLoader(ds, **loader_kwargs)

        num_of_matches_topN = [0 for _ in range(len(top_k))]
        if self.verbose:
            print("Performing semantic similarity search...")
        for batch in tqdm(query_loader, disable=self.verbose is False):
            ids, texts = batch

            with torch.no_grad():
                query_emb = model.encode(
                    texts,
                    batch_size = len(texts),
                    device=utils.get_device(),
                    convert_to_tensor=True
                )
                top_results = semantic_search(
                    query_emb, database_embeddings, query_chunk_size=100, 
                    corpus_chunk_size=10_000, top_k=max(top_k)
                )

                for datapoint_matches, datapoint_id in zip(top_results, ids):
                    predicted_ids = np.array([db_obj_ids[match["corpus_id"]] for match in datapoint_matches])
                    gt_ids = ground_truth_ids[int(datapoint_id)]

                    for it, k in enumerate(top_k):
                        num_of_matches_topN[it] += int(len(np.intersect1d(predicted_ids[:k], gt_ids)) > 0)
                            
        return {
            f"acc_top_{k}": num_of_matches_topN[it] / len(df_query_dataset)
            for it, k in enumerate(top_k)
        }
    
class EvaluateExplanation:
    """Compute quantitative metrics to evaluate the model explanation

    Using this class we can calculate multiple quantitative measures, found in 
    ferret library, that we use to determine the quality of explanation created by 
    specific XAI algorithm
    """
    def __init__(
        self, faithfulness_weight: int = 1, plausability_weight: int = 1, 
        faithfulness_classes: list[Type[BaseEvaluator]] = [], 
        plausability_classes: list[Type[BaseEvaluator]] = [],
        rationale_path: str = None, verbose: bool = False
    ) -> None:    
        self.faithfulness_weight = faithfulness_weight
        self.plausability_weight = plausability_weight
        self.verbose = verbose
        
        self.faithfulness_classes = faithfulness_classes
        if len(faithfulness_classes) == 0:
            self.faithfulness_classes = [
               m.AOPC_Sufficiency_Evaluation,
               m.AOPC_Comprehensiveness_Evaluation,
            #    m.TauLOO_Evaluation # TODO we dont use this metric for now as it consumes ton of GPU memory
            ]
        
        self.rationale_path = rationale_path
        self.plausability_classes = []

        self.rationale_json = None
        if rationale_path is not None:
            with open(rationale_path) as f:
                self.rationale_json = json.load(f)
            self.rationale_json_mapping = np.array(
                [(data["fact_check_id"], data["post_id"]) 
                for data in self.rationale_json
            ])
            
        if self.rationale_path:
            if len(plausability_classes) == 0:
                self.plausability_classes = [
                   m.AUPRC_PlausibilityEvaluation,
                   m.Tokenf1_PlausibilityEvaluation,
                   m.TokenIOU_PlausibilityEvaluation
                ]            
    def _process_maps(self,maps,data_loader,tokenizer,method_name):
        all_post_explanations=[]
        all_mask=[]
        for map in maps:
                all_post_explanations.append(map['explanation'][method_name][0].cpu().numpy().reshape(-1))
                if 'mask' in map:
                    all_mask.append(map['mask'])
        batches = []
        for i in range(0, len(all_post_explanations), data_loader.batch_size):
            batch = all_post_explanations[i:i + data_loader.batch_size]
            batches.append(batch)
        return batches,all_mask
    
    def evaluate(
            self, data_loader: DataLoader, 
            explain_method: ExplanationExecuter_STS|ExplanationExecuter_CT = None, 
            explain_method_additional_attribution_kwargs: dict = {},
            explanation_maps=None, 
            explanation_maps_token:bool =None,
            explanation_maps_word:bool =None,
            explanation_maps_sentence:bool =None,
            method_name=None,
            method_param=None,
            model_param=None,
            visualize=False,
            model=None,
            task:str=None,
            faithfulness_word_evaluation: bool = False,
            plausability_word_evaluation: bool = False,
    ) -> tuple[float, dict]:
        #TODO        
        # assert data_loader.batch_size == 1, "We don't support bigger batch_size than 1 due to the padding"

        # get explanations
        if explain_method is None: 
            sts_wrapper: STS_ExplainWrapper = model
            tokenizer: PreTrainedTokenizer = get_tokenizer(model)
        else: 
            tokenizer: PreTrainedTokenizer = explain_method.tokenizer
            sts_wrapper: STS_ExplainWrapper = explain_method.explain_wrapper

        if explanation_maps: 
            post_explanations,annotation = self._process_maps(explanation_maps,data_loader,tokenizer,method_name) 
            if not annotation: 
                ann=torch.tensor([])

        else:   
            post_explanations = explain_method.explain(
                    data_loader, 
                    additional_attribution_kwargs=explain_method_additional_attribution_kwargs,
                    return_post_explanation_only=True
                )
            ann=torch.tensor([])

       
        all_metrics = {
            "faithfulness": {},
            "plausibility": {},
            "visualization":[]
        }
        # compute an average of each quantitative measure throughout the whole dataset
        for ferret_cls in self.faithfulness_classes+self.plausability_classes:
            if self.verbose:
                print("Computing metric:", ferret_cls.SHORT_NAME)
            ferret_eval = ferret_cls(sts_wrapper, tokenizer)

            if explanation_maps_word:
                word_evaluation= True
                sentence_evaluation= False 
            if explanation_maps_sentence:
                sentence_evaluation=True
                word_evaluation= False
            if not explanation_maps_sentence and  not explanation_maps_word:
                sentence_evaluation=False
                word_evaluation= False                
            

            # if ferret_eval.TYPE_METRIC == "plausibility" and plausability_word_evaluation:
            #     assert type(explanation_maps_word)==list, "Explanations in word form are not created"
            #     post_explanations= post_explanations_word
            # elif ferret_eval.TYPE_METRIC == "faithfulness" and faithfulness_word_evaluation:
            #     assert type(explanation_maps_word)==list, "Explanations in word form are not created"
            #     post_explanations= post_explanations_word
            # else:
            #     post_explanations= post_explanations_token
            
            results = []
            results_stats= []
            df_all_metrics={}
            probs=[]
            offset = 0
            for (claims, posts), explain_batch in tqdm(
                zip(data_loader, post_explanations), 
                total=len(data_loader), 
                disable=self.verbose==False
            ):
                for it, (claim, post, expl) in enumerate(zip(claims, posts, explain_batch)):
                    _map = data_loader.dataset.fact_check_post_mapping[offset + it]
                    if annotation:
                        ann=annotation[offset + it]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res= self._eval_pair(ferret_eval, tokenizer,  #aopc
                            _map, (claim, post), expl,
                            word_evaluation,sentence_evaluation,ann,task
                        )
                        if isinstance(res,list):
                            probs.append(res[1])
                            res=res[0]
                    if hasattr(res,'score'):
                        if res.score.item() != 0:
                            results.append(res.score)
                            # if hasattr(res,'probs'):
                            #     probs.append(res.probs)
                    else:
                        pass
                offset += len(explain_batch)
            df = pd.DataFrame(results_stats)          
            if len(results) > 0:
                results = [tens for tens in results if tens is not None and not torch.equal(tens, torch.zeros_like(tens))]
                results = torch.hstack(results).cpu().numpy().mean()
                all_metrics[ferret_eval.TYPE_METRIC][ferret_eval.SHORT_NAME] = results
                df_all_metrics[ferret_eval.SHORT_NAME]=df
                df=None
            if probs and visualize:


                perc_dict = defaultdict(lambda: {'sum': 0, 'count': 0})
                for records in probs:
                    for record in records:
                        perc = record['erassing_itterations']
                        prob = record['prob']
                        perc_dict[perc]['sum'] += prob
                        perc_dict[perc]['count'] += 1

                result = [{'erassing_itterations': perc, 'prob': perc_dict[perc]['sum'] / perc_dict[perc]['count']} for perc in sorted(perc_dict.keys())]

                
                # max_length = max(len(sublist) for sublist in probs)
                # normalized_data = [sublist + [float('nan')] * (max_length - len(sublist)) for sublist in probs]
                # means = [np.nanmean([row[i] for row in normalized_data]) for i in range(max_length)]

                # perc_cut =  [round(i * 0.1, 1) for i in range(max_length)]
                all_metrics['visualization'].append({'method':method_name,'method_param':method_param,'model_param':model_param,'metric':res.name,'probabilities':result})
                # if ferret_cls.SHORT_NAME=='aopc_suff': 
                #     perc_cut.reverse()
                # plt.plot(perc_cut,means, marker='o', linestyle='-', color='b', label='Data Line')
                # plt.xlabel('Percentage')
                # plt.ylabel('Means')
                # plt.title(f'{ferret_cls.SHORT_NAME}--{method_param}--{model_param}')
                # plt.grid(True)
                # plt.show()
                # probs=[]


        faithfulness = np.array(list(all_metrics["faithfulness"].values())).mean()


        plausability = np.array(list(all_metrics["plausibility"].values())).mean()

        if not self.faithfulness_classes:
            return plausability, all_metrics
        if not self.plausability_classes:
            return faithfulness, all_metrics
        final_metric = (
            self.faithfulness_weight * faithfulness 
            + self.plausability_weight * plausability 
        ) / (self.faithfulness_weight + self.plausability_weight)

        return final_metric, all_metrics

    def _eval_pair(
        self, ferret_eval: BaseEvaluator, tokenizer: PreTrainedTokenizer,  
        _map: tuple[int, int], data_pair: tuple[str, str], post_explain: torch.Tensor,
        word_evaluation:bool=False,
        sentence_evaluation: bool=False,
        annotation:torch.tensor=None,
        task: str= None
        # faithfulness_word_evaluation: bool = False,
        # plausability_word_evaluation: bool = False

    ) -> torch.Tensor:
        rationale = None
        claim, post = data_pair
        # if ferret_eval.TYPE_METRIC == "plausibility":
        #     word_evaluation = plausability_word_evaluation
        # else:
        #     word_evaluation = faithfulness_word_evaluation
        if word_evaluation:
            real_input_ids_with_special = post.split()
            post_explain = post_explain[: len(real_input_ids_with_special)][None]
        if not sentence_evaluation and not word_evaluation:
            token_convert = TokenConversion(tokenizer)
            # real_input_ids_with_special = tokenizer(post, truncation=False)["input_ids"]
            real_input_ids_without_special = tokenizer(post, truncation=False, add_special_tokens=False)["input_ids"]
            post_explain = post_explain[: len(real_input_ids_without_special)][None]

        if post_explain.ndim == 2:
            post_explain = post_explain[0]

        if ferret_eval.TYPE_METRIC == "plausibility":
            if annotation.numel() == 0:
                idx = np.where(
                    self.rationale_json_mapping == _map
                )[0]
                if len(idx) == 0:
                    return None

                rationale = self.rationale_json[idx[0]]                    
                rationale = [rationale["post_rationale"]] + rationale["ocr_rationale"]

                interp_tokens = [x for i in rationale if 'tokens' in i for x in i['tokens']]
                rationale_for_interp_tokens= [x for i in rationale if 'mask' in i for x in i['mask']]


                if not word_evaluation:
                    #change rationale mask with interpretable tokens to model token size
                    input_ids = tokenizer(post, truncation=False, add_special_tokens=False)["input_ids"]
                    real_tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    token_mapping = token_convert.create_interpretable_tokens(
                        post, real_tokens=real_tokens, return_token_mapping=True
                    )

                    rationale_for_real_tokens = np.zeros(len(real_tokens))
                    for _map, interp_rationale, interp_token in zip(token_mapping, rationale_for_interp_tokens, interp_tokens):
                        # if interp_token != _map["interpretable_token"]:
                        #     return None
                        offset = _map["real_token_offset"]
                        num_tokens = len(_map["modified_real_tokens"])

                        rationale_for_real_tokens[
                            offset: offset + num_tokens
                        ] = interp_rationale
                    rationale = rationale_for_real_tokens
                else: 
                    rationale=rationale_for_interp_tokens
            else:
                rationale=annotation
            if isinstance(rationale,torch.Tensor):
                rationale = rationale.tolist()
                rationale = np.array(rationale, dtype="int")
            # TODO ignore cases when the input is longer than 512 tokens...
            if len(rationale) > 512:
                return None
        explanation_obj = ExplanationWithRationale(
            input_text=post,
            ctx_text=claim,
            scores=post_explain,
            rationale=rationale
        )
        # calculating the evaluation measure for a specific datapoint 
        return ferret_eval.compute_evaluation(explanation_obj,word_evaluation,sentence_evaluation,task,normalize=True)

# if __name__ == "__main__":
    # dataset = OurDataset(csv_dirpath="./data", split="test")

    # #smaller ds for testing purposes
    # dataset.fact_check_post_mapping = dataset.fact_check_post_mapping[:20] 
    # loader = DataLoader(dataset, batch_size=4)

    # from  explain import STS_ExplainWrapper
    # import captum.attr as a
    # model = STS_ExplainWrapper.setup_transformer(model_path="../models/GTR-T5-FT",embeddings_module_name='encoder.embed_tokens')
    # method = a.LayerGradientXActivation(model, layer=model.get_embedding_layer())
    # explain = ExplanationExecuter(method, compute_baseline=False, visualize_explanation=False)

    # evaluation = EvaluateExplanation(rationale_path="./tmp/rationale-test.json", verbose=True)
    # final_metric, all_metrics = evaluation.evaluate(loader, explain)

    # pass
