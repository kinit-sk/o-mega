from __future__ import annotations

import numpy as np
from typing import Literal, Union, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from captum.attr import visualization as viz
from captum.attr import Attribution
import matplotlib.pyplot as plt
import utils
from architecture import (
    SentenceTransformerToHF, RepresentationModel, 
    Basic_RepresentationModel, HierarchicalLM_RepresentationModel
)

        
class STS_ExplainWrapper(torch.nn.Module):
    """Wrapper used for computing text similarity
    
    This implementation expects one of the embeddings, we use to calculate similarity,
    to be already precomputed so that the possible backprop can only be distributed to 
    one of the inputs

    In other words, we usually precompute the embeddings of claims, since we dont want to 
    retrieve the relevance of individual tokens of the claim input, but rather want to focus
    only on the post content instead
    """
    def __init__(self, model: RepresentationModel,**kwargs) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, inputs: Union[torch.Tensor, tuple[torch.Tensor]], 
        input_field_name: str,
        secondary_embedding: Optional[torch.Tensor] = None, 
        additional_forward_kwargs: Optional[dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if additional_forward_kwargs is None:
            additional_forward_kwargs = {}
        encodings = additional_forward_kwargs.copy()     
        encodings["input_encodings"][input_field_name] = inputs  
        primary_embedding = self.model._forward(encodings)
        return torch.nn.functional.cosine_similarity(primary_embedding, secondary_embedding)


    def forward_tokens(
        self,
        primary_input: torch.Tensor,
        secondary_embedding: torch.Tensor
    ) -> torch.Tensor:
        
        primary_input = primary_input.copy()
        primary_embedding = self.model.forward_tokens(primary_input)
        return torch.nn.functional.cosine_similarity(primary_embedding, secondary_embedding)

    def get_embedding_layer(self) -> torch.nn.Module:
        module = self.model.transformer.hf_transformer
        module_path = self.model.transformer.embeddings_module_name

        for mod_name in module_path.split("."):
            module = getattr(module, mod_name)
        return module

    @staticmethod    
    def setup_transformer(
        model_path: str,embeddings_module_name:str,interpretable_embeddings: bool = False
    ) -> STS_ExplainWrapper:
        # if interpretable_embeddings:
        #     raise ValueError("We have disabled the option to use InterpretableEmbeddingBase wrapper")

        transformer = SentenceTransformerToHF(
            model_path, interpretable_embeddings=interpretable_embeddings,
            embeddings_module_name=embeddings_module_name
        )
        model = Basic_RepresentationModel(
            transformer, tokenizer=transformer.tokenizer, pooling="none"
        )
        model.to(utils.get_device())
        model.eval()

        return STS_ExplainWrapper(model)

    @staticmethod    
    def setup_t5_transformer(
        model_path: str, interpretable_embeddings: bool = False
    ) -> STS_ExplainWrapper:
        # if interpretable_embeddings:
        #     raise ValueError("We have disabled the option to use InterpretableEmbeddingBase wrapper")

        transformer = SentenceTransformerToHF(
            model_path, interpretable_embeddings=interpretable_embeddings,
            embeddings_module_name='encoder.embed_tokens'
        )
        model = Basic_RepresentationModel(
            transformer, tokenizer=transformer.tokenizer, pooling="none"
        )
        model.to(utils.get_device())
        model.eval()

        return STS_ExplainWrapper(model)
    
    @staticmethod    
    def setup_e5_transformer(
        model_path: str, interpretable_embeddings: bool = False
    ) -> STS_ExplainWrapper:
        # if interpretable_embeddings:
        #     raise ValueError("We have disabled the option to use InterpretableEmbeddingBase wrapper")

        transformer = SentenceTransformerToHF(
            model_path, interpretable_embeddings=interpretable_embeddings,
            embeddings_module_name='embeddings.word_embeddings'
        )
        model = Basic_RepresentationModel(
            transformer, tokenizer=transformer.tokenizer, pooling="none"
        )
        model.to(utils.get_device())
        model.eval()

        return STS_ExplainWrapper(model)

    @staticmethod
    def setup_hierarchical_e5_transformer(
        chunk_pooling: str = "mean", max_supported_chunks: int = 11,
        interpretable_embeddings: bool = False
    ) -> STS_ExplainWrapper:
        # if interpretable_embeddings:
        #     raise ValueError("We have disabled the option to use InterpretableEmbeddingBase wrapper")

        transformer = SentenceTransformerToHF(
            "intfloat/multilingual-e5-large", 
            interpretable_embeddings=interpretable_embeddings,
            embeddings_module_name="embeddings"
        )
        model = HierarchicalLM_RepresentationModel(
            transformer, 
            tokenizer=transformer.tokenizer,
            token_pooling="none",
            chunk_pooling=chunk_pooling,
            max_supported_chunks=max_supported_chunks
        )
        model.to(utils.get_device())
        model.eval()

        return STS_ExplainWrapper(model)
        
class ClassificationWrapper(torch.nn.Module):
    """Wrapper for text classification."""
    def __init__(
        self, 
        model: RepresentationModel, 
        num_classes: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.model = model
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.LazyLinear(num_classes)
        )

    def forward(
        self, 
        inputs, 
        input_field_name: str,
        additional_forward_kwargs: Optional[dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        if additional_forward_kwargs is None:
            additional_forward_kwargs = {}
        encodings = additional_forward_kwargs.copy()     
        encodings["input_encodings"][input_field_name] = inputs  
        primary_embedding = self.model._forward(encodings)
        logits = self.classifier(primary_embedding)
        return torch.softmax(logits, dim=-1)
    @staticmethod
    def setup_transformer(
        model_path: str,
        embeddings_module_name: str,
        num_classes: int,  
        pooling: str = "none", 
        dropout: float = 0.1, 
        interpretable_embeddings: bool = False
    ) -> torch.nn.Module:  
        transformer = SentenceTransformerToHF(
            model_path,
            interpretable_embeddings=interpretable_embeddings, 
            embeddings_module_name=embeddings_module_name
        )
        model = Basic_RepresentationModel(
            transformer,
            tokenizer=transformer.tokenizer,
            pooling=pooling 
        )
        model.to(utils.get_device())
        model.eval()
        return ClassificationWrapper(
            model,
            num_classes=num_classes,
            dropout=dropout
        )


    
def postprocess_explanations(expl: torch.Tensor,apply_normalization, normalization_approach) -> torch.Tensor:
    expl = expl.sum(dim=-1)
    
    if apply_normalization and normalization_approach == "min-max":
        for i in range(len(expl)):
            max_r = torch.quantile(expl, 0.95)
            min_r = torch.quantile(expl, 0.05)
            expl[i] = (expl[i] - min_r) / (max_r - min_r + 1e-9)
            expl[i] = torch.clip(expl[i], min=0, max=1)
    elif apply_normalization and normalization_approach == "l2":
        expl = expl / torch.norm(expl, p=2)
    elif apply_normalization and normalization_approach == "log-min-max":
            log_tensor = torch.log1p(expl)
            max_log = torch.quantile(expl, 0.95)
            min_log = torch.quantile(expl, 0.05)
            expl = (log_tensor - min_log) / (max_log - min_log + 1e-9)
            expl = torch.clip(expl, min=0, max=1)
    elif apply_normalization and normalization_approach == "max-abs":
        for i in range(len(expl)):
            max_abs_r = torch.quantile(expl.absolute(), 0.95)
            expl[i] = expl[i] / (max_abs_r + 1e-9)
            expl[i] = torch.clip(expl[i], min=-1, max=1)
    elif apply_normalization:
        return expl
    return expl


class ExplanationExecuter_STS:
    """Class that takes a model (STS_ExplainWrapper) and XAI attribution method from captum framework 
    and computes the explanation
    Work with post-claim matching
    """
    def __init__(
        self, method: Attribution, compute_baseline: bool = False, 
        apply_normalization: bool = True, 
        normalization_approach: Literal["min-max",'log-min-max', "l2", "max-abs"] = "log-min-max",
        token_groups_for_feature_mask: bool = False,
        visualize_explanation: bool = False, verbose: bool = True, parameters: dict = {}

    ) -> None:
        self.explain_method = method
        self.compute_baseline = compute_baseline
        self.apply_normalization = apply_normalization
        self.normalization_approach = normalization_approach
        self.visualize_explanation = visualize_explanation
        self.verbose = verbose
        self.explain_wrapper: STS_ExplainWrapper = method.forward_func
        self.model_wrapper: RepresentationModel = self.explain_wrapper.model
        self.encoder: SentenceTransformerToHF = self.model_wrapper.transformer

        self.tokenizer = self.encoder.tokenizer
        self.interpretable_emb_layer = self.encoder.interpretable_emb_layer
        self.token_groups_for_feature_mask = token_groups_for_feature_mask
        self.parameters=parameters

    def explain(
        self, data_loader: DataLoader, additional_attribution_kwargs: dict = {}, 
        return_post_explanation_only: bool = True
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]: 
        all_post_explanations = []
        all_claim_explanations = []
        
        for batch in tqdm(data_loader, disable=self.verbose==False):
            claims, posts = batch

            predictions, explanations, encodings = self.explain_batch(
                posts, claims, additional_attribution_kwargs, 
                return_one_explanation=return_post_explanation_only
            )
                
            post_explanation, claim_explanation = explanations, None
            if return_post_explanation_only == False:
                post_explanation, claim_explanation = explanations
                all_claim_explanations.append(claim_explanation.cpu().numpy())
            all_post_explanations.append(post_explanation.cpu().numpy())
            
            if self.visualize_explanation:
                enc_post, enc_claim = encodings
                post_tokens = [
                    self.tokenizer.convert_ids_to_tokens(input_ids) 
                    for input_ids in enc_post["input_encodings"]["input_ids"]
                ]
                claim_tokens = [
                    self.tokenizer.convert_ids_to_tokens(input_ids) 
                    for input_ids in enc_claim["input_encodings"]["input_ids"]
                ]
                self._vizualize_pair(
                    post_explanation,
                    claim_explanation,
                    post_tokens,
                    claim_tokens,
                    posts,
                    claims,
                    predictions,
                )   

        if return_post_explanation_only:
            return all_post_explanations
        return all_post_explanations, all_claim_explanations

    def explain_batch(
        self, posts: tuple[str], claims: tuple[str], 
        additional_attribution_kwargs: dict = None, 
        return_one_explanation: bool = False
    ) -> tuple[
        torch.Tensor, 
        Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], 
        tuple[dict, dict]
    ]:
        # preprocessing 
        inputs = self._preprocess_batch_inputs(posts, claims)
        enc_posts, enc_claims = inputs["enc_posts"], inputs["enc_claims"]

        emb_posts, emb_claims, predictions = self._precompute_embeddings(enc_posts, enc_claims)
        post_attribute_kwargs, claim_attribute_kwargs = self._aggregate_attribute_kwargs(
            inputs, emb_posts, emb_claims,
            additional_attribution_kwargs
        )
        
        #calculating relevance and further postprocessing

        # self.explain_method.forward_func.zero_grad()
        post_explanation = self.explain_method.attribute(**post_attribute_kwargs,**self.parameters).detach()
        post_explanation = postprocess_explanations(post_explanation,self.apply_normalization,self.normalization_approach)
        if return_one_explanation == False:
            self.explain_method.forward_func.zero_grad()
            claim_explanation = self.explain_method.attribute(**claim_attribute_kwargs,**self.parameters).detach()
            claim_explanation = postprocess_explanations(claim_explanation,self.apply_normalization,self.normalization_approach)

            return predictions, (post_explanation, claim_explanation), (enc_posts, enc_claims)
        
        return predictions, (post_explanation), (enc_posts, enc_claims)

    def _preprocess_batch_inputs(self, posts: list[str], claims: list[str]) -> dict:
        dev = utils.get_device()
        enc_posts = self.model_wrapper.preprocess_input(posts) #magic behind cutting tokens into smaller pieces
        enc_claims = self.model_wrapper.preprocess_input(claims)
        baseline_posts_ref, baseline_claims_ref = self._build_baselines(enc_posts, enc_claims)
        post_feature_mask, claim_feature_mask = None, None

        with torch.no_grad():
            if self.interpretable_emb_layer is not None:
                post_input_embeds = self.interpretable_emb_layer.indices_to_embeddings(enc_posts["input_encodings"]["input_ids"])
                claim_input_embeds = self.interpretable_emb_layer.indices_to_embeddings(enc_claims["input_encodings"]["input_ids"])

                # convert encodings to embeddings
                enc_posts["input_encodings"]["inputs_embeds"] = post_input_embeds
                enc_claims["input_encodings"]["inputs_embeds"] = claim_input_embeds
                
                baseline_posts_ref = self.interpretable_emb_layer.indices_to_embeddings(baseline_posts_ref)
                baseline_claims_ref = self.interpretable_emb_layer.indices_to_embeddings(baseline_claims_ref)

                if self.token_groups_for_feature_mask:
                    # masks for grouping features of the same token together
                    # used for attribution methods like feature permutation or ablation
                    # we want to permutate/ablate the entire tokens, not specific float values of their embeddings
                    pi, pj, pk = post_input_embeds.shape
                    post_feature_mask = torch.arange(0, pi*pj, dtype=torch.int, device=dev)
                    post_feature_mask = post_feature_mask.reshape(pi, pj, 1)
                    post_feature_mask = post_feature_mask.expand(-1, -1, pk)

                    ci, cj, ck = claim_input_embeds.shape
                    claim_feature_mask = torch.arange(0, ci*cj, dtype=torch.int, device=dev)
                    claim_feature_mask = claim_feature_mask.reshape(ci, cj, 1)
                    claim_feature_mask = claim_feature_mask.expand(-1, -1, ck)

        return {
            "enc_posts": enc_posts,
            "enc_claims": enc_claims,
            "baseline_posts_ref": baseline_posts_ref,
            "baseline_claims_ref": baseline_claims_ref,
            "post_feature_mask": post_feature_mask,
            "claim_feature_mask": claim_feature_mask,
        }

    def _precompute_embeddings(
        self, enc_posts: dict[str, torch.Tensor], enc_claims: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor]: 
        with torch.no_grad():
            emb_posts = self.model_wrapper._forward(enc_posts)
            emb_claims = self.model_wrapper._forward(enc_claims)
            predictions = torch.nn.functional.cosine_similarity(emb_posts, emb_claims)
    
        return emb_posts, emb_claims, predictions
    
    def _aggregate_attribute_kwargs(
            self, inputs: dict, emb_posts: torch.Tensor, 
            emb_claims: torch.Tensor, 
            additional_attribution_kwargs: dict
    ) -> tuple[dict, dict]:
        input_key = "input_ids" if self.interpretable_emb_layer is None else "inputs_embeds"
        post_inputs = inputs["enc_posts"]["input_encodings"][input_key]
        claim_inputs = inputs["enc_claims"]["input_encodings"][input_key]

        post_attribute_kwargs = {
            "inputs": post_inputs,
            "target": None,
            "additional_forward_args": (
                input_key,
                emb_claims,
                inputs["enc_posts"]
                # emb_claims,
                # inputs["enc_posts"]
            )
        }
        claim_attribute_kwargs = {
            "inputs": claim_inputs,
            "target": None,
            "additional_forward_args": (
                input_key,
                emb_posts,
                inputs["enc_claims"]
                # emb_posts,
                # inputs["enc_claims"]
                # inputs["enc_claims"]['input_encodings']['attention_mask']

            )
        }

        if self.compute_baseline is True:
            post_attribute_kwargs["baselines"] = inputs["baseline_posts_ref"]
            claim_attribute_kwargs["baselines"] = inputs["baseline_claims_ref"]
        if self.token_groups_for_feature_mask is True:
            post_attribute_kwargs["feature_mask"] = inputs["post_feature_mask"]
            claim_attribute_kwargs["feature_mask"] = inputs["claim_feature_mask"]
        if additional_attribution_kwargs is not None and type(additional_attribution_kwargs) == dict:
            post_attribute_kwargs.update(additional_attribution_kwargs)
            claim_attribute_kwargs.update(additional_attribution_kwargs)

        return post_attribute_kwargs, claim_attribute_kwargs
        
    def _build_baselines(self, enc_posts: dict[str, torch.Tensor], enc_claims: dict[str, torch.Tensor]):
        post_input_ids = enc_posts["input_encodings"]["input_ids"]
        claim_input_ids = enc_claims["input_encodings"]["input_ids"]
        
        baseline_posts_ref = torch.full_like(post_input_ids, self.tokenizer.pad_token_id)
        baseline_claims_ref = torch.full_like(claim_input_ids, self.tokenizer.pad_token_id)

        if self.tokenizer.bos_token_id is not None:
            baseline_posts_ref[post_input_ids == self.tokenizer.bos_token_id] = self.tokenizer.bos_token_id
            baseline_claims_ref[claim_input_ids == self.tokenizer.bos_token_id] = self.tokenizer.bos_token_id

        if self.tokenizer.eos_token_id is not None:
            baseline_posts_ref[post_input_ids == self.tokenizer.eos_token_id] = self.tokenizer.eos_token_id
            baseline_claims_ref[claim_input_ids == self.tokenizer.eos_token_id] = self.tokenizer.eos_token_id

        return baseline_posts_ref, baseline_claims_ref

    def _vizualize_pair(
            self, post_explanation: torch.Tensor, claim_explanation: torch.Tensor, 
            post_tokens: list[list[str]], claim_tokens: list[list[str]], 
            posts: tuple[str], claims: tuple[str], predictions: torch.Tensor
    ) -> None:
        """Function that visualizes token relevance for a specific prediction"""
        # TODO extract common code from this function and 
        # compare_multiple_explanation_methods function

        for i in range(len(post_explanation)):
            records = []
            vis_record_post = viz.VisualizationDataRecord(
                word_attributions=post_explanation[i],
                pred_prob=predictions[i],
                pred_class=1 if predictions[i] > 0 else -1,
                true_class=1,
                attr_class=1,
                attr_score=post_explanation[i].sum(),
                raw_input_ids=post_tokens[i],
                convergence_score=1
            )
            records.append(vis_record_post)

            if claim_explanation is not None:
                vis_record_claim = viz.VisualizationDataRecord(
                    word_attributions=claim_explanation[i],
                    pred_prob=predictions[i],
                    pred_class=1 if predictions[i] > 0 else -1,
                    true_class=1,
                    attr_class=1,
                    attr_score=claim_explanation[i].sum(),
                    raw_input_ids=claim_tokens[i],
                    convergence_score=1
                )
                records.append(vis_record_claim)
            
            print("--- Explaining mutual similarity between a [POST] and a [CLAIM] ---")
            print(f"\t[POST]: {posts[i]}")
            print(f"\t[CLAIM]: {claims[i]}")    
        
            viz.visualize_text(records)

    
class ExplanationExecuter_CT:
    """Class that takes a model (Classificationwrapper) and XAI attribution method from captum framework 
    and computes the explanation
    Work with classification task 
    """
    def __init__(
        self, method: Attribution, compute_baseline: bool = False, 
        apply_normalization: bool = True, 
        normalization_approach: Literal["min-max",'log-min-max', "l2", "max-abs"] = "log-min-max",
        token_groups_for_feature_mask: bool = False,
        visualize_explanation: bool = False, verbose: bool = True, parameters: dict = {}

    ) -> None:
        self.explain_method = method
        self.compute_baseline = compute_baseline
        self.apply_normalization = apply_normalization
        self.normalization_approach = normalization_approach
        self.visualize_explanation = visualize_explanation
        self.verbose = verbose
        self.explain_wrapper: ClassificationWrapper = method.forward_func
        self.model_wrapper: RepresentationModel = self.explain_wrapper.model
        self.encoder: SentenceTransformerToHF = self.model_wrapper.transformer

        self.tokenizer = self.encoder.tokenizer
        self.interpretable_emb_layer = self.encoder.interpretable_emb_layer
        self.token_groups_for_feature_mask = token_groups_for_feature_mask
        self.parameters=parameters

    def _build_baselines(self, enc_posts: dict[str, torch.Tensor]):
        post_input_ids = enc_posts["input_encodings"]["input_ids"]
        baseline_texts_ref = torch.full_like(post_input_ids, self.tokenizer.pad_token_id)
        if self.tokenizer.bos_token_id is not None:
            baseline_texts_ref[post_input_ids == self.tokenizer.bos_token_id] = self.tokenizer.bos_token_id
        if self.tokenizer.eos_token_id is not None:
            baseline_texts_ref[post_input_ids == self.tokenizer.eos_token_id] = self.tokenizer.eos_token_id
        return baseline_texts_ref

    def _preprocess_batch_inputs(self, posts: list[str]) -> dict:
        dev = utils.get_device()
        enc_posts = self.model_wrapper.preprocess_input(posts) #magic behind cutting tokens into smaller pieces
        baseline_posts_ref= self._build_baselines(enc_posts)
        post_feature_mask  = None

        with torch.no_grad():
            if self.interpretable_emb_layer is not None:
                post_input_embeds = self.interpretable_emb_layer.indices_to_embeddings(enc_posts["input_encodings"]["input_ids"])
                # convert encodings to embeddings
                enc_posts["input_encodings"]["inputs_embeds"] = post_input_embeds                
                baseline_posts_ref = self.interpretable_emb_layer.indices_to_embeddings(baseline_posts_ref)

                if self.token_groups_for_feature_mask:
                    # masks for grouping features of the same token together
                    # used for attribution methods like feature permutation or ablation
                    # we want to permutate/ablate the entire tokens, not specific float values of their embeddings
                    pi, pj, pk = post_input_embeds.shape
                    post_feature_mask = torch.arange(0, pi*pj, dtype=torch.int, device=dev)
                    post_feature_mask = post_feature_mask.reshape(pi, pj, 1)
                    post_feature_mask = post_feature_mask.expand(-1, -1, pk)

        return {
            "enc_posts": enc_posts,
            "baseline_posts_ref": baseline_posts_ref,
            "post_feature_mask": post_feature_mask,
        }

    def _aggregate_attribute_kwargs(
            self, inputs: dict, claim: int, 
            additional_attribution_kwargs: dict
    ) -> tuple[dict, dict]:
        input_key = "input_ids" if self.interpretable_emb_layer is None else "inputs_embeds"
        post_inputs = inputs["enc_posts"]["input_encodings"][input_key]
        post_attribute_kwargs = {
            "inputs": post_inputs,
            "target": claim,
            "additional_forward_args": (
            input_key,
            inputs["enc_posts"]
            )
        }
        if self.compute_baseline is True:
            post_attribute_kwargs["baselines"] = inputs["baseline_posts_ref"]
        if self.token_groups_for_feature_mask is True:
            post_attribute_kwargs["feature_mask"] = inputs["post_feature_mask"]
        if additional_attribution_kwargs is not None and type(additional_attribution_kwargs) == dict:
            post_attribute_kwargs.update(additional_attribution_kwargs)
        return post_attribute_kwargs
    def _precompute_prediction(
        self, 
        post_attribute_kwargs: dict[torch.Tensor, tuple[str,torch.Tensor]],
        claim: list
    ) -> tuple[torch.Tensor]: 
        with torch.no_grad():
            predictions=self.explain_wrapper.forward(post_attribute_kwargs['inputs'],
                                        post_attribute_kwargs['additional_forward_args'][0],
                                        post_attribute_kwargs['additional_forward_args'][1])
        predictions=predictions[:,claim]
        return predictions

    def explain_batch(
        self, posts: tuple[str], claims: tuple[str], 
        additional_attribution_kwargs: dict = None, 
        return_one_explanation: bool = False
    ) -> tuple[
        torch.Tensor, 
        Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], 
        tuple[dict, dict]
    ]:
        inputs = self._preprocess_batch_inputs(posts)
        enc_posts = inputs["enc_posts"]

        
        post_attribute_kwargs = self._aggregate_attribute_kwargs(
            inputs,claims, additional_attribution_kwargs
        )
        predictions=self._precompute_prediction(post_attribute_kwargs,claims)
        #calculating relevance and further postprocessing

        # self.explain_method.forward_func.zero_grad()
        post_explanation = self.explain_method.attribute(**post_attribute_kwargs, **self.parameters).detach()
        post_explanation = postprocess_explanations(post_explanation,self.apply_normalization,self.normalization_approach)
        
        return predictions,post_explanation, (enc_posts)




def concise_tensor(tokenizer,tensor,post):
    if isinstance(post,str):
        post=post.split(' ')
    tensor=tensor.squeeze()
    adj_tensor=tensor.clone()
    adjusted_tensor=[]
    for word in zip(post):
        tokenized_word=tokenizer(word)
        list_tok=tokenizer.convert_ids_to_tokens(tokenized_word['input_ids'][0])
        list_tok = [tok for tok in list_tok if tok not in tokenizer.all_special_tokens]
        try: 
        #tensor from rationale to normal
            word_tensor=adj_tensor[0:len(list_tok)]
            word_max_tensor = torch.max(word_tensor)
        # word_mean_tensor = torch.mean(word_tensor)
        # perc_dist=[]
        # for token in list_tok:
        #     # len(''.join(list_tok))/len(token)
        #     perc=(100/len(''.join(list_tok)))* len(token)
        #     perc_dist.append(perc/100)
        # word_mean_tensor = sum(value * weight for value, weight in zip(word_tensor, perc_dist))
        except:
            print('')
        try: 
            max_tensor = word_max_tensor.item()
        except:
            print('function: concise_tensor: mean tensor not properly set')
            max_tensor=0
        adjusted_tensor.append(max_tensor)
        adj_tensor=adj_tensor[len(list_tok):]
    adjusted_tensor=torch.Tensor(adjusted_tensor) 
    len_tens=list(adjusted_tensor.size())
    if len_tens[0] > 512:
       adjusted_tensor=adjusted_tensor[:512]
    return adjusted_tensor
    


# TODO Postprocessing of hierarchical model explanations
    # Multiple chunks belong to one document
    # There may be an overlap between chunks that we need to eliminate
    # Average relevances of tokens that are found in multiple chunks
    # ...


def compare_multiple_explanation_methods(
    explain_executors: list[ExplanationExecuter_STS], post: str, 
    claim: str, additional_attribution_kwargs: list[dict] = None, 
    method_names: list[str] = None, task: str=None ,visualize: bool = True

) -> None:
    all_explanations = {}
    if method_names is None:
        method_names = [f"Method {i}" for i in range(len(explain_executors))]
    
    for expl_it, (explain_wrapper, method_name) in enumerate(zip(explain_executors, method_names)):
        attribute_kwargs = {}

        if additional_attribution_kwargs is not None:
            if isinstance(additional_attribution_kwargs, dict):
                attribute_kwargs = additional_attribution_kwargs
            elif isinstance(additional_attribution_kwargs, list) and len(additional_attribution_kwargs) == len(explain_executors):
                attribute_kwargs = additional_attribution_kwargs[expl_it]

        predictions, explanations, encodings = explain_wrapper.explain_batch(
            [post], [claim], additional_attribution_kwargs=attribute_kwargs
        )
        if task == 'post_claim_matching':
            texts=(post,claim)
        else: 
            texts=(post)

        if  visualize == True:
            concise_post_expl=concise_tensor(explain_wrapper.tokenizer,explanations[0],post)
            concise_claim_expl=concise_tensor(explain_wrapper.tokenizer,explanations[1],claim)
            all_explanations[method_name] = (concise_post_expl,concise_claim_expl)
        else:
            if isinstance(explanations, torch.Tensor):
                explanations= [explanations]
            all_explanations[method_name] = tuple(explanation for explanation in explanations)
 
    if task == 'post_claim_matching':
        print("--- Explaining mutual similarity between a [POST] and a [CLAIM] ---")        
        print(f"\t[POST]: {post}")
        print(f"\t[CLAIM]: {claim}")
        visualize_list=["[POST]", "[CLAIM]"]
    else:
        print("--- Explaining classification between a [Text] and [Label] ---")        
        print(f"\t[TEXT]: {post}")
        print(f"\t[LABEL]: {claim}")      
        visualize_list=["[TEXT]"]  
    methods_values=[]
    for expl_it, type_of_string in enumerate(visualize_list):
        print(f"------- Explaining relevance of {type_of_string} -------")
        if visualize == True:

            for method_name in method_names:
                print(f"------------- Using method {method_name} -------------")

                relevance = all_explanations[method_name][expl_it]
                min_value =float(min(relevance[~torch.isnan(relevance)]))
                relevance = torch.nan_to_num(relevance,nan=min_value-1e-6)

                vis_record = viz.VisualizationDataRecord(
                    word_attributions=relevance,
                    pred_prob=predictions[0],
                    pred_class=1 if predictions[0] > 0 else -1,
                    true_class=1,
                    attr_class=1,
                    attr_score=relevance.sum(),
                    raw_input_ids=texts[expl_it].split(' '),
                    convergence_score=1
                )
                res=viz.visualize_text([vis_record])
                # with open(f"C:/Users/Dell/Desktop/Kinit/result_xai/methods visualization/qualitative/{method_name}-{type_of_string}-{post[0:10]}.html", "w") as file:
                #     file.write(res.data)
 
    if visualize == False:
            methods_values.append({'explanation':all_explanations,'post':post,'claim':claim})
            return methods_values





# if __name__ == "__main__":
    # import captum.attr as a
    # from torch.utils.data import DataLoader

    # from dataset import OurDataset
    # from explain import STS_ExplainWrapper, ExplanationExecuter

    # #smaller ds for testing purposes
    # # dataset = OurDataset(csv_dirpath="./data", split="test")
    # # dataset.fact_check_post_mapping = dataset.fact_check_post_mapping[:20] 
    # # loader = DataLoader(dataset, batch_size=4)

    # claims = [
    #     "sadads ad ads asd asd",
    #     "df hdfh dffh dfh " 
    # ] 
    # posts = [
    #     "sad adasfasf",
    #     "ioi kiu uik usd"
    # ]

    # model = STS_ExplainWrapper.setup_hierarchical_e5_transformer()
    # # model = STS_ExplainWrapper.setup_t5_transformer("./models/GTR-T5-FT")

    # method = a.InputXGradient(model)
    # method = a.LayerGradientXActivation(model, model.get_embedding_layer())
    # explain = ExplanationExecuter(method, compute_baseline=False, visualize_explanation=True)
    
    # # explanation = explain.explain(loader, return_post_explanation_only=True)
    # # print(explanation)

    # predictions, explanations, encodings = explain.explain_batch(
    #         posts, claims, return_one_explanation=True
    #     )
    # pass

