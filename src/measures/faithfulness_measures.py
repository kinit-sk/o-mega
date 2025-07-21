import copy
import numpy as np
from scipy.stats import kendalltau
import torch
from transformers import PreTrainedTokenizer
import torch
from explain import STS_ExplainWrapper
import re
from utils import get_device
from .utils import Explanation, BaseEvaluator, Evaluation,divide_text_to_sentences
from .utils import (
    _check_and_define_get_id_discrete_rationale_function,
    parse_evaluator_args
)
from architecture import get_tokenizer

import matplotlib.pyplot as plt
import numpy as np


class AOPC_Evaluation(BaseEvaluator):
    def __init__(self, model: STS_ExplainWrapper, tokenizer: PreTrainedTokenizer):
        super().__init__(model, tokenizer)
    @staticmethod
    def forward_PCM(helper,enc_text,emb_ctx):
        with torch.no_grad():
            predictions = helper.model(enc_text["input_encodings"]["input_ids"], "input_ids", emb_ctx, additional_forward_kwargs=enc_text)
            predictions = (predictions + 1) / 2
            return predictions 
    @staticmethod
    def forward_TC(helper,enc_text,ctx_text):
        with torch.no_grad():
            predictions = helper.model(enc_text["input_encodings"]["input_ids"], "input_ids", additional_forward_kwargs=enc_text)
            predictions=predictions[:,ctx_text.tolist()]  
            return predictions 

    def compute_evaluation(self, explanation: Explanation, aopc_func, word_evaluation,sentence_evaluation,task,**evaluation_args):
            _, only_pos, removal_args, _ = parse_evaluator_args(
                evaluation_args
            )
            input_text = explanation.input_text
            ctx_text = explanation.ctx_text
            score_explanation = explanation.scores
            model_wrapper = self.helper.model.model # Basic_RepresentationModel()
            enc_text = model_wrapper.preprocess_input(input_text) #'input_encodings':{input_ids,attention_mask}

            if task == 'post_claim_matching':
                enc_ctx = model_wrapper.preprocess_input(ctx_text)
                # with torch.no_grad():
                emb_ctx = model_wrapper._forward(enc_ctx)
                fowr_function=AOPC_Evaluation.forward_PCM
                target=emb_ctx 
            if task == 'text_classification':
                fowr_function=AOPC_Evaluation.forward_TC
                target= ctx_text

            if sentence_evaluation:
                sample=divide_text_to_sentences(input_text)
            # with torch.no_grad():
            baseline = fowr_function(self.helper,enc_text,target)
            input_len = enc_text["input_encodings"]["attention_mask"].sum().item()
            input_ids = enc_text["input_encodings"]["input_ids"][0][:input_len].tolist()
            tokenizer=get_tokenizer(model_wrapper)
            input_ids = [token_id for token_id in input_ids if token_id not in set(tokenizer.all_special_ids)]
            full_discrete_expl_ths = []
            # If tokens where not right excluded, (for security/sanity check)
            if word_evaluation== False and sentence_evaluation== False:
                if explanation.scores.size != len(input_ids):
                    a = enc_text["input_encodings"]["input_ids"][0][:input_len].tolist()
                    ids_special_tokens= [idx for idx, token_id in enumerate(a) if token_id in set(tokenizer.all_special_ids)]
                    for id in ids_special_tokens:
                        explanation.scores=np.delete(explanation.scores, id)
                        if explanation.scores.size == len(input_ids):
                            score_explanation = explanation.scores
                            break


            # print()
            id_tops = []
            get_discrete_rationale_function = (
                _check_and_define_get_id_discrete_rationale_function(
                    removal_args["based_on"]
                )
            )
            thresholds = removal_args["thresholds"]
            last_id_top = None
            full_discrete_expl_ths = [] # HERE: doplnit pre aopc
            if self.SHORT_NAME=='aopc_suff':
                combination_exp=[score_explanation,-score_explanation]
            else:
                combination_exp=[-score_explanation,score_explanation]
            for i,score_exp in enumerate(combination_exp):
                discrete_expl_ths = []
                for v in thresholds:
                    # Get rationale from score explanation
                    id_top = get_discrete_rationale_function(score_exp, v, only_pos) #numpy.ndarray nd1
                    # If the rationale is the same, we do not include it. In this way, we will not consider in the average the same omission.
                    if (
                        id_top is not None
                        and last_id_top is not None
                        and set(id_top) == last_id_top
                    ):
                        id_top = None
                    id_tops.append(id_top)
                    if id_top is None:
                        continue

                    
                    # combine positive and negative together in second part
                    if i ==1:
                        if last_id_top is None:
                            last_id_top=set()
                        last_id_top = last_id_top| set(id_top)
                        id_top=list(last_id_top)
                    else: 
                        last_id_top = set(id_top)
                    # Comprehensiveness
                    # The only difference between comprehesivenss and sufficiency is the computation of the removal.
                    # For the comprehensiveness: we remove the terms in the discrete rationale.
                    if word_evaluation:
                        sample= input_text.split()
                    try:
                        if word_evaluation or sentence_evaluation:
                            discrete_expl_th=aopc_func(sample,id_top, removal_args)
                            discrete_expl_th=' '.join(discrete_expl_th)
                            discrete_expl_ths.append(discrete_expl_th)
                        else:
                            sample = np.array(copy.copy(input_ids))
                            discrete_expl_th_token_ids = aopc_func(self.tokenizer, sample, id_top, removal_args)
                            discrete_expl_th = self.tokenizer.decode(discrete_expl_th_token_ids)
                            discrete_expl_ths.append(discrete_expl_th)
                    except Exception as e :
                        print(e)
                        print("Error")
                full_discrete_expl_ths.append(discrete_expl_ths)
            if (full_discrete_expl_ths[0]==[] and self.SHORT_NAME == 'aopc_suff') or (full_discrete_expl_ths[1]==[] and self.SHORT_NAME == 'aopc_compr'):
                # TODO it looks like this value is also obtained if we've only negative attributions as explanation 
                # I haven't really taken time to explore this in more detail yet
                sc = torch.zeros(1, device=get_device(),dtype=torch.float32)
                evaluation_output= Evaluation(self.SHORT_NAME, sc)
                probs_removing= []
                return evaluation_output,probs_removing
            probs_removing=[]
            # FORWARD FUNC ----
            for discrete_expl_ths in full_discrete_expl_ths:
                if discrete_expl_ths:
                    # with torch.no_grad():
                    enc_new_texts = model_wrapper.preprocess_input(discrete_expl_ths)
                    probs = fowr_function(self.helper,enc_new_texts,target)
                    probs_removing.append(probs)
                else:
                    probs_removing.append(torch.empty(0,device=get_device()))
            # -----

            # compute probability difference
            if self.SHORT_NAME == 'aopc_suff':
                removal_importance = baseline - probs_removing[0]
            else:
                removal_importance = baseline - probs_removing[1]
            #  compute AOPC 
            aopc_ev = removal_importance.mean()
            evaluation_output = Evaluation(self.SHORT_NAME, aopc_ev)
            # print(f'Full text:\n{input_text}\n{baseline}')
            # for i,p in zip(discrete_expl_ths,probs_removing):
            #     print(f'{i}:{p}')
            # with torch.no_grad():
            empty_string = model_wrapper.preprocess_input('')
            empty_baseline = fowr_function(self.helper,empty_string,target)
            probs_removing= torch.cat((empty_baseline,probs_removing[0],probs_removing[1],baseline))
            probs_removing=probs_removing.tolist()
            # print(baseline)
            # print(probs_removing)
            # discrete_expl_ths.insert(0,input_text)
            # thresholds=np.insert(thresholds,0,0)
            if self.SHORT_NAME=='aopc_suff':
                n_thresholds= np.arange(-0.1 * len(full_discrete_expl_ths[0]), 0, 0.1) # array([-1. , -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,-0.1]) for specific set 
            else: 
                n_thresholds= np.arange(-0.1, -0.1 * (len(full_discrete_expl_ths[0])+1), -0.1) # array([-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1. ]) for base algorithm
            thresholds = np.concatenate(([0], n_thresholds,thresholds, [100]))
            thresholds= np.round(thresholds, 1)
            results = [{'erassing_itterations': float(perc), 'prob': float(p)} for perc, p in zip(thresholds, probs_removing)]
            try:
                return evaluation_output,results #,discrete_expl_ths,thresholds
            except Exception as e:
                print(e)
            
class AOPC_Comprehensiveness_Evaluation(AOPC_Evaluation):
    NAME = "aopc_comprehensiveness"
    SHORT_NAME = "aopc_compr"
    HIGHER_IS_BETTER = True
    RANGE = (0, 1)
    TYPE_METRIC = "faithfulness"

    def __init__(self, model: STS_ExplainWrapper, tokenizer: PreTrainedTokenizer):
        super().__init__(model, tokenizer)

    @staticmethod
    def _aopc_func(tokenizer, sample, id_top, removal_args):
        if removal_args["remove_tokens"]:
            discrete_expl_th_token_ids = np.delete(sample, id_top)
        else:
            sample[id_top] = tokenizer.mask_token_id
            discrete_expl_th_token_ids = sample

        return discrete_expl_th_token_ids

    @staticmethod
    def _aopc_func_words(sample, id_top, removal_args):
        if removal_args["remove_tokens"]:
            discrete_expl_th_token_ids = [word for i,word in enumerate(sample) if i not in id_top ]
        else:
            discrete_expl_th_token_ids = [word if i not in id_top else '[MASK]' for i, word in enumerate(sample)]

        return discrete_expl_th_token_ids

    @staticmethod
    def _aopc_func_sentence(sample, id_top, removal_args):
        if removal_args["remove_tokens"]:
            discrete_expl_th_token_ids = [word for i,word in enumerate(sample) if i not in id_top ]
        else:
            discrete_expl_th_token_ids = [word if i not in id_top else '[MASK]' for i, word in enumerate(sample)]

        return discrete_expl_th_token_ids 



    def compute_evaluation(self, explanation: Explanation, word_evaluation: bool,sentence_evaluation:bool,task:str,normalize=False, **evaluation_args) -> Evaluation:
        # if (explanation.scores < 0).all():
        #     explanation.scores=explanation.scores + abs(min(explanation.scores))
        try:
            if word_evaluation:
                result,probs_removing= super().compute_evaluation(explanation, self._aopc_func_words,word_evaluation,sentence_evaluation,task,**evaluation_args)# ,probs_removing, discrete_expl_ths,thresholds
            elif sentence_evaluation:
                result,probs_removing= super().compute_evaluation(explanation, self._aopc_func_sentence,word_evaluation,sentence_evaluation,task, **evaluation_args)# ,probs_removing, discrete_expl_ths,thresholds
            else:
                result,probs_removing= super().compute_evaluation(explanation, self._aopc_func,word_evaluation,sentence_evaluation,task,**evaluation_args)# ,probs_removing, discrete_expl_ths,thresholds 
        except Exception as e :
            print('----')
            print(e)
            result=0

        # thresholds = thresholds[:len(probs_removing)]

        # list_3=thresholds
        # list_2=discrete_expl_ths
        # list_1=probs_removing
        
        # list_3 = list_3[:len(list_1)]
        # cmap = plt.cm.Blues
        # colors = cmap(np.linspace(0.3, 1, len(list_3)))  # Generate shades of blue

        # plt.plot(list_3, list_1, color='grey', linestyle='-', marker=None)

        # # Create the plot
        # for i in range(len(list_3)):
        #     plt.plot(list_3[i], list_1[i], marker='o', linestyle='-', color=colors[i]) #, label=list_2[i]

        # # Add labels and title
        # plt.xlabel('Deleting of best ')
        # plt.ylabel('Probability')
        # plt.title('AOPC_Comprehensiveness')

        # # Add a legend outside the plot
        # # plt.legend(title='Data Point Legends', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # # Show the plot
        # plt.grid(True)
        # for i in list_2:
        #     print(i)
        # plt.savefig(f'C:/Users/Dell/Desktop/Kinit/result_xai/metrics visualization/{discrete_expl_ths[0][0:5]}.png', format='png', dpi=300)
        # plt.show()
        # with open(f'C:/Users/Dell/Desktop/Kinit/result_xai/metrics visualization/{discrete_expl_ths[0][0:5]}.txt', 'w') as f:
        #     for line in list_2:
        #         f.write("%s\n" % line)
        if not hasattr(result,'score'):
            result=Evaluation(name='aopc_suff',score=torch.zeros(1, dtype=torch.float32))
        if result.score.numel() == 0:
            result.score= torch.zeros(1, dtype=torch.float32)
        if result.score == 0:
            result.score=torch.zeros(1, dtype=torch.float32)
            return result#, {'thresholds':thresholds,'discrete_expl_ths':discrete_expl_ths,'probs_removing':probs_removing}

        if normalize:
            sc = result.score
            result.score = (
                (sc - self.RANGE[0]) 
                / (self.RANGE[1] - self.RANGE[0])
            )
            if self.HIGHER_IS_BETTER == False:
                result.score = 1 - result.score
        for i in probs_removing: 
            if i['erassing_itterations'] == 0.0:
                result.score = torch.as_tensor(i['prob'])+result.score
        res=[result,probs_removing]
        return res#, {'thresholds':thresholds,'discrete_expl_ths':discrete_expl_ths,'probs_removing':probs_removing}


class AOPC_Sufficiency_Evaluation(AOPC_Evaluation):
    NAME = "aopc_sufficiency"
    SHORT_NAME = "aopc_suff"
    HIGHER_IS_BETTER = False
    RANGE = (0, 1)
    TYPE_METRIC = "faithfulness"
    
    def __init__(self, model: STS_ExplainWrapper, tokenizer: PreTrainedTokenizer):
        super().__init__(model, tokenizer)

    @staticmethod
    def _aopc_func(tokenizer, sample, id_top, removal_args):
        id_top = np.sort(id_top)

        if removal_args["remove_tokens"]:
            discrete_expl_th_token_ids = sample[id_top]
        else:
            sample[id_top] = tokenizer.mask_token_id
            discrete_expl_th_token_ids = sample

        return discrete_expl_th_token_ids
    
    @staticmethod
    def _aopc_func_words(sample, id_top, removal_args):
        id_top = np.sort(id_top)

        if removal_args["remove_tokens"]:
            discrete_expl_th_token_ids = [sample[i] for i in id_top]
        else:
            discrete_expl_th_token_ids = [word if i not in id_top else '[MASK]' for i, word in enumerate(sample)]

        return discrete_expl_th_token_ids

    @staticmethod
    def _aopc_func_sentence(sample, id_top, removal_args):
        if removal_args["remove_tokens"]:
            discrete_expl_th_token_ids = [word for i,word in enumerate(sample) if i not in id_top ]
        else:
            discrete_expl_th_token_ids = [word if i not in id_top else '[MASK]' for i, word in enumerate(sample)]

        return discrete_expl_th_token_ids 

    def compute_evaluation(self, explanation: Explanation, word_evaluation: bool,sentence_evaluation:bool,task,normalize=False, **evaluation_args) -> Evaluation:
        # if (explanation.scores < 0).all():
        #     explanation.scores=explanation.scores + abs(min(explanation.scores))
        try:
            if word_evaluation:
                result,probs_removing= super().compute_evaluation(explanation, self._aopc_func_words,word_evaluation,sentence_evaluation,task, **evaluation_args)# ,probs_removing, discrete_expl_ths,thresholds
            elif sentence_evaluation:
                result,probs_removing= super().compute_evaluation(explanation, self._aopc_func_sentence,word_evaluation,sentence_evaluation,task, **evaluation_args)# ,probs_removing, discrete_expl_ths,thresholds
            else:
                result,probs_removing= super().compute_evaluation(explanation, self._aopc_func,word_evaluation,sentence_evaluation,task, **evaluation_args)# ,probs_removing, discrete_expl_ths,thresholds 
        except Exception as e :
            print('----')
            print(e)
            result=0


        # thresholds = thresholds[:len(probs_removing)]

        # list_3=thresholds
        # list_2=discrete_expl_ths
        # list_1=probs_removing
        # list_3 = list_3[:len(list_1)]
        # cmap = plt.cm.Blues
        # colors = cmap(np.linspace(0.3, 1, len(list_3)))  # Generate shades of blue

        # plt.plot(list_3, list_1, color='grey', linestyle='-', marker=None)

        # # Create the plot
        # for i in range(len(list_3)):
        #     plt.plot(list_3[i], list_1[i], marker='o', linestyle='-', color=colors[i]) #, label=list_2[i]

        # # Add labels and title
        # plt.xlabel('Using of best ')
        # plt.ylabel('Probability')
        # plt.title(__class__.__name__)

        # Add a legend outside the plot
        # plt.legend(title='Data Point Legends', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Show the plot
        # plt.grid(True)
        # for i in list_2:
        #     print(i)
        # plt.savefig(f'C:/Users/Dell/Desktop/Kinit/result_xai/metrics visualization/AOPC_Sufficiency/{discrete_expl_ths[0][0:5]}.png', format='png', dpi=300)
        # plt.show()
        # with open(f'C:/Users/Dell/Desktop/Kinit/result_xai/metrics visualization/AOPC_Sufficiency/{discrete_expl_ths[0][0:5]}.txt', 'w') as f:
        #     for line in list_2:
        #         f.write("%s\n" % line)
        if not hasattr(result,'score'):
            result=Evaluation(name='aopc_suff',score=torch.zeros(1, dtype=torch.float32))
        if result.score.numel() == 0:
            result.score= torch.zeros(1, dtype=torch.float32)
        if result.score == 0:
            result.score=torch.zeros(1, dtype=torch.float32)
            return result#, {'thresholds':thresholds,'discrete_expl_ths':discrete_expl_ths,'probs_removing':probs_removing}

        if normalize:
            sc = result.score
            result.score = (
                (sc - self.RANGE[0]) 
                / (self.RANGE[1] - self.RANGE[0])
            )
            if self.HIGHER_IS_BETTER == False:
                result.score = 1 - result.score
        res= [result,probs_removing]
        return res#, {'thresholds':thresholds,'discrete_expl_ths':discrete_expl_ths,'probs_removing':probs_removing}
    

class TauLOO_Evaluation(BaseEvaluator):
    NAME = "tau_leave-one-out_correlation"
    SHORT_NAME = "taucorr_loo"
    TYPE_METRIC = "faithfulness"
    HIGHER_IS_BETTER = True
    RANGE = (-1, 1)

    def compute_leave_one_out_occlusion(self, input_text, ctx_text):
        model_wrapper = self.helper.model.model
        
        enc_text = model_wrapper.preprocess_input(input_text)
        enc_ctx = model_wrapper.preprocess_input(ctx_text)
    
        with torch.no_grad():
            emb_ctx = model_wrapper._forward(enc_ctx)
        
            baseline = self.helper.model(enc_text["input_encodings"]["input_ids"], "input_ids", emb_ctx, additional_forward_kwargs=enc_text)
            baseline = (baseline + 1) / 2

        input_len = enc_text["input_encodings"]["attention_mask"].sum().item()
        input_ids = enc_text["input_encodings"]["input_ids"][0][:input_len].tolist()

        input_ids = input_ids[:-1]
    
        samples = []
        for occ_idx in range(len(input_ids)):
            sample = copy.copy(input_ids)
            sample.pop(occ_idx)
            sample = self.tokenizer.decode(sample)
            samples.append(sample)

        enc_new_texts = model_wrapper.preprocess_input(samples)
        with torch.no_grad():
            leave_one_out_removal = self.helper.model(enc_new_texts["input_encodings"]["input_ids"], "input_ids", emb_ctx, attention_mask=enc_new_texts)
            leave_one_out_removal = (leave_one_out_removal + 1) / 2

        occlusion_importance = leave_one_out_removal - baseline
        return occlusion_importance.cpu().numpy()

    def compute_evaluation(self, explanation: Explanation, normalize=False, **evaluation_args) -> Evaluation:
        """Evaluate an explanation on the tau-LOO metric,
        i.e., the Kendall tau correlation between the explanation scores and leave one out (LOO) scores,
        computed by leaving one feature out and computing the change in the prediciton probability

        Args:
            explanation (Explanation): the explanation to evaluate
            target (int): class label for which the explanation is evaluated
            evaluation_args (dict):  additional evaluation args

        Returns:
            Evaluation : the tau-LOO score of the explanation
        """

        input_text = explanation.input_text
        ctx_text = explanation.ctx_text
        
        score_explanation = explanation.scores[:-1]

        loo_scores = self.compute_leave_one_out_occlusion(input_text, ctx_text) * -1
        kendalltau_score = kendalltau(loo_scores, score_explanation)[0]

        evaluation_output = Evaluation(self.SHORT_NAME, kendalltau_score)

        if normalize:
            sc = evaluation_output.score
            evaluation_output.score = (
                (sc - self.RANGE[0]) 
                / (self.RANGE[1] - self.RANGE[0])
            )
            if self.HIGHER_IS_BETTER == False:
                evaluation_output.score = 1 - evaluation_output.score

        evaluation_output.score = torch.tensor(evaluation_output.score)
        return evaluation_output