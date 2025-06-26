import os
import re
from tqdm import tqdm
import json
import numpy as np
from transformers import PreTrainedTokenizer
from typing import Union
from datetime import datetime

from dataset import OurDataset


class TokenConversion:
    """Class capable of converting tokens from transformers tokenizers
    into interpretable tokens, or rather words that we can easily understand

    So called interpretable tokens, or words, if you will, are a small units of texts
    that have its own semantic meaning
    """
    def __init__(self, tokenizer: PreTrainedTokenizer) -> None:
        self.tokenizer = tokenizer

    # TODO this function is tailored specifically to GTR T5 tokenizer and its not 
    # guaranteed it works with other tokenizers as well
    # TODO hence we need to create a seperate function for each tokenizer type...
    # TODO This logic may need to be rebuild
    def create_interpretable_tokens(
        self, text: str, real_tokens: list[str] = None,
        return_string: bool = False, return_token_mapping: bool = False
    ) -> Union[str, list[dict], list[str]]:
        """Create words / interpretable tokens that are used for human-friendly processing of real tokens
        Tokens created by Tokenizers may not have semantic meaning especially when we talk about some
        wordpieces... We'd rather want to combine pieces of various tokens to a semantically meaningful units of texts
        """
        if real_tokens is None:
            input_ids = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
            real_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # list of UNK token positions
        unk_tokens_indices = []
        for i in range(len(text)):
            char = text[i]
            char_input_ids = self.tokenizer(char)["input_ids"]
            if len(char_input_ids) > 1 and char_input_ids[1] == self.tokenizer.unk_token_id:
                unk_tokens_indices.append(i)

        # creating coherent word tokens from original tokens
        interpretable_tokens = []
        unk_encountered_iter = 0

        real_tokens_divided = []
        rt_offsets = []
    
        for it, real in enumerate(real_tokens):
            if ord(real[0]) == 9601:
                interpretable_tokens += [real[1:]]
                real_tokens_divided.append([real[1:]])
                rt_offsets.append(it)
            elif real == self.tokenizer.unk_token:                
                unknown_char = text[unk_tokens_indices[unk_encountered_iter]]
                interpretable_tokens[-1] += unknown_char
                real_tokens_divided[-1].append(real)
                unk_encountered_iter += 1
            elif real == self.tokenizer.eos_token:
                break
            else:
                interpretable_tokens[-1] += real
                real_tokens_divided[-1].append(real)
        
        if return_string:
            return " ".join(interpretable_tokens)
        if return_token_mapping:
            return [
                {
                    "interpretable_token": it,
                    "modified_real_tokens": rt,
                    "real_token_offset": rt_off
                } 
                for it, rt, rt_off in 
                zip(
                    interpretable_tokens, 
                    real_tokens_divided, 
                    rt_offsets
                )
            ]
        return interpretable_tokens
    
    def assign_rationale(
        self, annotated_string: str, orig_string: str, 
        annotations: list[dict], rationale_for_real_tokens: bool = False
    ) -> dict:  
        """Creates a binary /rationale/ mask representing which real tokens given to the transformer should in fact
        be responsible for the prediction from the human perspective

        The real token mask is created from the decomposition of the interpretable tokens, 
        hence we can also create masks that reflect the importance of intepretable tokens insteaad
        """
        input_ids = self.tokenizer(
            orig_string, truncation=False, add_special_tokens=False
        )["input_ids"]
        real_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        token_mapping = self.create_interpretable_tokens(
            orig_string,
            real_tokens=real_tokens,
            return_string=False,
            return_token_mapping=True
        )

        rationale_size = len(real_tokens) if rationale_for_real_tokens else len(token_mapping)
        rationale_mask = np.zeros(rationale_size, dtype=np.uint8)
        
        word_ranges = []
        word_offset = 0
        for idx, c in enumerate(annotated_string):
            if c == " ":
                word_ranges.append([word_offset, idx])
                word_offset = word_ranges[-1][1] + 1
        word_ranges.append([word_offset, len(annotated_string)])
        word_ranges = np.array(word_ranges)

        for annot in annotations:
            interp_word_indices = np.where(
                (word_ranges[:, 1] > annot["range"][0]) & 
                (word_ranges[:, 0] < annot["range"][1])
            )[0]

            for idx in interp_word_indices:
                token_map = token_mapping[idx]
                curr_it_range = word_ranges[idx]

                intersect = Annotations._calculate_range_intersection(curr_it_range, annot["range"])
                overlap = intersect[1] - intersect[0]

                # check if the annotation exceed the half of the token 
                # if it doesnt, we dont consider that token to be annotated
                if overlap >= len(token_map["interpretable_token"]) / 2:
                    if rationale_for_real_tokens:
                        mask_indices = (
                            np.arange(len(token_map["modified_real_tokens"])) + 
                            token_map["real_token_offset"]
                        )
                        rationale_mask[mask_indices] = 1
                    else:
                        rationale_mask[idx] = 1

        return {
            "interpretable_tokens": rationale_for_real_tokens == False,
            "tokens": (
                real_tokens 
                if rationale_for_real_tokens 
                else [t["interpretable_token"] for t in token_mapping]
            ),
            "mask": rationale_mask.tolist()
        }


class Annotations:
    """Class for creating data to annotate, as well as for interpreting the exported 
    JSONL file created by Doccano. This class is also responsible for creating rationale masks
    """
    def __init__(self, annotations_path: str, original_json_path: str = None) -> None:
        """
        `original_json_path` parameter is necessary for assigning annotations to a specific
        claim-post pair. If we define `original_json_path` argument, it means that both 
        the annotations and the original JSON data are in the same order (Nth annotation 
        corresponds to Nth original JSON object)

        If we dont define the `original_json_path`, then it means that in the annotations
        themselves we carry the information of which claim-post pair that particular annotation
        belongs to (we have claim_id + post_id at the end of the text to annotate). 
        In such a case we dont need the `original_json_path` argument that is only necessary
        for retreving IDs of objects of annotation pairs...
        """

        self.original_json = None
        if original_json_path is not None:
            with open(original_json_path, "r", encoding="utf-8") as f:
                self.original_json = json.load(f)
        with open(annotations_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.annotations = []
        for line in lines:
            self.annotations.append(json.loads(line))

    def preprocess_annot(self) -> list[dict]:
        """Function used to process JSONL Doccano annotation file and retrieve the explanation ranges / 
        annotations created by the users for specific post-claim pairs. This function assigns 
        explanation ranges to proper fields, while also documenting the corresponding post and claim IDs

        This function doesnt convert the explanation ranges to token/word masks /rationale/
        """
        processed_annotations = []
        for annot_it, annot in enumerate(self.annotations):
            text = annot["text"]
            try: 
                labels = annot["label"]
            except:
                labels=annot['annotations']
            
            post_key = "POST:"
            ocr_key = "\n\nOCR:"
            claim_key = "\n\nCLAIM:"
            post_id_key = "\n\nPOST_ID:"
            claim_id_key = "\n\nCLAIM_ID:"

            post_idx = 0
            ocr_idx = text.find(ocr_key)
            claim_idx = text.find(claim_key)
            post_id_idx = text.find(post_id_key)
            claim_id_idx = text.find(claim_id_key)

            post = text[post_idx + len(post_key) + 1: ocr_idx]
            ocrs = text[ocr_idx + len(ocr_key) + 1: claim_idx].split("\n")
            if ocrs[0] == "":
                ocrs = []
            
            if self.original_json is None:
                post_id = int(text[post_id_idx + len(post_id_key) + 1: claim_id_idx])
                claim_id = int(text[claim_id_idx + len(claim_id_key) + 1:])
            else:
                post_id = self.original_json[annot_it]["post_id"]
                claim_id = self.original_json[annot_it]["claim_id"]
            
            post_annotations = []
            ocr_annotations = [[] for _ in ocrs]
            for label in labels:
                if isinstance(label,dict):
                    start_index = text.find(label['text'])          
                    end_index = start_index + len(label['text']) - 1
                    label_range=[start_index, end_index]
                    label_type=label['label']
                else:
                    label_range = label[:2]
                    label_type = label[2]

                if label_type != "Explanation":
                    continue
            
                # check which field this particular explanation range belongs to
                # both could be highlighted simulatiously using one explanation                
                post_intersect = self._calculate_range_intersection(
                    label_range, (post_idx + len(post_key) + 1, ocr_idx)
                )
                self._add_annotation(
                    text,
                    post_annotations, 
                    post_idx + len(post_key) + 1, 
                    post_intersect
                )

                ocr_offset = ocr_idx + len(ocr_key) + 1
                for it, ocr in enumerate(ocrs):
                    ocr_interesect = self._calculate_range_intersection(
                        label_range, (ocr_offset, ocr_offset + len(ocr))
                    )
                    self._add_annotation(
                        text,
                        ocr_annotations[it],
                        ocr_offset, 
                        ocr_interesect
                        )
                    ocr_offset += len(ocr) + 1

            if len(post_annotations) + sum([len(ocr_annot) for ocr_annot in ocr_annotations]) > 0:
                # create a new annotation record containing all explanations of
                # text field and OCR field
                processed_annotations.append({
                    "post_id": post_id,
                    "claim_id": claim_id,
                    "post": post,
                    "ocrs": ocrs,
                    "post_annotations": post_annotations,
                    "ocr_annotations": ocr_annotations
                })

        return processed_annotations
    
    def create_rationale_masks(
        self, dataset: OurDataset, 
        token_converter: TokenConversion, 
        savepath: str = None,
        rationale_for_real_tokens: bool = True
    ) -> None:
        """Wrapper function that encapsulates the whole logic of converting the JSONL file
        from Doccano to rationale masks that will be used for computing understandability
        metrics
        """
        annotations = self.preprocess_annot()
        rationale_json_data = []

        for annot in tqdm(annotations):            
            post_rationale = {}
            ocr_rationale = []
            post = dataset.all_df_posts.loc[annot["post_id"]]

            if len(annot["post"]) > 0:
                orig_post_string = dataset.clean_text(post["text"][dataset.text_version == "english"])
                post_rationale = token_converter.assign_rationale(
                    annot["post"], orig_post_string, annot["post_annotations"], 
                    rationale_for_real_tokens=rationale_for_real_tokens
                )

            for idx in range(len(annot["ocrs"])):
                orig_ocr_string = dataset.clean_text(post["ocr"][idx][dataset.ocr_version == "english"])
                ocr_rationale.append(
                    token_converter.assign_rationale(
                        annot["ocrs"][idx], orig_ocr_string, annot["ocr_annotations"][idx], 
                        rationale_for_real_tokens=rationale_for_real_tokens
                    )
                )

            if len(post_rationale) + len(ocr_rationale) > 0:
                rationale_json_data.append({
                    "post_id": annot["post_id"],
                    "fact_check_id": annot["claim_id"],
                    "post_rationale": post_rationale,
                    "ocr_rationale": ocr_rationale
                })
                
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, "w", encoding="utf-8") as f:
            json.dump(rationale_json_data, f, ensure_ascii=False)
    
    @staticmethod
    def _add_annotation(text,annotation_list: list[dict], offset: int, intersect: tuple[int, int]) -> None:
        if intersect is not None:
            annotation_list.append({
                "text":text[intersect[0]:intersect[1]],
                "range": [intersect[0] - offset, intersect[1] - offset],
                "type": "explanation"
            })

    @staticmethod
    def _calculate_range_intersection(range1: list[int], range2: list[int]) -> np.ndarray:
        intersection_start = max(range1[0], range2[0])
        intersection_end = min(range1[1], range2[1])

        if intersection_start < intersection_end:
            return np.array([intersection_start, intersection_end])
        else:
            return None
    
    @staticmethod
    def create_data_to_annotate(
        convert: TokenConversion, dataset: OurDataset, 
        json_savepath: str = "./data-to-annotate.json", 
        jsonl_savepath: str = "./data-to-annotate.jsonl",
        verbose: bool = True
    ) -> None:
        """Function to create annotations for a specific claim dataset and its matching pairs
        with its corresponding posts

        This function creates two files, a JSON file that contains metadata about 
        the data being annotated -> this file is also required for later reconstruction 
        and intepretation of annotations exported from Doccano

        And a JSONL file that contains only text and is imported to Doccano. This file
        is rather poor information wise, hence the need for the intermediate JSON file
        """
        data = []
        for (fc_id, post_id) in tqdm(dataset.fact_check_post_mapping, disable=verbose==False):
            post = dataset.all_df_posts.loc[post_id]
            
            post_text = post["text"]
            if len(post_text) > 0:
                post_text = dataset.clean_text(post["text"][dataset.text_version == "english"])
                post_text = " ".join(convert.create_interpretable_tokens(post_text))

            ocrs_text = [dataset.clean_text(ocr[dataset.ocr_version == "english"]) for ocr in post["ocr"]]
            ocrs_text = [" ".join(convert.create_interpretable_tokens(ocr)) for ocr in ocrs_text]

            claim_text = dataset.id_to_fact_check_vocab[fc_id]
            claim_text = " ".join(convert.create_interpretable_tokens(claim_text))

            data.append({
                "post_id": int(post_id),
                "claim_id": int(fc_id),
                "post": post_text,
                "claim": claim_text,
                "ocr": ocrs_text
            })

        if json_savepath is not None:
            os.makedirs(os.path.dirname(json_savepath), exist_ok=True)
            with open(json_savepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        Annotations._final_postprocessing_steps_for_doccano(data, jsonl_savepath)

    @staticmethod
    def _final_postprocessing_steps_for_doccano(
        json_data: dict, savepath: str = "./data-to-annotate.jsonl"
    ) -> None:
        """Converting JSON metadata to a more compact format that is further being used
        by Doccano (JSONL file)
        """
        line_data = []
        for data in json_data:
            ocr_text = "\n".join(data["ocr"])
            text = (
                f"POST: {data['post']}\n\nOCR: {ocr_text}\n\nCLAIM: {data['claim']}" +
                f"\n\nPOST_ID: {data['post_id']}\n\nCLAIM_ID: {data['claim_id']}"
            )
            line_data.append(json.dumps({ "text": text }, ensure_ascii=False) + "\n")

        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, "w") as f:
            f.writelines(line_data)    

    @staticmethod 
    def aggregate_rationales(dirpath: str, save_filename: str = "rationale.json") -> None:
        """
        Combine the rationales found in multiple JSON files
        The rationale JSON files are expected to end with YYYY-MM-DD.json substring
        that we use to sort them based on their age as a newer rationale can override
        an older one
        """
        file_names = sorted(os.listdir(dirpath))
        pattern = r".*[0-9]{4}-[0-9]{2}-[0-9]{2}.(json|JSON)$"
        assert sum([re.match(pattern, f) is not None for f in file_names]) == len(file_names), \
        "Not all files end with YYYY-MM-DD.json date substring"
    
        dates = [datetime.strptime(f[-15:-5], "%Y-%m-%d").timestamp() for f in file_names]
        indices = np.array(dates).argsort()
        file_names = np.array(file_names)[indices]

        aggregated = np.array([])
        aggregated_map = np.array([])
        for file in file_names:
            with open(os.path.join(dirpath, file),encoding='utf-8') as fp:
                annot = np.array(json.load(fp))
            
            _map = np.array([f'{a["fact_check_id"]}-{a["post_id"]}' for a in annot])
            duplicates_mask = np.isin(_map, aggregated_map)

            # override old data
            if duplicates_mask.sum() > 0:
                agg_indices = np.array([aggregated_map.tolist().index(val) for val in _map[duplicates_mask]])
                aggregated[agg_indices] = annot[np.where(duplicates_mask)[0]]
            
            # new data
            aggregated = np.hstack([aggregated, annot[~duplicates_mask]])
            aggregated_map = np.hstack([aggregated_map, _map[np.where(duplicates_mask == False)[0]]])

        with open(os.path.join(dirpath, save_filename), "w",encoding='utf-8') as fp:
            json.dump(aggregated.tolist(), fp, ensure_ascii=False)


def create_annotations_for_doccano(
    dataset: OurDataset, tokenizer: PreTrainedTokenizer, 
    annot_offset: int = 0, annot_limit: int = 1000,
    user_names=["MV", "MT", "AR", "JK", "IB", "QP"],
    ignore_first_users: int = 6,
    savepath: str = "./data/for-doccano"
) -> None:
    convert = TokenConversion(tokenizer)
    dataset.fact_check_post_mapping = np.random.permutation(
        dataset.fact_check_post_mapping
    )[annot_offset: annot_offset+annot_limit]

    os.makedirs(savepath, exist_ok=True)
    for it, user in enumerate(user_names):
        dataset.fact_check_post_mapping = np.random.permutation(
            dataset.fact_check_post_mapping
        )
        if it < ignore_first_users:
            continue
        Annotations.create_data_to_annotate(
            convert, dataset, 
            json_savepath=os.path.join(savepath, f"{user}.json"),
            jsonl_savepath=os.path.join(savepath, f"{user}.jsonl")
        )


if __name__ == "__main__":
    import utils
    from sentence_transformers import SentenceTransformer

    utils.no_randomness()
    dataset = OurDataset(csv_dirpath="./data", split=None)
    model = SentenceTransformer("./models/GTR-T5-FT")

    user_names = ["MV", "MT", "AR", "JK", "IB", "QP"]
    user_names += [f"user_{i}" for i in range(100)]

    create_annotations_for_doccano(
        dataset, model.tokenizer, 
        annot_offset=1000, annot_limit=1000,
        user_names=user_names, ignore_first_users=6, 
        savepath="./data/for-doccano"
    )