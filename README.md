# O-MEGA-pipeline
*This repository contains scripts created for automizing the process of searching an optimal XAI algorithm based on quantitative metrics*

In this repository we validate the idea of automating the XAI algorithm selection by executing the O-MEGA pipeline 
on a claim matching (PFCR) task. For this particular task we use our [multiclaim](https://arxiv.org/abs/2305.07991) dataset that contains a database of various social posts 
with their corresponding fact checks that either further acknowledge or disprove claims found in the posts. 
In this work, we solely work with English version of this dataset.

## Repo setup

In this section you can find the necessary steps to take for setting up the repository.

### Dependencies
1. Create a new conda virtual environment using Python 3.9
    
    ```
    conda create --name <ENV_NAME> python=3.9
    conda activate <ENV_NAME>
    ```
2. Install the necessary dependencies
    
    ```
    pip install -r requirements.txt
    ```

### Data preprocessing

1. Download the following CSV files containing the posts and claims of multiclaim dataset. [LINK](https://drive.google.com/file/d/1TGJFi0rkRTwhsPV52e0WvP5AxRVNyw-n/view)
1. Place the downloaded folder in the root directory of the project.
1. Preprocess the data by executing various filtering and cleaning steps using the following command: `python src/preprocess.py` (This may take up to a minute)

The previous script creates the following important artifacts necessary for further experimentations:
- `notebooks/data/posts.csv`, `notebooks/data/fact_checks.csv`, `notebooks/data/fact_check_post_mapping.csv` files representing clean multiclaim dataset
- `data/annotations/rationale.json` file representing all the user rationale gathered from all other rationale JSON files

After having preprocessed the data, you may call OurDataset constructor to instantiate training or test subset that is used for either 
training the O-MEGA pipeline or for further XAI algorithm evaluation. 

```
    from dataset import OurDataset
    from torch.utils.data import DataLoader

    # Training subset
    train_ds = OurDataset(csv_dirpath="./data", split="train")
    train_loader = DataLoader(train_ds)

    # Testing subset
    test_ds = OurDataset(csv_dirpath="./data", split="test")
    test_loader = DataLoader(test_ds)
```

| Subset    | Subset Size |
| --------- | ----------- |
| Train     | 5108        |
| Test      | 1278        |


### Model preprocessing

1. Download all the files (the entire directory) found on link corresponding to the fine-tuned model trained on multiclaim dataset. [LINK](https://drive.google.com/drive/folders/1PyCXoJBIi7zu26_HVSOORRK1pF6DNToT)
1. Move the directory into the following path: `models/GTR-T5-FT`
1. If you have correctly placed the said directory, invoking the SentenceTransformer constructor with the model path and path to embedding layer should correctly instatiate the model.

    ```
    # Check whether you've correctly placed the model data into the codespace
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("./models/GTR-T5-FT","encoder.embed_tokens")
    ```
#### Optional: 
It is a possibility to load models from Huggingface. List of models:
| Model name | Embedding layer |
|-----------------------------------------------|--------------------------|
| sentence-transformers/gtr-t5-large            | encoder.embed_tokens      |
| sentence-transformers/gtr-t5-xl               | encoder.embed_tokens      |
| sentence-transformers/sentence-t5-xl          | encoder.embed_tokens      |
| sentence-transformers/all-mpnet-base-v2       | embeddings.word_embeddings|
| sentence-transformers/multi-qa-mpnet-base-cos-v1 | embeddings.word_embeddings|

## O-MEGA pipeline - Optimization 
Run of O-MEGA pipeline can be with run_opti.py and yaml file (./config_hyperoptimalization.yaml) for a setting hyperparameters. 

1. First, follow the instructions in [Repo setup](#Repo-setup) section
1. Setup specific parameters in `config_hyperoptimalization.yaml`
1. Run python file with active conda environment

```
python ./src/run_opti.py
```
Second option is jupyter notebook On path `./notebooks/hyper_param.ipynb`

### Setting parameters 

Hyperoptimalization creates the possibility of finding the best combination of normalization and explanation method with specific hyperparameters . Using the `Hyper_optimalization` class and the `Hyper_optimalization.run_optimalization()` function, we can use optimalization algorithms from the [Optuna](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html) library to speed up the process of finding the best combination of explainability and normalization methods. Class `Hyper_optimalization` set up several parameters neccesery for evaluation:
- `methods`: (list(string)) methods which will be computed. 
- `normalization`: (list(string)) Normalizations which will be used to normalize explanations. Normalizations must be same as names of functions in `Compare_docano_XAI` class.
- `rationale_path`: (string) path to rationales. Without rationales is not able to do plausibility metrics. 
- `dataset`: (OurDataset) Load dataset with posts and claims
- `model_path` and `embeddings_module_name`: (string) Specific path of model loaded from and their embedding layer 
- `exlanations_path`: (string) path where will be saved and loaded already created explanations
- `plausibility_weights` and `faithfulness_weights`:(float) Weights for groups of metrics (plausibility_weights+faithfulness_weights=1)
- `model_param` and `method_param`: (dict) select specific set of possible hyperparameters
- `explanation_maps_token`, `explanation_maps_word`,`explanation_maps_sentence` : (boolean) Define on which level are explanations post-processed
- `multiple_object`: (boolean) Set Optuna hyperoptimalization to multiple-objective (plausability,faithfulness) optimalization

After creation of explanations, explanations can be post-processed into interpretable_embeddings (`explanation_maps_word`=True) and sentences (`explanation_maps_sentence`=True). Unfortunately, you are able create only one type of explanations during hyperoptimalization. 

### Setting model and method parameters

Setting Hyperparameters describes the `model_param` and `method_param` dictionaries used in the Hyper_optimalization class as variables for configuring explanation methods and their parameters. 

The `model_param` dictionary contains model-specific configurations for explanation methods. It allows customization of how different explanation techniques interact with the model architecture.
- `implemented_method`: (boolean)

The `method_param` dictionary contains method-specific parameters that control how each explanation technique operates. It allows fine-tuning of the explanation generation process. Hyperparameter can be scalar, list or tuple. Each datatype has own reason. Scalar is only one parameter, List contains more parameters and tuple in form: (int,int,step=int) make an interface between the upper and lower bounds of parameters. In float format can be also used 'log=float' besides step argument in tuple. Next important hyperparameters are `compute_baseline` and `token_groups_for_feature_mask`. (Specific methods which need this paramater you can find in [captum documentation](https://captum.ai/api))
- `token_groups_for_feature_mask`: (boolean) Masking of features are neccessery for specific captum methods. If True, pipeline create masks of feature based on token level. 
- `compute_baseline`: (boolean) Controls whether to compute a baseline for the method


<br />
<br />
<br />


# Notebooks examples

On path `notebooks/`, we've prepared a few example notebooks that contain the main logic how to work with this repository. To be precise we've created following Jupyter notebooks:
- `xai.ipynb`, `xai_quant_evaluation.ipynb`, `xai_qual_evaluation.ipynb` - for working with captum XAI attribution methods and for their further evaluation
- `task_evaluation.ipynb` - for evaluating transformer on the task of PFCR (database of claims, query is a post) or invPFCR (database of posts, query is a claim)
- `annotation_creation.ipynb`, `annotation_parsing.ipynb` - for creating annotations for Doccano and their further interpretation to a binary mask representing human rationale
- `hyper_param.ipynb` - for selection of the best normalizations and explanation methods 

<br />
<br />
<br />


# Concepts

## Annotation process

The annotation process consists of the following steps:
- Data selection for annotation - JSON file (`Annotations.create_data_to_annotate`)
- Further annotation processing necessary for Doccano - JSONL file (`Annotations._final_postprocessing_steps_for_doccano`)
- Human evaluation done using Doccano
- Translating human annotations to rationale mask mapped to individual tokens - JSON file (`Annotations(...).create_rationale_masks`)

The translation of human annotations in a form of highligted spans of raw text to rationale masks tied to specific tokens is rather difficult. Individual tokens 
themselves, created by the model tokenizer, may not necessarily have any semantic meaning, hence attributing the rationale to each token may not be desired. 

We'd rather want to find the smallest unit of text that on itself should have a semantic meaning. We came up with a concept of **interpretable tokens** that represent individual words
found in the original text of post or claim.

**Interpretable tokens** could be considered as words that create the entire original text. However these tokens are built from **real tokens**, that are created by the tokenizer. 
Since there exists a mapping between interpretable tokens and the real tokens, we can easily convert data from one format to another. Even though the intepretable tokens are built from the real tokens 
and thus are dependent on the specific tokenizer used, the resulting interpretable tokens are still more generalizable than the real tokens themselves that may vary quite a bit between each tokenizer.

We use the aforementioned interpretable tokens to store the user rationale mask representing the reasoning for matching a specific post and a claim in a JSON format. Furthermore, to compute the
plausibility metrics we convert the mask mapped to the interpretable tokens to the real tokens in order for the measures to work properly.

To summarize, we distinguish 3 types of text formatting:
- real text / string / original text...
- real tokens => created by converting token IDs to their string counterparts
- interpretable tokens => semantically meaningful tokens that tend to be mapped to indivudal real words in the text


## Captum XAI attribution methods integration with Siamese network

In order to calculate the attribution of an input to a specific prediction we firstly need to perform some modifications necessary for our use case to work
- Use of Captum layer called interpretable embeddings (`InterpretableEmbeddingBase`)
    - This allows us to use Captum XAI methods like IntegratedGradients, ... instead of their "layer" counterparts (LayerIntegratedGradients, ...)
    - This class encapsulates the embedding layer of the model and essentially deactivate its functionality of computing embeddings. It simply passes the input it get forward
    - Due to this changed behavior of the embedding layer, we wish to use already computed word embeddings of the input rather than token ids
        - We can compute the word embeddings, and essentially the whole embedding logic of the embedding layer by calling the `InterpretableEmbeddingBase.indices_to_embeddings` function
    - <span style="color:red">UPDATE: Using "Layer" XAI versions may be prefered than using `InterpretableEmbeddingBase` due to the fact there are some assumptions `InterpretableEmbeddingBase` expect the internal implementation of the model to adhere to...</span>
- Computing the input attribution from similarity (STS task)

We can choose not to use the InterpretableEmbeddingBase wrapper, but utilize the "layer" versions of Captum XAI algorithms instead. In such a case, we feed the token IDs to the model (forward function defined in the XAI algorithm) as we traditionally do.

Normally, XAI attribution methods are used to explain the attribution of the input to the prediction of a specific class. In other words, they're used 
when explaining classification tasks. However in our case, we don't have any output neurons representing the individual classes, but rather a single scalar,
a similarity score that describes why a specific post and a claim are similar to each other.

The similarity of a post and a claim is computed from two seperate inputs (post and claim). To quantify the influence of one input to another, we need to 
freeze one of the computational branches so that we do not allow the distribution of backpropagation back to the input. In other words, we precompute one of the embeddings 
used for computing a similarity score and consider it as a constant value instead. This approach allows us to analyze the attribution of one input at the time. 

To summarize, in order to understand why a post is so similar to a specific claim, we precompute the embedding of the claim and use it as a constant when computing the similarity score 
of the post and the claim. Next, by applying XAI method, we propagate the gradients from the similarity score back to the input, to the post itself (or rather to its token embeddings) to
quantify the attribution of each token to the similarity score.


In this project we work with transformers imported from sentence_transformers library. In order for us to have a traditional transformer HF interface for encoding the tokens and computing the 
embedding of a specific input, we use a class `SentenceTransformerToHF`, that wraps the `SentenceTransformer` object and creates more accessible forward function. Subsequently, `STS_ExplainWrapper` class further wraps the logic of computing similarity of two inputs, one of which has already been precomputed, and class `ExplanationExecuter` incorporates the previous wrapper class and a specific Captum attribution XAI algorithm and calculates the attribution of one input to the similarity score.


## XAI evaluation process

In order to evaluate the attribution maps created by multiple explanation methods, we extend [ferret](https://ferret.readthedocs.io/) library that
implements 6 quantitative measures, three of which are faithfulness metrics that doesn't require any annotations, and three of which are plausibility metrics
we've created rationale masks for. Since the ferret library only supports evaluation of attribution on classification tasks we had to further extend their implementation and 
tailor it to our needs, to the STS task.


For O-MEGA pipeline, we use the aforementioned 5 quantitative measures from the Ferret library, divided into two groups: plausibility, feasibility. Also, besides that we added [average precision score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) as an another metric for plausibility.

