import sys
sys.path.append("../src")

import optuna
import warnings
from captum.attr._core.lime import get_exp_kernel_similarity_function
from captum._utils.models.linear_model import SkLearnLasso
from compare_docano_XAI import Hyper_optimalization,Check_docano_XAI,Visualization_opt,Compare_docano_XAI
from explain import STS_ExplainWrapper,SentenceTransformerToHF
from dataset import OurDataset
import architecture
import torch
import yaml

def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def create_hyper_opt_object(config):
    hyper_config = config.get('Hyperoptimalization_parameters', {})
    dataset_path = hyper_config.pop('dataset', None)
    dataset = OurDataset(csv_dirpath=dataset_path)
    return Hyper_optimalization(dataset=dataset, **hyper_config)

def run_optimalization_function(hyper_opt_object):
    results_config=config.get('Optuna_parameters', {})
    best,trials,Visualization_opt=hyper_opt_object.run_optimization(**results_config)
    return best,trials,Visualization_opt

def Visualize_class_activation(Visualization_opt):
    results_config=config.get('Results_parameters', {})
    optional_keys = ['table_counter','table_sampler','visual_aopc']

    for key in optional_keys:
        if key in results_config.keys():
            Visualization_opt.visual_aopc(save_path_plot=key)


if __name__ == "__main__":
    config = load_config("config_hyperoptimalization.yaml")
    hyper_opt = create_hyper_opt_object(config)
    best,trials,Visualization_opt=run_optimalization_function(hyper_opt)
    Visualize_class_activation(Visualization_opt)