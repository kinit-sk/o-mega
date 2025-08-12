import matplotlib.pyplot as plt
import copy
import numpy as np
import pandas as pd


class Visualization_opt:
    def __init__(self,methods,normalizations,counter,counter_normalizations,counter_normalizations_dup,counter_methods,counter_methods_dup,visualizations_metric,study,number_of_trials,time_opt,sampler,current,peak,additional_results,best):
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
        self.best=best
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
            plt.savefig(f"{save_path_plot}_{vis_list[0]['metric']}.png", bbox_inches='tight')
        plt.show()
        plt.close() 

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




    def visual_aopc(self,save_path_plot:str):
            list_filtred_vis_suff=[]
            list_filtred_vis_com=[]
            if self.visualizations_metric: 
                for method in self.methods:
                    filtered_trials = [ trial for trial in self.study.trials if trial.params['method'] == method]
                    best_trial=Visualization_opt.choose_best(filtered_trials)
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




