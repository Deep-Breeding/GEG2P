import warnings
warnings.filterwarnings("ignore")
from genetic_algorithm import run
import pandas as pd



def GEG2P_v1(models, method, plant,traits,phe_path,cvf_path,kmax):
    for i in range(1,5):
        run(models[i-1],method[i-1], plant,traits,phe_path,cvf_path,kmax)

def GEG2P_v2(models, method, plant,traits,phe_path,cvf_path,kmax):
    for i in range(1, 5):
        for k in range(1, kmax+1):
            for trait in traits:
                dtt_file_path = f'results/{plant}/k{k}/{trait}.csv'

                w3_file_path = f'results/{plant}/{method[i-1]}/10_weights.csv'
                # Reading the CSV files
                dtt_df = pd.read_csv(dtt_file_path)
                w3_df = pd.read_csv(w3_file_path)
                print(w3_file_path)
                w3_trait_weights = w3_df[(w3_df['Trait'] == trait) & (w3_df['Way'] == f'T{k}')].iloc[0]
                # Automatically get model names that are common in both datasets
                common_models = list(set(dtt_df.columns).intersection(w3_trait_weights.index))

                # Extract the relevant weights for the common models
                model_weights = {model: w3_trait_weights[model] for model in common_models}

                # Check if the calculated column already exists in dtt_df to avoid duplication
                col_name =method[i-1]

                dtt_df[col_name] = sum(dtt_df[model] * weight for model, weight in model_weights.items())

                # Save the updated DataFrame by appending the new columns
                dtt_df.to_csv(dtt_file_path, index=False)

                print(f"Results for {trait} saved to {dtt_file_path}")

    run(models, "GEG2P(v2)" ,plant,traits,phe_path,cvf_path,kmax)

#v3——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
import pandas as pd
import os
import copy

def GEG2P_v3(model5, method, plant,traits,phe_path,cvf_path,kmax):
    """
    Greedily remove the worst performing model from the model combination until GEG2P(v3) no longer improves.
    Run run(model5, "GEG2P(v3)", ...) each time and update avg_pcc.csv, compare whether the value of GEG2P(v3) improves.
    """

    # Initial model combination
    current_model5 = copy.deepcopy(model5)
    last_GEG2P_v2_scores = {}

    for trait in traits:
        print(f"Processing trait: {trait}")
        improved = True
        history_model5 = []

        while improved :

            # Record current combination
            history_model5.append(copy.deepcopy(current_model5))

            # Run model
            run(current_model5, method,plant, [trait], phe_path, cvf_path,kmax)

            # Read GEG2P(v2) scores
            avg_pcc_path = f"results/{plant}/{method}/{trait}/avg_pcc.csv"
            if not os.path.exists(avg_pcc_path):
                raise FileNotFoundError(f"File not found: {avg_pcc_path}")
            df = pd.read_csv(avg_pcc_path, index_col=0)

            current_score = df.loc[trait, "GEG2P(v3)"]
            last_score = last_GEG2P_v2_scores.get(trait, -float("inf"))

            print(f"Trait: {trait}, \ncurrent_score: {current_score}, \nLast score: {last_score}")


            if current_score > last_score:
                last_GEG2P_v2_scores[trait] = current_score

                # Remove the worst model from current model combination
                df_row = df.loc[trait]
                min_model = df_row[current_model5].idxmin()
                print("————————————————————————————————————————————————————————————")
                print(f"Try to remove : {min_model}")
                current_model5.remove(min_model)
            else:
                print(f"No improvement for trait: {trait}. Stopping.")
                improved = False

        # Restore the last improved model combination
        if len(history_model5) >= 2:
            best_model5 = history_model5[-2]  # Previous valid combination
            print(f"Re-running best model5 for trait {trait}: {best_model5}")
            run(best_model5, method,plant, [trait], phe_path, cvf_path,kmax)
        elif len(history_model5) == 1:
            best_model5 = history_model5[0]
            print(f"Only one valid model5 tried. Using it for trait {trait}: {best_model5}")
            run(best_model5, method,plant, [trait], phe_path, cvf_path,kmax)
        else:
            print(f"No valid model5 combinations recorded for trait {trait}. Skipping.")

