import os
from time import sleep

import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from deap import base, creator, tools, algorithms
import random



POP_SIZE = 50
NGEN = 40
CXPB = 0.7
MUTPB = 0.2
THRESHOLD = 1e-4






def calculate_metrics(y_true, y_pred):
    r, _ = pearsonr(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    return r, mse, rmse


def average_ensemble(predictions):
    avg_predictions = predictions.mean(axis=1)
    return avg_predictions

def run(models, method, plant,traits,phe_path,cvf_path,kmax):
    seed = 41
    np.random.seed(seed)
    random.seed(seed)

    # Initialize an empty DataFrame to store all results
    results_df = pd.DataFrame()

    # Column names are trait, fold number, model name (including genetic algorithm)
    column_names = ['Trait', 'Way'] + models + [method]
    results = pd.DataFrame(columns=column_names)

    # Define genetic algorithm
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    def init_individual():
        return creator.Individual(np.random.rand(len(models)))

    toolbox = base.Toolbox()
    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def create_evaluate(y_true, prediction_columns):
        def evaluate(individual):
            individual = np.clip(individual, 1e-5, 1)
            weighted_predictions = np.dot(prediction_columns, individual)

            if np.sum(individual) > 0:
                weighted_predictions /= np.sum(individual)
            else:
                return float('inf'),  # Avoid invalid individuals

            if np.any(np.isnan(weighted_predictions)):
                return float('inf'),  # Prevent NaN

            rmse = math.sqrt(mean_squared_error(y_true, weighted_predictions))
            return rmse,

        return evaluate

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=1e-5, up=1, eta=0.1, indpb=0.2)

    def kaiming_genetic_algorithm(prediction_columns, predictions, y_test):
        evaluate = create_evaluate(y_test, prediction_columns)
        toolbox.register("evaluate", evaluate)

        pop = toolbox.population(n=POP_SIZE)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        model_rmse = {}
        for model in models:
            y_pred = predictions[model]
            _, _, rmse = calculate_metrics(y_test, y_pred)
            model_rmse[model] = rmse

        kaiming_init = 1.0
        initial_weights = np.array([kaiming_init / (model_rmse[model] + 1e-5) for model in models])
        initial_weights /= np.sum(initial_weights)
        initial_individual = creator.Individual(initial_weights)

        pop[0] = initial_individual
        



        for generation in range(NGEN):
            pop, log = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=1,
                                           stats=stats, halloffame=hof, verbose=False)

            best_individual = hof[0]


            best_individual = np.clip(best_individual, 1e-5, 1)
            best_individual[best_individual < THRESHOLD] = 0
            best_individual /= np.sum(best_individual)

            all_best_individuals.append([generation] + best_individual.tolist())




        # Get the best individual and save
        best_weights = hof[0]

        # Process weights of the best individual
        best_weights = np.clip(best_weights, 1e-5, 1)
        best_weights[best_weights < THRESHOLD] = 0
        best_weights /= np.sum(best_weights)

        weighted_predictions = np.dot(prediction_columns, best_weights)
        if np.sum(best_weights) > 0:
            weighted_predictions /= np.sum(best_weights)

        return best_weights, weighted_predictions

    # Initialize result DataFrame containing trait names and Pearson R values for each fold
    column_names = ['Trait', 'Way'] + models + [method]
    final10_pcc = pd.DataFrame(columns=column_names)

    avg_pcc_df = pd.DataFrame(columns=column_names)

    se_pcc_df = pd.DataFrame(columns=column_names)

    final10_mse = pd.DataFrame(columns=column_names)
    # Initialize DataFrame to save inter-fold MSE

    avg_mse_df = pd.DataFrame(columns=column_names)
    se_mse_df = pd.DataFrame(columns=column_names)
    # Initialize weight DataFrame to save weights for each fold
    weight_columns = ['Trait', 'Way'] + [f'{model}' for model in models]
    weights_df = pd.DataFrame(columns=weight_columns)

    # Loop through each trait
    for trait in traits:
        all_best_individuals = []
        fold_results_r = []  # Save Pearson R results for each fold
        fold_weights = []  # Save weights for each fold
        fold_results_mse = []  # Save MSE results for each fold
        pcc_values = {model: [] for model in models}  # Used to store PCC of each model in 10 folds
        mse_values = {model: [] for model in models}  # Used to store MSE of each model in 10 folds
        pcc_values[method] = []  # Store PCC of genetic algorithm
        mse_values[method] = []  # Store MSE of genetic algorithm
        weight_values = {model: [] for model in models}  # Used to store weights of each model

        for k in range(1, kmax+1):  # Iterate through each fold
            # Read original data
            phe_data = pd.read_csv(phe_path)       # Phenotype data
            cvf = pd.read_csv(cvf_path)            # Cross-validation split file
            predictions_path = f'results/{plant}/k{k}/{trait}.csv'
            predictions = pd.read_csv(predictions_path)  # Prediction results

            # Unify ID column
            # If there is no ID column, rename the first column to ID
            if 'ID' not in phe_data.columns:
                phe_data = phe_data.rename(columns={phe_data.columns[0]: 'ID'})
            phe_data['ID'] = phe_data['ID'].astype(str)

            # CVF file: prioritize ID column, otherwise report error
            if 'ID' in cvf.columns:
                cvf['ID'] = cvf['ID'].astype(str)
                test_ids = cvf.loc[cvf['cv_1'] == k, 'ID'].tolist()
            else:
                raise ValueError(f"No ID column found in CVF file for fold {k}.")

            # Prediction file: if there is no ID column, also rename the first column to ID
            if 'ID' not in predictions.columns:
                predictions = predictions.rename(columns={predictions.columns[0]: 'ID'})
            predictions['ID'] = predictions['ID'].astype(str)

            # Construct y_true and align with predictions
            # Phenotype only keeps ID and current trait
            ytrue_df = phe_data[['ID', trait]].copy()
            # Prediction only keeps ID and model columns
            pred_df = predictions[['ID'] + models].copy()

            # Take intersection through inner merge to ensure ID exists on both sides
            merged = pred_df.merge(ytrue_df, on='ID', how='inner')

            # Only keep samples belonging to current fold test_ids
            merged = merged[merged['ID'].isin(test_ids)]

            # Sort by test_ids order to ensure strict alignment of y and y_pred
            order_cat = pd.Categorical(merged['ID'], categories=test_ids, ordered=True)
            merged = merged.assign(_order=order_cat).sort_values('_order').drop(columns=['_order'])

            # If no samples are aligned, report error
            if merged.empty:
                raise ValueError(f"True value-prediction value samples for fold {k} cannot be aligned, please check sample naming.")

            # Convert prediction values to numeric and fill missing values
            for model in models:
                merged[model] = merged[model].astype(str).str.strip('[]')
                # Convert to numeric, set unparseable values to NaN, fill later
                merged[model] = pd.to_numeric(merged[model], errors='coerce')

            # Fill missing values with 0.5, maintain original logic
            merged[models] = merged[models].fillna(0.5)

            # y_test is true phenotype value; predictions_aligned is the aligned prediction matrix
            y_test = merged[trait].values
            predictions_aligned = merged[models].copy()

            # Calculate metrics model by model
            result_row_r = {'Trait': trait, 'Way': f'T{k}'}
            result_row_mse = {'Trait': trait, 'Way': f'T{k}'}
            result_row_weights = {'Trait': trait, 'Way': f'T{k}'}

            for model in models:
                y_pred = predictions_aligned[model].values
                r, mse, _ = calculate_metrics(y_test, y_pred)
                result_row_r[model] = r
                result_row_mse[model] = mse
                pcc_values[model].append(r)
                mse_values[model].append(mse)

            # Genetic algorithm weighted prediction
            prediction_columns = predictions_aligned[models].values
            best_weights, weighted_predictions = kaiming_genetic_algorithm(
                prediction_columns, predictions_aligned, y_test
            )
            r, mse, _ = calculate_metrics(y_test, weighted_predictions)
            result_row_r[method] = r
            result_row_mse[method] = mse
            pcc_values[method].append(r)
            mse_values[method].append(mse)

            # Save weights corresponding to each model
            for i, model in enumerate(models):
                result_row_weights[f'{model}'] = best_weights[i]

            # Write results to corresponding lists
            fold_weights.append(result_row_weights)
            fold_results_r.append(result_row_r)
            fold_results_mse.append(result_row_mse)

        fold_results_df_r = pd.DataFrame(fold_results_r)
        final10_pcc = pd.concat([final10_pcc, fold_results_df_r], ignore_index=True)

        fold_results_df_mse = pd.DataFrame(fold_results_mse)
        final10_mse = pd.concat([final10_mse, fold_results_df_mse], ignore_index=True)  # Merge MSE results

        fold_weights_df = pd.DataFrame(fold_weights)
        weights_df = pd.concat([weights_df, fold_weights_df], ignore_index=True)

        # Calculate and save average PCC and SE
        avg_pcc = {model: np.nanmean(pcc_values[model]) for model in models}
        avg_pcc[method] = np.nanmean(pcc_values[method])

        se_values = {model: np.nanstd(pcc_values[model]) / math.sqrt(kmax) for model in models}
        se_values[method] = np.nanstd(pcc_values[method]) / math.sqrt(kmax)

        # Calculate and save average MSE and SE
        avg_mse = {model: np.mean(mse_values[model]) for model in models}
        avg_mse[method] = np.mean(mse_values[method])

        mse_se_values = {model: np.std(mse_values[model]) / math.sqrt(kmax) for model in models}
        mse_se_values[method] = np.std(mse_values[method]) / math.sqrt(kmax)

        # Save calculation results to SE DataFrame
        avg_row = pd.DataFrame({'Trait': [trait], 'Way': ['avg'], **avg_pcc})
        avg_pcc_df = pd.concat([avg_pcc_df, avg_row], ignore_index=True)

        avg_row_se = pd.DataFrame({'Trait': [trait], 'Way': ['SE'], **se_values})
        se_pcc_df = pd.concat([se_pcc_df, avg_row_se], ignore_index=True)

        # Save average MSE and SE results to new se_MSE DataFrame
        avg_row_mse = pd.DataFrame({'Trait': [trait], 'Way': ['avg'], **avg_mse})
        avg_mse_df = pd.concat([avg_mse_df, avg_row_mse], ignore_index=True)

        row_se_mse = pd.DataFrame({'Trait': [trait], 'Way': ['SE'], **mse_se_values})
        se_mse_df = pd.concat([se_mse_df, row_se_mse], ignore_index=True)

        df = pd.DataFrame(all_best_individuals, columns=['Generation'] + models)  # Add Generation column
        # Group by generation and calculate average of each group
        df = df.groupby('Generation').mean()
        os.makedirs(f'results/{plant}/{plant}_GA_process', exist_ok=True)
        df.to_csv(f'results/{plant}/{plant}_GA_process/{trait}_{NGEN}.csv', header=True)



    # Save results
    if method=="GEG2P(v3)":
        result_path=f"results/{plant}/{method}/{traits[0]}/"
    else:
        result_path = f"results/{plant}/{method}/"
    os.makedirs(result_path, exist_ok=True)
    final10_pcc.to_csv(result_path+"10_pcc.csv", index=False)
    avg_pcc_df.to_csv(result_path+"avg_pcc.csv", index=False)
    se_pcc_df.to_csv(result_path+"se_pcc.csv", index=False)
    final10_mse.to_csv(result_path+"10_MSE.csv", index=False)
    avg_mse_df.to_csv(result_path+"avg_MSE.csv", index=False)
    se_mse_df.to_csv(result_path+"se_MSE.csv", index=False)
    weights_df.to_csv(result_path+"10_weights.csv", index=False)

    models = [col for col in weights_df.columns if col not in ['Trait', 'Way']]
    traits = weights_df['Trait'].unique()

    # Pre-create a result list to store all results
    result_list = []
    result_list_se = []
    for trait in traits:
        model_data = weights_df[weights_df['Trait'] == trait][models]
        avg_weight = model_data.mean()
        se = model_data.std() / math.sqrt(kmax)
        # Add results to the result list
        result_list.append({'Trait': trait, 'Way': 'avg', **avg_weight})
        result_list_se.append({'Trait': trait, 'Way': 'SE', **se})

    # Convert result list to DataFrame
    results = pd.DataFrame(result_list)
    result_se = pd.DataFrame(result_list_se)
    # Save results
    results.to_csv(result_path+"avg_weights.csv", index=False)
    result_se.to_csv(result_path+"se_weights.csv", index=False)
    print("Final results, weights, MSE, SE, and weight SE saved.")



