from core import *
#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"





def main(args):


   # Preprocessing the data to get spike counts, behavioral variables and names, trial IDs, neuron names, and neuron info from manual curation
    counts, variables, variable_names, trial_ids, neu_names, neu_info = preprocess(args.dir, args.save_dir, args.mouse, args.session, args.window_size, args.step_size, args.sigma_smooth, args.use_units, args.fs, args.sfs)

 
    # Loop through the two regions and get the spike rates
    for region in ['OB', 'HC']:
        region_start_time = time.time()
        print(f'Processing region: {region}', flush=True)
        use_units = [key for key, value in neu_info.items() if value['area'] == region]
        spike_rates = counts[:, np.isin(neu_names, use_units)]

        # Loop through the shuffles of the data
        for shift in range(args.n_shifts + 1):
            shift_start_time = time.time()
            print(f'Processing shift: {shift}', flush=True)
            save_dir = os.path.join(args.save_dir, region, f'shift_{shift}')
            if shift > 0:
                random_roll = np.random.randint(0, spike_rates.shape[0])
                print(f'rolling spike rates by {random_roll} time bins')
                spike_rates = np.roll(spike_rates, random_roll, axis=0)
            print(f'spike rates shape (# time bins, # neurons): {spike_rates.shape}', flush=True)

            # Extracting the behavioral variables based on the use_behaviors list
            behavior_components = []
            if 'position_x' in args.use_behaviors:
                pos_x = np.array(variables[variable_names.index('position_x')])
                behavior_components.append(pos_x)

            if 'position_y' in args.use_behaviors:
                pos_y = np.array(variables[variable_names.index('position_y')])
                behavior_components.append(pos_y)

            if 'velocity_x' in args.use_behaviors:
                vel_x = np.array(variables[variable_names.index('velocity_x')])
                behavior_components.append(vel_x)

            if 'velocity_y' in args.use_behaviors:
                vel_y = np.array(variables[variable_names.index('velocity_y')])
                behavior_components.append(vel_y)

            if 'sns' in args.use_behaviors:
                sniff_rate = np.array(variables[variable_names.index('sns')])
                behavior_components.append(sniff_rate)

            if 'latency' in args.use_behaviors:
                latency = np.array(variables[variable_names.index('latency')])
                behavior_components.append(latency)

            if 'phase' in args.use_behaviors:
                phase = np.array(variables[variable_names.index('phase')])
                behavior_components.append(phase)

            if 'speed' in args.use_behaviors:
                speed = np.array(variables[variable_names.index('speed')])
                behavior_components.append(speed)

            if 'clicks' in args.use_behaviors:
                clicks = np.array(variables[variable_names.index('clicks')])
                behavior_components.append(clicks)

            behavior = np.stack(behavior_components, axis=1)
            print(f'behavior shape (# time bins, # dims): {behavior.shape}', flush=True)

            # looping through the folds of cross-validation to build datasets
            arg_list = []
            for k in range(args.k_CV):
                rates_train, rates_test, train_switch_ind, test_switch_ind = cv_split(spike_rates, k, args.k_CV, args.n_blocks)
                behavior_train, behavior_test, _, _ = cv_split(behavior, k, args.k_CV, args.n_blocks)

                current_save_path = os.path.join(save_dir, "model fits")
                os.makedirs(current_save_path, exist_ok=True)


                # Decide which data to use for input (X) and output (y)
                if args.model_input == 'neural':
                    y_train = behavior_train
                    y_test = behavior_test

                    X_train = rates_train
                    X_test = rates_test
                elif args.model_input == 'behavioral':
                    y_train = rates_train
                    y_test = rates_test

                    X_train = behavior_train
                    X_test = behavior_test

                arg_list.append((X_train, X_test, y_train, y_test, train_switch_ind, test_switch_ind, current_save_path, None, shift, k, args.hidden_dim, args.num_layers, args.dropout, args.sequence_length, args.target_index, args.batch_size, args.lr, args.num_epochs, args.patience, args.min_delta, args.factor, args.plot_predictions))


            results = []
            results_save_dir = os.path.join(save_dir, "results")
            os.makedirs(results_save_dir, exist_ok=True)
            for k in range(args.k_CV):
                print(f"Processing fold {k + 1}/{args.k_CV}...", flush=True)
                fold_start_time = time.time()
                rmse, targets, predictions = process_fold(*arg_list[k])

                # Define file paths for saving results
                results_file = f'results_shift{shift}_fold{k}.npz'
                results_path = os.path.join(results_save_dir, results_file)
                
                # Save rmse as compressed .npz file
                np.savez_compressed(results_path, rmse=rmse, targets=targets, predictions=predictions)


                # Store just the path in the results table
                results.append({
                    'mouse': args.mouse,
                    'session': args.session,
                    'shift': shift,
                    'fold': k,
                    'results_file': results_file,
                })

                print(f"Fold {k + 1} completed in {time.time() - fold_start_time:.2f} seconds\n\n", flush=True)

            # convert to DataFrame and save as CSV
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(results_save_dir, 'results.csv'), index=False)

        print(f"Shift {shift} completed in {time.time() - shift_start_time:.2f} seconds\n\n", flush=True)
    print(f"Region {region} completed in {time.time() - region_start_time:.2f} seconds\n\n", flush=True)





if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="Run LSTM decoding model on neural and behavioral data.")
    parser.add_argument('--dir', type=str, default=r"D:\clickbait-mmz",
                        help='Main directory with the data')
    parser.add_argument('--save_dir', type=str, default=r"C:\Users\smearlab\analysis_code\EncodingModels\outputs",
                        help='Main directory to save the figures and results')
    parser.add_argument('--use_behaviors', type=str, default="['position_x', 'position_y', 'velocity_x', 'velocity_y', 'sns', 'latency', 'phase', 'speed']",
                    help='List of behavioral variables to use as input to the model')
    parser.add_argument('--mouse', type=str, default = '6002',
                        help='Mouse to decode')
    parser.add_argument('--session', type=str, default = '9',
                        help='Session to decode (e.g., 1, 2, 3, etc.)')
    parser.add_argument('--window_size', type=float, default=0.1,
                        help='size of the window for spike rate computation in seconds (e.g., 0.1, 0.03)')
    parser.add_argument('--step_size', type=float, default=0.1,
                        help='size of the step for sliding window in seconds (Should be equal to window_size)')
    parser.add_argument('--sigma_smooth', type=float, default=2.5, 
                        help='Standard deviation for Gaussian smoothing of spike rates in time bins (e.g., 2.5, 1.0)')
    parser.add_argument('--use_units', type=str, choices=['good', 'good/mua', 'mua'], default='good/mua',
                        help='What kilosort cluster labels to use')
    parser.add_argument('--n_shifts', type=int, default=10,
                        help='Number of shifts for null distribution construction (0 means no shift)')
    parser.add_argument('--k_CV', type=int, default=10,
                        help='Number of cross-validation folds')
    parser.add_argument('--n_blocks', type=int, default=10,
                        help='Number of blocks for cross-validation')
    parser.add_argument('--plot_predictions', type=bool, default=True,
                        help='Whether to plot the predictions')
    parser.add_argument('--sequence_length', type=int, default=10,
                    help='Define the sequence length for LSTM input')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='Hidden dimension for LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for LSTM')
    parser.add_argument('--num_epochs', type=int, default=1000,
                        help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.01,
                        help='Minimum change to qualify as an improvement')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Factor for learning rate reduction')
    parser.add_argument('--fs', type=int, default=30_000,
                        help='Sampling frequency for spike rates')
    parser.add_argument('--target_index', type=int, default=-1,
                        help='Index of the target variable in the output (default is last variable)')   
    parser.add_argument('--model_input', type=str, choices=['neural', 'behavioral'], default='neural',
                        help='Whether to use neural data or behavioral data as input to the model')
                 
    parser.add_argument('--sfs', type=int, default=1000,
                        help='Sampling frequency for sniff signal')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    args = parser.parse_args()


    # Set the save directory based on mouse and session
    import os
    params_name = f"window_{args.window_size}_sequence_length_{args.sequence_length}_behaviors_{''.join(args.use_behaviors)}"
    save_dir = os.path.join(args.save_dir, params_name, args.mouse, args.session)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    # Convert the use_behaviors string to a list
    import ast
    args.use_behaviors = ast.literal_eval(args.use_behaviors)  # Convert string representation of list to actual list


    # Print the arguments for debugging    # Save the args to a file for reference
    args_file = os.path.join(args.save_dir, 'args.json')
    if not os.path.exists(args_file):
        import json
        with open(args_file, 'w') as f:
            json.dump(vars(args), f, indent=4)

    
    # Run the main function with the parsed arguments
    main(args)
    
    



        