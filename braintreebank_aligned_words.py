# Part of the code is adapted from https://braintreebank.dev/, file "quickstart.ipynb"
import os
import json
import pandas as pd
import numpy as np
from braintreebank_config import *

# Data frames column IDs
start_col, end_col, lbl_col = 'start', 'end', 'pos'
trig_time_col, trig_idx_col, est_idx_col, est_end_idx_col = 'movie_time', 'index', 'est_idx', 'est_end_idx'
word_time_col, word_text_col, is_onset_col, is_offset_col = 'word_time', 'text', 'is_onset', 'is_offset'


def obtain_estimated_sample_index(trigs_df, movie_time):
    last_t = trigs_df[trig_time_col].iloc[-1]
    last_t_idx = trigs_df[trig_idx_col].idxmax()

    start_index = np.searchsorted(trigs_df[trig_time_col].values, movie_time)
    return int(trigs_df.loc[start_index, trig_idx_col] + (movie_time - trigs_df.loc[start_index, trig_time_col]) * SAMPLING_RATE)



def obtain_aligned_words_trigs_df(sub_id, trial_id, verbose=True, save_to_dir=None):
    # Path to trigger times csv file
    trigger_times_file = os.path.join(ROOT_DIR, f'subject_timings/sub_{sub_id}_trial{trial_id:03}_timings.csv')
    # Path format to trial metadata json file
    metadata_file = os.path.join(ROOT_DIR, f'subject_metadata/sub_{sub_id}_trial{trial_id:03}_metadata.json')
    with open(metadata_file, 'r') as f:
        meta_dict = json.load(f)
        title = meta_dict['title']
        movie_id = meta_dict['filename']
    # # Path to transcript csv file
    transcript_file_format = os.path.join(ROOT_DIR, f'transcripts/{movie_id}/features.csv')
    # Path format to electrode labels file -- mapping each ID to an subject specific label
    electrode_labels_file = os.path.join(ROOT_DIR, f'electrode_labels/sub_{sub_id}/electrode_labels.json')

    if verbose: print(f"Computing words dataframe for subject {sub_id} trial {trial_id}")
    trigs_df = pd.read_csv(trigger_times_file)
    words_df = pd.read_csv(transcript_file_format.format(movie_id)).set_index('Unnamed: 0')
    words_df = words_df.drop(['word_diff', 'onset_diff'], axis=1) # remove those columns because they are unnecessary and cause excessive filtering with NaN values
    words_df = words_df.dropna().reset_index(drop=True)

    # Vectorized sample index estimation
    def add_estimated_sample_index_vectorized(w_df, t_df):
        last_t = t_df[trig_time_col].iloc[-1]
        last_t_idx = t_df[trig_idx_col].idxmax()
        w_df = w_df[w_df[start_col] < last_t].copy()

        # Vectorized nearest trigger finding
        start_indices = np.searchsorted(t_df[trig_time_col].values, w_df[start_col].values)
        end_indices = np.searchsorted(t_df[trig_time_col].values, w_df[end_col].values)
        end_indices = np.minimum(end_indices, last_t_idx) # handle the edge case where movie cuts off right at the word
        start_indices = np.maximum(start_indices, 0) # handle the edge case where movie starts right at the word
        
        # Vectorized sample index calculation
        w_df[est_idx_col] = np.round(
            t_df.loc[start_indices, trig_idx_col].values + 
            (w_df[start_col].values - t_df.loc[start_indices, trig_time_col].values) * SAMPLING_RATE
        )
        w_df[est_end_idx_col] = np.round(
            t_df.loc[end_indices, trig_idx_col].values + 
            (w_df[end_col].values - t_df.loc[end_indices, trig_time_col].values) * SAMPLING_RATE
        )
        return w_df

    words_df = add_estimated_sample_index_vectorized(words_df, trigs_df)  # align all words to data samples
    words_df = words_df.dropna().reset_index(drop=True)  # no need to keep words with no start time

    # Remove words that would create invalid windows (too close to the start or end of the trial)
    total_samples = trigs_df.loc[len(trigs_df) - 1, trig_idx_col]
    words_df = words_df.reset_index(drop=True)

    if verbose: print(f"Kept {len(words_df)} words after removing invalid windows")
    # Save the processed words dataframe
    if save_to_dir is not None:
        if not os.path.exists(save_to_dir):
            os.makedirs(save_to_dir)
        words_df.to_csv(f'{save_to_dir}/subject{sub_id}_trial{trial_id}_words_df.csv', index=False)
    return words_df, trigs_df