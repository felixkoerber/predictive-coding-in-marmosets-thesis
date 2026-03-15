import os
import pickle
import numpy as np
import mne
import mne_bids
from idtxl.multivariate_te import MultivariateTE
from idtxl.data import Data
from acme import ParallelMap, bic_cluster_setup

# --- DATA PREPARATION FUNCTION ---
def get_gfp_threshold(epochs, percentile=95):

    data = epochs.get_data(copy=False)
    
    gfp_per_epoch = np.std(data, axis=1)
    max_gfp_per_epoch = gfp_per_epoch.max(axis=1)
    
    # Determine the threshold
    thresh = np.percentile(max_gfp_per_epoch, percentile)
    return thresh
def standard_and_deviants(subject, subjects=["Fr","Go", "Kr"],picks="all"): # 
    # Constants
    RESAMPLE_RATE = 500
    NOTCH_FREQS = np.arange(50, RESAMPLE_RATE/2, 50)
    STANDARD_EVENT_OFFSET = 503
    EPOCH_TMIN, EPOCH_TMAX = -0.1, 0.25
    LARGE_TMIN, LARGE_TMAX = -0.603, 0.25

    all_epochs_deviant = []
    all_epochs_standard = []
    all_large_epochs = []

    
    subject_name = subjects[subject]
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    bids_root = os.path.join(project_root, "BIDS")
    #bids_root = os.path.join(cwd, "BIDS")
    task = "oddball"

    # Get subject name and session directories
    subject_name = subjects[subject]
    sub_dirs = os.listdir(os.path.join(cwd, bids_root, f"sub-{subject_name}"))
    print(bids_root)
    print(f"Processing {len(sub_dirs)} sessions for subject {subject_name}")

    for session_idx in range(len(sub_dirs)):
        session_num = session_idx + 1

        # Load raw data
        bids_path = mne_bids.BIDSPath(
            subject=subject_name,
            session=str(session_num),
            task=task,
            acquisition="01",
            run="01",
            root=bids_root,
        )
        # 1. Load Data
        raw = mne_bids.read_raw_bids(bids_path=bids_path, verbose="CRITICAL")
        raw.load_data(verbose="CRITICAL")
        raw.pick(picks)
        raw.apply_function(lambda x: x * 1e-6) 
        # 2. Notch and High-Pass Filter
        raw.filter(0.3, None, verbose="CRITICAL")
        raw.notch_filter(freqs=NOTCH_FREQS, verbose="CRITICAL")
        raw.resample(RESAMPLE_RATE, verbose="CRITICAL")
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose="CRITICAL")

  
        # 6. Extract Events & Epoching
        events, _ = mne.events_from_annotations(raw, verbose="CRITICAL")
        
        # Create standard events by shifting the timing
        standard_events = events.copy()
        standard_events[:, 0] -= int(STANDARD_EVENT_OFFSET * (RESAMPLE_RATE / 1000))
        

        # Define common kwargs for epoching
        epoch_params = dict(tmin=EPOCH_TMIN, tmax=EPOCH_TMAX, baseline=(-0.1, 0), 
                            preload=True, verbose="CRITICAL")

        all_epochs_deviant.append(mne.Epochs(raw, events=events, **epoch_params))
        all_epochs_standard.append(mne.Epochs(raw, events=standard_events, **epoch_params))
        
        # Large Epoch for sync rejection
        all_large_epochs.append(mne.Epochs(raw, events=events, 
                                           tmin=LARGE_TMIN, tmax=LARGE_TMAX, 
                                           baseline=(-0.1, 0), preload=True))

    # 7. Concatenate and Final Rejection
    epochs_deviant = mne.concatenate_epochs(all_epochs_deviant)
    epochs_standard = mne.concatenate_epochs(all_epochs_standard)
    large_epochs = mne.concatenate_epochs(all_large_epochs)

    # Use the GFP-based thresholding on the large epochs
    gfp_thresh = get_gfp_threshold(large_epochs, percentile=95)
    
    # Calculate GFP peak per epoch
    data = large_epochs.get_data(copy=False)
    max_gfps = np.std(data, axis=1).max(axis=1)
    bad_epoch_mask = max_gfps > gfp_thresh
    indices_to_drop = np.where(bad_epoch_mask)[0]

    # Drop from all sets to maintain synchronization
    epochs_deviant.drop(indices_to_drop, reason="GFP_THRESHOLD")
    epochs_standard.drop(indices_to_drop, reason="GFP_THRESHOLD")

    return epochs_standard, epochs_deviant



def f(task_info, te_settings):
    import os
    import pickle
    import numpy as np
    from idtxl.multivariate_te import MultivariateTE
    from idtxl.data import Data

    # Unpack the task details
    subject, condition, target_idx, data_path, window_idx, t_start, t_end = task_info
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define results folder inside the script's directory
    output_dir = os.path.join(script_dir, "te_results_perm_in_time", subject, condition, f"win_{window_idx:03d}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # from each loading the full array into RAM simultaneously.
    full_data_arr = np.load(data_path, mmap_mode='r') 	

    # Extract the 100ms window: [samples, channels, repetitions]
    window_slice = full_data_arr[t_start:t_end, :, :]
    
    # Initialize IDTxl with the ensemble (repetitions) dimension
    te_data = Data(window_slice, dim_order="spr", normalise=True)

    network_analysis = MultivariateTE()
    # analyse_single_target is used to parallelize across target nodes
    result = network_analysis.analyse_single_target(
        settings=te_settings, target=target_idx, data=te_data
    )

    save_name = os.path.join(output_dir, f"target_{target_idx}.pkl")
    with open(save_name, "wb") as f_out:
        pickle.dump(result, f_out)
    
    return save_name 

if __name__ == "__main__":
    subjects = ["Fr", "Go", "Kr"]#["Fr"
    conditions = ["stand", "dev"]
    picks_indices = [[7,9,11,31,24,28,1,17],[32,52,63,56,47,40,31,7],[32,51,61,55,45,40,28,7]]
    
    # Timing parameters
    FS = 500  
    WIN_SIZE_SAMP = 50  # 100 ms
    STEP_SAMP = 50      # 20 ms (Smooth step)
    
    task_list = []
    tmp_files = []
    
    for i, subj in enumerate(subjects):
    	if i!=4:
            ep_stand, ep_dev = standard_and_deviants(i, picks=picks_indices[i])
            
            # Use the full epoch range to allow for sliding
            cond_data = {
                "stand": ep_stand.get_data(copy=False),
                "dev":   ep_dev.get_data(copy=False)
            }

            for cond_name, raw_arr in cond_data.items():
                # IDTxl 'spr' format: (samples, channels, repetitions)
                formatted_data = np.transpose(raw_arr, (2, 1, 0)).astype(np.float32)
                
                # Save as raw .npy for memory-mapping
                npy_name = f"data_{subj}_{cond_name}.npy"
                np.save(npy_name, formatted_data)
                tmp_files.append(npy_name)
        
                n_samples = formatted_data.shape[0]
                n_channels = formatted_data.shape[1]
                
                # Generate windows based on 20ms steps
                win_starts = range(0, n_samples - WIN_SIZE_SAMP + 1, STEP_SAMP)
                    
                script_dir = os.path.dirname(os.path.abspath(__file__))

                for w_idx, t_start in enumerate(win_starts):
                    t_end = t_start + WIN_SIZE_SAMP
                    for ch_idx in range(n_channels):
                        expected_output = os.path.join(
                            script_dir, "te_results_perm_in_time",
                            subj, cond_name, f"win_{w_idx:03d}",
                            f"target_{ch_idx}.pkl"
                        )
                        if os.path.exists(expected_output):
                            continue  # Already computed, skip
                        task_list.append((
                            subj, cond_name, ch_idx,
                            os.path.abspath(npy_name),
                            w_idx, t_start, t_end
                        ))

    # --- CLUSTER SETTINGS ---
    te_settings = {
        "cmi_estimator": "JidtKraskovCMI",
        "max_lag_sources": 12, 
        "min_lag_sources": 1,
        "max_lag_target": 12,
	"tau_target": 3, 
        "fdr_correction": True,   
        "n_perm_max_stat": 200,
	"permute_in_time": True,
        "normalise": True,
        "verbose": True
    }

    # Setup the cluster context
    client = bic_cluster_setup(
        partition="64GBLppc", 
        n_workers=2, 
        mem_per_worker="20GB"
    )

    print(f"Submitting {len(task_list)} tasks to 70 workers...")

    with ParallelMap(f, task_list, te_settings, write_worker_results=False) as pmap:
        pmap.compute()

    # Cleanup temp files
    for f_temp in tmp_files:
        if os.path.exists(f_temp): os.remove(f_temp)