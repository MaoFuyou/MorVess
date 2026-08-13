import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
from torch.nn import functional as F
import pickle
import random
import pandas as pd
from tqdm import tqdm
import cv2
import SimpleITK as sitk
from scipy.ndimage import zoom

def process_all_data_to_5slice(data_pth, save_pth):
    """
    [ç»ä¸å¤çå½æ°]
    å°3Då¾åãæ©ç ä»¥åä¸¤ç§è·ç¦»å¾å·é½å¤çæ2Dåçï¼
    æ¯ä¸ªæ ·æ¬åå«5ä¸ªåçä¸ä¸æ(å2ä¸ª, å½å, å2ä¸ª)ã
    """
    print(f"æ°æ®æºè·¯å¾: {data_pth}")
    print(f"è¾åºä¿å­è·¯å¾: {save_pth}")

    # è·åææçä¾æä»¶å¤¹
    patient_dirs = [d for d in os.listdir(data_pth) if d.startswith("PA") and os.path.isdir(os.path.join(data_pth, d))]
    patient_dirs.sort()

    for patient_dir in tqdm(patient_dirs, desc="æ­£å¨å¤ççä¾"):
        case_id = patient_dir
        
        # --- ç¡®ä¿ææç®æ ç®å½é½å­å¨ ---
        os.makedirs(os.path.join(save_pth, case_id, 'images'), exist_ok=True)
        os.makedirs(os.path.join(save_pth, case_id, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(save_pth, case_id, 'boundary_potential'), exist_ok=True)
        os.makedirs(os.path.join(save_pth, case_id, 'internal_distance'), exist_ok=True)
        os.makedirs(os.path.join(save_pth, case_id, 'thickness_map'), exist_ok=True)
        
        # --- å è½½ææåç§æ°æ® ---
        # 1. å¾å
        img_path = os.path.join(data_pth, patient_dir, "image", f"{patient_dir}.nii.gz")
        if not os.path.exists(img_path):
            print(f"è­¦å: å¨ {patient_dir} ä¸­æªæ¾å° image æä»¶ï¼è·³è¿æ­¤çä¾ã")
            continue
        img_arr = nib.load(img_path).get_fdata()

        # 2. æ©ç 
        mask_path = os.path.join(data_pth, patient_dir, "label", f"{patient_dir}.nii.gz")
        if not os.path.exists(mask_path):
            print(f"è­¦å: å¨ {patient_dir} ä¸­æªæ¾å° label æä»¶ï¼è·³è¿æ­¤çä¾ã")
            continue
        mask_arr = nib.load(mask_path).get_fdata()

        # 3. è¾¹çå¿è½å¾ (æºè½æ¥æ¾)
        potential_map_folder = os.path.join(data_pth, patient_dir, 'potential_map')
        boundary_files = [f for f in os.listdir(potential_map_folder) if f.endswith('_boundary_potential.nii.gz')]
        if len(boundary_files) != 1:
            print(f"è­¦å: å¨ {patient_dir}/potential_map ä¸­æ¾å° {len(boundary_files)} ä¸ªè¾¹çå¾ï¼åºä¸º1ä¸ªãè·³è¿ã")
            continue
        boundary_pot_arr = nib.load(os.path.join(potential_map_folder, boundary_files[0])).get_fdata()
        
        # 4. åé¨è·ç¦»å¾ (æºè½æ¥æ¾)
        internal_files = [f for f in os.listdir(potential_map_folder) if f.endswith('_internal_distance.nii.gz')]
        if len(internal_files) != 1:
            print(f"è­¦å: å¨ {patient_dir}/potential_map ä¸­æ¾å° {len(internal_files)} ä¸ªåé¨è·ç¦»å¾ï¼åºä¸º1ä¸ªãè·³è¿ã")
            continue
        internal_dist_arr = nib.load(os.path.join(potential_map_folder, internal_files[0])).get_fdata()
        
        thickness_map_path = os.path.join(data_pth, patient_dir, 'thickness_map', f"{patient_dir}_thickness_map.nii.gz")
        if not os.path.exists(thickness_map_path):
            print(f"è­¦å: å¨ {patient_dir} ä¸­æªæ¾å° thickness_map æä»¶ï¼è·³è¿æ­¤çä¾ã")
            continue
        thickness_map_arr = nib.load(thickness_map_path).get_fdata()

        # --- å¯¹ææåç§æ°æ®è¿è¡åæ ·çpaddingååæ¢æä½ ---
        all_arrays = [img_arr, mask_arr, boundary_pot_arr, internal_dist_arr, thickness_map_arr]
        padded_arrays = [np.concatenate((arr[:, :, 0:1], arr[:, :, 0:1], arr, arr[:, :, -1:], arr[:, :, -1:]), axis=-1) for arr in all_arrays]
        
        # éåçå¤ç
        for slice_indx in range(2, padded_arrays[0].shape[2]-2):
            slice_num = slice_indx - 2
            
            # æå5ä¸ªåçå¹¶è¿è¡åæ ·çåæ¢
            slice_blocks = [arr[:,:,slice_indx-2 : slice_indx+3] for arr in padded_arrays]
            transformed_blocks = [np.flip(np.rot90(block, k=1, axes=(0, 1)), axis=1) for block in slice_blocks]

            # --- ä¿å­ææåç§å¤çåçæ°æ® ---
            paths_and_data = [
                (os.path.join(save_pth, case_id, 'images', f'2Dimage_{slice_num:04d}.pkl'), transformed_blocks[0]),
                (os.path.join(save_pth, case_id, 'masks', f'2Dmask_{slice_num:04d}.pkl'), transformed_blocks[1]),
                (os.path.join(save_pth, case_id, 'boundary_potential', f'2Dboundary_{slice_num:04d}.pkl'), transformed_blocks[2]),
                (os.path.join(save_pth, case_id, 'internal_distance', f'2Dinternal_{slice_num:04d}.pkl'), transformed_blocks[3]),
                (os.path.join(save_pth, case_id, 'thickness_map', f'2Dthickness_{slice_num:04d}.pkl'), transformed_blocks[4])
            ]

            for path, data in paths_and_data:
                with open(path, 'wb') as file:
                    pickle.dump(data, file)


def get_unified_csv(save_pth, test_fd_list=None):
    """
    [ç»ä¸çæå½æ°]
    çæåå«å¨é¨åç§æ°æ®è·¯å¾ç training.csv å test.csv æä»¶
    """
    training_csv = os.path.join(save_pth, 'training.csv')
    test_csv = os.path.join(save_pth, 'test.csv')

    data_fd_list = [d for d in os.listdir(save_pth) if d.startswith('PA') and os.path.isdir(os.path.join(save_pth, d))]
    
    for _ in range(5):
        random.shuffle(data_fd_list)

    test_fd_list = test_fd_list or ['PA000005', 'PA000016', 'PA000024', 'PA000026', 'PA000027', 'PA000036']
    missing_test_cases = sorted(set(test_fd_list) - set(data_fd_list))
    if missing_test_cases:
        raise ValueError(f"Test cases not found below {save_pth}: {missing_test_cases}")
    training_fd_list = list(set(data_fd_list) - set(test_fd_list))

    # --- åå»ºè®­ç»éCSV ---
    path_list_all_train = []
    for data_fd in training_fd_list:
        # ä»¥ 'images' æä»¶å¤¹ä¸ºåºåçæè·¯å¾
        slice_list = os.listdir(os.path.join(save_pth, data_fd, 'images'))
        slice_pth_list = [os.path.join(data_fd, 'images', slice_name) for slice_name in slice_list]
        path_list_all_train.extend(slice_pth_list)
    
    for _ in range(5):
        random.shuffle(path_list_all_train)
    
    # --- åå»ºåå«å¨é¨ååçDataFrame ---
    df_train = pd.DataFrame(path_list_all_train, columns=['image_pth'])
    df_train['mask_pth'] = df_train['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'masks/2Dmask_'))
    df_train['boundary_pth'] = df_train['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'boundary_potential/2Dboundary_'))
    df_train['distance_pth'] = df_train['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'internal_distance/2Dinternal_'))
    df_train['thickness_map_pth'] = df_train['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'thickness_map/2Dthickness_'))
    df_train.to_csv(training_csv, index=False)
    print(f"è®­ç»éCSVå·²çæ: {training_csv}, åå« {len(df_train)} æ¡è®°å½ã")

    # --- åå»ºæµè¯éCSV ---
    path_list_all_test = []
    for data_fd in test_fd_list:
        slice_list = os.listdir(os.path.join(save_pth, data_fd, 'images'))
        slice_list.sort()
        slice_pth_list = [os.path.join(data_fd, 'images', slice_name) for slice_name in slice_list]
        path_list_all_test.extend(slice_pth_list)

    df_test = pd.DataFrame(path_list_all_test, columns=['image_pth'])
    df_test['mask_pth'] = df_test['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'masks/2Dmask_'))
    df_test['boundary_pth'] = df_test['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'boundary_potential/2Dboundary_'))
    df_test['distance_pth'] = df_test['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'internal_distance/2Dinternal_'))
    df_test['thickness_map_pth'] = df_test['image_pth'].apply(lambda x: x.replace('images/2Dimage_', 'thickness_map/2Dthickness_'))
    df_test.to_csv(test_csv, index=False)
    print(f"æµè¯éCSVå·²çæ: {test_csv}, åå« {len(df_test)} æ¡è®°å½ã")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Parse2022 five-slice inputs and optional CSV indices.")
    parser.add_argument("--data_root", required=True, help="Root containing PA*/image, label, potential_map, and thickness_map.")
    parser.add_argument("--output", required=True, help="Destination for the five-slice .pkl hierarchy.")
    parser.add_argument("--build_csv", action="store_true", help="Also write training.csv and test.csv after creating samples.")
    parser.add_argument(
        "--test_cases",
        default="PA000005,PA000016,PA000024,PA000026,PA000027,PA000036",
        help="Comma-separated held-out Parse2022 cases used by the original release.",
    )
    cli_args = parser.parse_args()
    process_all_data_to_5slice(cli_args.data_root, cli_args.output)
    if cli_args.build_csv:
        get_unified_csv(cli_args.output, [item for item in cli_args.test_cases.split(",") if item])
