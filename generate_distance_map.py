# -*- coding: utf-8 -*-
"""
æ­¤èæ¬ç¨äºå¤ç 3D NIfTI æ ¼å¼çè¡ç®¡åå²æ©æ¨¡ï¼maskï¼ï¼å¹¶çæä¸¤ç§ç±»åçè¡ç®¡è·ç¦»å¾ï¼
1. åé¨è·ç¦»å¾ (Internal Distance Map): ä»å¨è¡ç®¡åé¨æ¾ç¤ºå°èæ¯çæç­è·ç¦»ï¼åæ è¡ç®¡çä¸­å¿çº¿åååº¦ã
2. è¾¹çå¿åºå¾ (Boundary Potential Map): å¨è¡ç®¡è¾¹çå¤å¼æé«ï¼ååå¤å¹³æ»è¡°åï¼ç¨äºå¼å¯¼ç½ç»å­¦ä¹ è¾¹çã

è¯·ç¡®ä¿å·²å®è£å¿è¦çåº:
pip install SimpleITK numpy scipy
"""
import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
import os
import argparse

def generate_distance_maps(mask_path, output_dir, lambda_param=0.5):
    """
    ä»ä¸ä¸ª NIfTI æ ¼å¼çæ©æ¨¡æä»¶çæè¡ç®¡è·ç¦»å¾ã

    åæ°:
    - mask_path (str): è¾å¥çæ©æ¨¡æä»¶è·¯å¾ (.nii.gz)ã
    - output_dir (str): è¾åºæä»¶çä¿å­ç®å½ã
    - lambda_param (float): è¾¹çå¿åºå¾ä¸­ææ°è¡°åçç³»æ°ï¼å¯æ ¹æ®éè¦è°æ´ã
                           å¼è¶å°ï¼å¿åºèå´è¶å¹¿ï¼å¼è¶å¤§ï¼å¿åºè¶éä¸­å¨è¾¹çã
    """
    if not os.path.exists(mask_path):
        print(f"éè¯¯ï¼è¾å¥æä»¶ä¸å­å¨ -> {mask_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"åå»ºè¾åºç®å½: {output_dir}")

    print(f"æ­£å¨å¤çæä»¶: {mask_path}")

    # 1. ä½¿ç¨ SimpleITK å è½½ NIfTI æä»¶
    # SimpleITK å¯ä»¥å¾å¥½å°ä¿çå¾åçåæ°æ®ï¼å¦spacing, origin, directionï¼
    mask_image = sitk.ReadImage(mask_path, sitk.sitkFloat32)
    mask_array = sitk.GetArrayFromImage(mask_image)

    # ç¡®ä¿æ©æ¨¡æ¯äºå¼ç (0 æ 1)
    mask_array = (mask_array > 0).astype(np.uint8)

    # --- çæåé¨è·ç¦»å¾ (Internal Distance Map) ---
    # è®¡ç®ä»æ¯ä¸ªåæ¯ç¹ï¼è¡ç®¡ï¼å°æè¿èæ¯ç¹ï¼éè¡ç®¡ï¼çæ¬§å éå¾è·ç¦»ã
    # æä»¬éè¦è·åå¾åçspacingä¿¡æ¯ï¼ä½¿å¾è·ç¦»è®¡ç®æ¯ç©çè·ç¦»ï¼ä¾å¦æ¯«ç±³ï¼ï¼èä¸æ¯åç´ åä½ã
    spacing = mask_image.GetSpacing()
    
    # distance_transform_edt ä¼è®¡ç®æ¯ä¸ªéé¶ç¹å°æè¿é¶ç¹çè·ç¦»
    # è¿æ­£æ¯æä»¬æ³è¦çåé¨è·ç¦»å¾
    internal_dist_array = distance_transform_edt(mask_array, sampling=spacing)
    
    # å°ç»æè½¬æ¢å SimpleITK å¾å
    internal_dist_image = sitk.GetImageFromArray(internal_dist_array)
    internal_dist_image.CopyInformation(mask_image) # å¤å¶ææåæ°æ®

    # ä¿å­åé¨è·ç¦»å¾
    base_name = os.path.basename(mask_path).replace(".nii.gz", "")
    output_path_internal = os.path.join(output_dir, f"{base_name}_internal_distance.nii.gz")
    sitk.WriteImage(internal_dist_image, output_path_internal)
    print(f"å·²ä¿å­åé¨è·ç¦»å¾å°: {output_path_internal}")


    # --- çæè¾¹çå¿åºå¾ (Boundary Potential Map) ---
    # çµææ¥æºäºæ¨æä¾çæç® "Automatic kidney segmentation..."
    
    # a. æåè¾¹ç (åå§mask - èèåçmask)
    # sitk.BinaryErode éè¦ä¸ä¸ªåå¾åæ°ï¼è¿éæä»¬ä½¿ç¨ä¸ä¸ªåç´ åä½ççå½¢ç»æåç´ 
    # åå¾åä½æ¯ç©çåä½ï¼æä»¥æä»¬éè¦æ ¹æ®spacingæ¥å®
    # ä¸ºäºç®ååéç¨ï¼æä»¬ç¨ä¸ä¸ªåç´ çèè
    eroded_mask_image = sitk.BinaryErode(sitk.Cast(mask_image, sitk.sitkUInt8), [1, 1, 1])
    eroded_mask_array = sitk.GetArrayFromImage(eroded_mask_image)
    
    boundary_array = mask_array - eroded_mask_array
    
    # b. è®¡ç®å°è¾¹ççè·ç¦»
    # æä»¬å¸æè®¡ç®æ¯ä¸ªç¹å°è¾¹ççè·ç¦»ï¼æä»¥è¾¹çç¹åºè¯¥æ¯0ï¼å¶ä»ç¹é0
    # distance_transform_edt è®¡ç®çæ¯é0ç¹å°0ç¹çè·ç¦»
    # æä»¥æä»¬éè¦åè½¬è¾¹çå¾ (boundary_array == 0)
    boundary_dist_array = distance_transform_edt(boundary_array == 0, sampling=spacing)
    
    # c. ä½¿ç¨ææ°å½æ°è¿è¡å½ä¸åï¼çæå¿åº
    # D(p) = exp(-lambda * dist(p, boundary))
    potential_map_array = np.exp(-lambda_param * boundary_dist_array)
    
    # å°ç»æè½¬æ¢å SimpleITK å¾å
    potential_map_image = sitk.GetImageFromArray(potential_map_array)
    potential_map_image.CopyInformation(mask_image)

    # ä¿å­è¾¹çå¿åºå¾
    output_path_potential = os.path.join(output_dir, f"{base_name}_boundary_potential.nii.gz")
    sitk.WriteImage(potential_map_image, output_path_potential)
    print(f"å·²ä¿å­è¾¹çå¿åºå¾å°: {output_path_potential}")
    print("-" * 30)

if __name__ == '__main__':
    # --- ä½¿ç¨æ¹æ³ ---
    # 1. ç´æ¥å¨ä»£ç ä¸­ä¿®æ¹æä»¶è·¯å¾
    # input_mask_file = "data/parse2022/train/PA000005/label/PA000005.nii.gz"
    # output_directory = "pa005_label_distance_map.nii.gz"
    # generate_distance_maps(input_mask_file, output_directory)

    # 2. ä½¿ç¨å½ä»¤è¡åæ° (æ¨è)
    # å¨ç»ç«¯ä¸­è¿è¡:
    # python your_script_name.py -i /path/to/mask.nii.gz -o /path/to/output
    # python your_script_name.py -i /path/to/mask_folder -o /path/to/output_folder --batch
    
    parser = argparse.ArgumentParser(description="ä»NIfTIæ©æ¨¡çæè¡ç®¡è·ç¦»å¾")
    parser.add_argument('-i', '--input', type=str, required=True, help="è¾å¥çæ©æ¨¡æä»¶ææä»¶å¤¹è·¯å¾ã")
    parser.add_argument('-o', '--output', type=str, required=True, help="è¾åºç»æçä¿å­ç®å½ã")
    parser.add_argument('-l', '--lambda_param', type=float, default=0.5, help="è¾¹çå¿åºå¾çææ°è¡°åç³»æ°ï¼è®ºæè®¾ä¸º 0.5ï¼ã")
    parser.add_argument('--batch', action='store_true', help="å¦æè¾å¥æ¯æä»¶å¤¹ï¼åå¯ç¨æ­¤åæ°ä»¥æ¹éå¤çææ.nii.gzæä»¶ã")
    
    args = parser.parse_args()

    if args.batch:
        if not os.path.isdir(args.input):
            print(f"éè¯¯ï¼æ¹éå¤çæ¨¡å¼ä¸ï¼è¾å¥è·¯å¾å¿é¡»æ¯ä¸ä¸ªæä»¶å¤¹: {args.input}")
        else:
            nii_files = [f for f in os.listdir(args.input) if f.endswith(".nii.gz")]
            if not nii_files:
                print(f"éè¯¯ï¼å¨æä»¶å¤¹ {args.input} ä¸­æªæ¾å°.nii.gzæä»¶ã")
            else:
                print(f"æ¾å° {len(nii_files)} ä¸ª .nii.gz æä»¶ï¼å¼å§æ¹éå¤ç...")
                for file_name in nii_files:
                    file_path = os.path.join(args.input, file_name)
                    generate_distance_maps(file_path, args.output, args.lambda_param)
                print("æææä»¶å¤çå®æ¯ï¼")
    else:
        if not os.path.isfile(args.input):
            print(f"éè¯¯ï¼åä¸ªæä»¶å¤çæ¨¡å¼ä¸ï¼è¾å¥è·¯å¾å¿é¡»æ¯ä¸ä¸ªæä»¶: {args.input}")
        else:
            generate_distance_maps(args.input, args.output, args.lambda_param)
