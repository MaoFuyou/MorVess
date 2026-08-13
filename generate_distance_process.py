# -*- coding: utf-8 -*-
"""
æ­¤èæ¬ç¨äºä¸ºæ°æ®éæ¹éçæè¡ç®¡è·ç¦»å¾ï¼å¹¶å°å¶ä¿å­å¨ä¸ä¸ªç¬ç«çè¾åºç®å½ä¸­ï¼
åæ¶ä¿æåå§çç®å½å±æ¬¡ç»æã

V4çæ¬ï¼ç§»é¤äºææå½ä»¤è¡åæ°ï¼æ¹ç¨å¨èæ¬åç´æ¥æå®è·¯å¾ååæ°çæ¹å¼ï¼
ä»¥é¿åç¯å¢åå½ä»¤è¡ä½¿ç¨é®é¢ã

è¯·ç¡®ä¿å·²å®è£å¿è¦çåº:
pip install SimpleITK numpy scipy tqdm

ä¹æ¯æåæä»¶æ¨¡å¼ï¼ç´æ¥æç»æåå° -o æå®ç®å½ï¼ï¼
python generate_distance_process.py \
  -i /home/ET/bnwu/MA-SAM/PA000005.nii.gz -o ./distance_out
  
  
  
python generate_distance_process.py \
  -i /home/ET/bnwu/MA-SAM/data/AIIB23_Train_T1/gt/AIIB23_30.nii.gz -o ./aiib_distance_out

"""
#   
import os
import re
import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt

# ---------------- æ ¸å¿è®¡ç®ï¼çæè¾¹çå¿åºå¾ ---------------- #

def generate_potential_map(mask_path,
                           input_root_dir=None,
                           output_root_dir=None,
                           out_subdir_name="potential_map",
                           lambda_param=0.5,
                           save_all_steps=False):
    """
    è¯»ååä¸ª NIfTI æ©æ¨¡ï¼çæè·ç¦»å¾åè¾¹çå¿åºå¾ã
    æ°å¢åè½ï¼å½ save_all_steps=True æ¶ï¼ä¿å­ææä¸­é´æ­¥éª¤çç»æã

    åæ°ï¼
      mask_path        : æ©æ¨¡æä»¶è·¯å¾ï¼.nii æ .nii.gzï¼
      input_root_dir   : è¾å¥æ°æ®éæ ¹ç®å½ï¼æ¹éæ¶å¿å¡«ï¼
      output_root_dir  : è¾åºæ ¹ç®å½ï¼æ¹éæ¶å¿å¡«ï¼
      out_subdir_name  : æ¯ä¸ªæ ·æ¬ç®å½ä¸çè¾åºå­ç®å½åç§°
      lambda_param     : è¾¹çå¿åºå¾çææ°è¡°åç³»æ°
      save_all_steps   : (æ°å¢) æ¯å¦ä¿å­ææä¸­é´æ­¥éª¤çç»æ
    """
    if not os.path.exists(mask_path):
        print(f"[WARN] æä»¶ä¸å­å¨ï¼è·³è¿ï¼{mask_path}")
        return

    print(f"\n--- æ­£å¨å¤ç: {mask_path} ---")

    # ---- ç»ç»è¾åºè·¯å¾ ----
    base_name = os.path.basename(mask_path)
    base_noext = re.sub(r'\.nii(\.gz)?$', '', base_name, flags=re.I)
    base_noext = base_noext.split('_label')[0]

    if input_root_dir and output_root_dir:
        rel_label_path = os.path.relpath(mask_path, input_root_dir)
        relative_sample_dir = os.path.dirname(os.path.dirname(rel_label_path))
        out_dir = os.path.join(output_root_dir, relative_sample_dir, out_subdir_name)
    else:
        out_dir = output_root_dir if output_root_dir else os.path.dirname(mask_path)

    os.makedirs(out_dir, exist_ok=True)

    # ---- é¢å¤å·¥ä½ï¼è¯»åå¾åå¹¶åå¤ä¿å­å½æ° ----
    mask_img = sitk.ReadImage(mask_path, sitk.sitkFloat32)
    spacing = mask_img.GetSpacing()  # (sx, sy, sz)

    def save_intermediate_step(data_arr, step_name):
        """ä¸ä¸ªä¾¿æ·çå½æ°ï¼ç¨äºä¿å­ä¸­é´æ­¥éª¤ç .nii.gz æä»¶"""
        if save_all_steps:
            img = sitk.GetImageFromArray(data_arr.astype(np.float32))
            img.CopyInformation(mask_img)
            filepath = os.path.join(out_dir, f"{base_noext}_{step_name}.nii.gz")
            sitk.WriteImage(img, filepath)
            print(f"  [æ­¥éª¤] å·²ä¿å­: {os.path.basename(filepath)}")

    # ---- å¼å§è®¡ç®åä¿å­ä¸­é´æ­¥éª¤ ----

    # æ­¥éª¤ 1: äºå¼åæ©æ¨¡
    mask_arr = (sitk.GetArrayFromImage(mask_img) > 0).astype(np.uint8)
    save_intermediate_step(mask_arr, "step1_binary_mask")

    # æ­¥éª¤ 2: è®¡ç®åé¨è·ç¦»å¾ (Internal Distance Map)
    # è¿æ¯æ©æ¨¡åæ¯ä¸ªç¹å°æè¿è¾¹ççè·ç¦»ï¼åä½ï¼mmï¼
    internal_dist_arr = distance_transform_edt(mask_arr, sampling=spacing[::-1])
    save_intermediate_step(internal_dist_arr, "step2_internal_distance_mm")
    
    # æ­¥éª¤ 3: èèæ©æ¨¡
    # ä½¿ç¨ sitk.BinaryErode åå»ºä¸ä¸ªæ¯åå§æ©æ¨¡å°ä¸åçæ©æ¨¡
    eroded_mask_img = sitk.BinaryErode(sitk.Cast(mask_img > 0, sitk.sitkUInt8), [1, 1, 1])
    eroded_mask_arr = sitk.GetArrayFromImage(eroded_mask_img)
    save_intermediate_step(eroded_mask_arr, "step3_eroded_mask")

    # æ­¥éª¤ 4: æåè¾¹ç (Boundary Mask)
    # åå§æ©æ¨¡åå»èèåçæ©æ¨¡ï¼å¾å°çå°±æ¯è¾¹ç
    boundary_arr = mask_arr - eroded_mask_arr
    save_intermediate_step(boundary_arr, "step4_boundary_mask")

    # æ­¥éª¤ 5: è®¡ç®å°è¾¹ççè·ç¦»å¾ (Boundary Distance Map)
    # è¿æ¯å¾åä¸­ææç¹å°âè¾¹çâçæè¿è·ç¦»
    boundary_dist_arr = distance_transform_edt(boundary_arr == 0, sampling=spacing[::-1])
    save_intermediate_step(boundary_dist_arr, "step5_boundary_distance_mm")

    # æ­¥éª¤ 6: çæè¾¹çå¿åºå¾ (Boundary Potential Map)
    potential_map_arr = np.exp(-lambda_param * boundary_dist_arr)
    save_intermediate_step(potential_map_arr, "step6_boundary_potential_map")

    # ---- ä¿å­æç»ç»æï¼ä¸åå§èæ¬è¡ä¸ºä¿æä¸è´ï¼ ----
    # 1. åé¨è·ç¦»å¾
    internal_dist_img = sitk.GetImageFromArray(internal_dist_arr.astype(np.float32))
    internal_dist_img.CopyInformation(mask_img)
    out_internal_dist_path = os.path.join(out_dir, f"{base_noext}_internal_distance.nii.gz")
    sitk.WriteImage(internal_dist_img, out_internal_dist_path)
    print(f"==> æç»åé¨è·ç¦»å¾å·²ä¿å­: {out_internal_dist_path}")

    # 2. è¾¹çå¿åºå¾
    potential_map_img = sitk.GetImageFromArray(potential_map_arr.astype(np.float32))
    potential_map_img.CopyInformation(mask_img)
    out_potential_map_path = os.path.join(out_dir, f"{base_noext}_boundary_potential.nii.gz")
    sitk.WriteImage(potential_map_img, out_potential_map_path)
    print(f"==> æç»è¾¹çå¿åºå¾å·²ä¿å­: {out_potential_map_path}")


# ---------------- å¥å£ï¼ä¸ä¹åèæ¬ä¸è´çâéå½æ¹éâ ---------------- #

def main():
    parser = argparse.ArgumentParser(description="ä» NIfTI æ©æ¨¡çæè·ç¦»å¾åè¾¹çå¿åºå¾ï¼å¹¶å¯éæ©ä¿å­ææä¸­é´æ­¥éª¤ã")
    parser.add_argument('-i', '--input',  type=str, required=True,
                        help="è¾å¥è·¯å¾ï¼åæä»¶ æ æ°æ®éæ ¹ç®å½ï¼å«è¥å¹² <sample>/label/*.nii.gzï¼")
    parser.add_argument('-o', '--output', type=str, required=True,
                        help="è¾åºæ ¹ç®å½ï¼æ¹éæ¶ä¼æåå±çº§åå¥ï¼åæä»¶æ¶ç´æ¥åå°è¯¥ç®å½ï¼")
    parser.add_argument('--batch', action='store_true',
                        help="æ¹éæ¨¡å¼ï¼ä»è¾å¥æ ¹ç®å½éå½æ¥æ¾ label/*.nii.gz å¹¶æ¹éå¤ç")
    parser.add_argument('--out_subdir', type=str, default='potential_map',
                        help="æ¯ä¸ªæ ·æ¬ç®å½ä¸çè¾åºå­ç®å½åï¼é»è®¤ potential_mapï¼")
    parser.add_argument('--lambda', type=float, default=0.5, dest='lambda_param',
                        help="è¾¹çå¿åºå¾çææ°è¡°åç³»æ°ï¼è®ºæè®¾ä¸º 0.5ï¼")
    parser.add_argument('--save_all_steps', action='store_true',default=True,
                        help="ï¼æ°å¢ï¼ä¿å­ææè®¡ç®çä¸­é´æ­¥éª¤æä»¶ï¼ç¨äºå±ç¤ºåè°è¯")
    args = parser.parse_args()

    if args.batch:
        if not os.path.isdir(args.input):
            raise ValueError(f"æ¹éæ¨¡å¼ä¸ï¼--input å¿é¡»æ¯ç®å½ï¼{args.input}")
        files = []
        for root, dirs, fs in os.walk(args.input):
            if os.path.basename(root) == 'label':
                for f in fs:
                    if f.endswith('.nii') or f.endswith('.nii.gz'):
                        files.append(os.path.join(root, f))
        if not files:
            raise RuntimeError(f"æªå¨ {args.input} ä¸æ¾å°ä»»ä½ label/*.nii[.gz] æä»¶")

        print(f"å±æ¾å° {len(files)} ä¸ª label æ©æ¨¡ï¼å¼å§å¤çâ¦â¦")
        for p in files:
            generate_potential_map(
                p,
                input_root_dir=args.input,
                output_root_dir=args.output,
                out_subdir_name=args.out_subdir,
                lambda_param=args.lambda_param,
                save_all_steps=args.save_all_steps
            )
        print("\nå¨é¨å¤çå®æã")
    else:
        # åæä»¶æ¨¡å¼
        if not os.path.isfile(args.input):
            raise ValueError(f"åæä»¶æ¨¡å¼ä¸ï¼--input å¿é¡»æ¯æä»¶ï¼{args.input}")
        os.makedirs(args.output, exist_ok=True)
        generate_potential_map(
            args.input,
            input_root_dir=None,
            output_root_dir=args.output,
            out_subdir_name=None,
            lambda_param=args.lambda_param,
            save_all_steps=args.save_all_steps
        )
        print("\nå¤çå®æã")

if __name__ == '__main__':
    main()
