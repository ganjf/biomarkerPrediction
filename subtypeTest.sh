# @author Jiefeng Gan

wsi_dir="/mnt/data/hzt/LUNG/full_LUAD"
patch_dir="/mnt/data_2/gjf/detection_data/luad_patch_4000_1000_tcga_slidewindow"
biomarker_txt="./subtypeDetection/LUAD_TMB_20210524.txt"
# pid_dis="./subtypeDetection/PID_TCGA.csv"

python ./subtypeDetection/inference/processSlideWindow.py \
    --wsi_dir $wsi_dir \
    --patch_dir $patch_dir \
    --biomarker_txt $biomarker_txt


inference_coco="./subtypeDetection/inference/TCGA_inference_coco.json"
python ./subtypeDetection/inference/data2mmdet.py \
    --wsi_dir $wsi_dir \
    --patch_dir $patch_dir \
    --inference_coco $inference_coco


detection_ckp='./subtypeDetection/cascade_rcnn.pth'
result_pkl="./subtypeDetection/inference/TCGA_out.pkl"
python tools/test.py ./subtypeDetection/train/cascade_rcnn.py \
    $detection_ckp \
    --eval bbox \
    --out $result_pkl


roi_json="./subtypeDetection/inference/TCGA_out.json"
roi_dir="/mnt/data_2/gjf/detection_data/TCGA_RoI_grid"
python ./subtypeDetection/inference/processRoI.py \
    --wsi_dir $wsi_dir \
    --inference_coco $inference_coco \
    --result_pkl $result_pkl \
    --roi_json $roi_json \
    --roi_dir $roi_dir


# data_train="./subtypeDetection/TCGA_train_grid.csv"
data_test="./subtypeDetection/TCGA_test_grid.csv"
python ./subtypeDetection/inference/makeDataset.py \
    --roi_dir $roi_dir \
    --biomarker_txt $biomarker_txt \
    --data_csv $data_test
    # --data_csv $data_train $data_test \
    # --pid_dis $pid_dis
