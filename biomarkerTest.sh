data_dir="/mnt/data_2/gjf/tiles_detected_cascadercnn_299/TCGA_RoI_grid"
biomarker_txt="./subtypeDetection/LUAD_TMB_20210524.txt"
# input_train="./subtypeDetection/inference/TCGA_train_grid.csv"
input_test="./subtypeDetection/inference/TCGA_test_grid.csv"

confidence_ckp="./ood/round_1/epoch_35.pth"
biomarker='TMB'
confidence_threshold_test=55 # which is expressed in percentages.
biomarker_threshold=206
python ./ood/ood_evaluate.py \
    --checkpoint $confidence_ckp \
    --data_dir $data_dir \
    --input $input_test \
    --biomarker_txt $biomarker_txt \
    --biomarker $biomarker \
    --confidence_threshold $confidence_threshold_test \
    --biomarker_threshold $biomarker_txt


# confidence_input_train="./ood/TCGA_train_grid_conf.csv"
# confidence_output_train="./ood/TCGA_train_grid_0.55.csv"
# confidence_threshold_train=55
confidence_input_test="./ood/TCGA_test_grid_conf.csv"
confidence_output_test="./ood/TCGA_test_grid_0.55.csv"
confidence_threshold_test=55
python ./TMB/ood_inference.py \
    --checkpoint $confidence_ckp \
    --data_dir $data_dir \
    --input $input_test \
    --output $confidence_input_test \
    --biomarker_txt $biomarker_txt \
    --biomarker $biomarker \
    --biomarker_threshold $biomarker_txt

python ./TMB/ood_screen.py \
    --input $confidence_input_test \
    --output $confidence_output_test \
    --threshold $confidence_threshold_test
# python ./TMB/ood_inference.py \
#     --checkpoint $confidence_ckp \
#     --data_dir $data_dir \
#     --input $input_train $input_test \
#     --output $confidence_input_train $confidence_input_test \
#     --input $confidence_input_train $confidence_input_test \
#     --biomarker_txt $biomarker_txt
# python ./TMB/ood_screen.py \
#     --output $confidence_output_train $confidence_output_test \
#     --threshold $confidence_threshold_train $confidence_threshold_test


cls_ckp="./biomarkerCls/threshold_206/Xception_0.38.pth"
python ./TMB/cls_evaluate.py \
    --checkpoint $cls_ckp \
    --data_dir $data_dir \
    --input $confidence_output_test \
    --biomarker_txt $biomarker_txt \
    --biomarker $biomarker \
    --biomarker_threshold $biomarker_txt