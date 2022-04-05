## Prepration
- Install MMDetection.
    - https://github.com/open-mmlab/mmdetection
- Biomarker documentation
    - The format is similar to [TMB_CPTAC_LUAD.txt](https://drive.google.com/file/d/1uOb6-L5OaHRh8d3nFcnnJdJLqGuCTa6T/view?usp=sharing)


## Subtype Lesion Detection.
1. Code Modification
    - ./subtypeDetection/inference/processSlideWindow.py
        - line 99-112: Set WSIs which need to be processed.
    - ./subtypeDetection/inference/makeDataset.py
        - line 45-46 and line 51-52: Set the releationship between WSI naming method and patient id in Biomarker documentation.

2. Configuration of subtypeTest.sh
    - --wsi_dir: Path to store WSIs.
    - --patch_dir: Path to store patches obtained by sliding window approach.
    - --biomarker_txt: Path of biomarker documentation.
    - --inference_coco: COCO format File to Detector, recording information of patches (`$patch_dir`) obtained by sliding window approach.
        
        > cascade_rcnn.py \
        > line 268: set 'ann_file' = $patch_dir \
        > line 269: set 'img_prefix' = $inference_coco
    - --detection_ckp: Checkpoint of detection model. [download](https://drive.google.com/file/d/1ZC_qpU3P5nVf1M-Pfy5mRpuDPHZLszOQ/view?usp=sharing)
    - --result_pkl: Path to store detection results.
    - --roi_json: Path to store the result after mapping detetcion results to the raw WSI.
    - --roi_dir: Path to store patches obtained according to the detection results.
    - --data_test: summary of information about the images under the path of `$roi_dir` which need to be test.
3. Run `bash subtypeTest.sh`

## OOD and Classification
1. Code Modification
    - ./TMB/ood_evaluate.py
        - line 50-51: Set the releationship between WSI naming method and patient id in biomarker documentation.
    - ./TMB/ood_screen.py
        - line 9-10, line 14-15, line 21-22 and line 36-37: Set the releationship between WSI naming method and patient id in biomarker documentation.
    - ./TMB/cls_evaluate.py
        - line 49-50: Set the releationship between WSI naming method and patient id in biomarker documentation.

2. Configuration of biomarkerTest.sh
    - --data_dir: Path to store patches obtained according to the detection results.(as same as `$roi_dir` in subtypeTest.sh)
    - --biomarker_txt: Path of biomarker documentation.
    - --input_test: summary of information about the images which need to be test. (as same as `$data_test` in subtypeTest.sh)
    - --confidence_ckp: Checkpoint of OOD model. [download](https://drive.google.com/file/d/15nG2sl0cp6kV-Fl96uzHqOHa4jvnP3fy/view?usp=sharing)
    - --biomarker: choices of ['TMB', 'CD274', 'CD8A']
    - --confidence_threshold: Threshold to conduct ood detection, default: 55(%).
    - --biomarker_threhold: Threshold to determine the status of biomarker positive.
    - --confidence_input_test: Path to store predictions and confidence estimation of OOD model.
    - --confidence_output_test: Path to store those patches picked by OOD model.
    - --cls_ckp: Checkpoint of classification model. [download](https://drive.google.com/file/d/1dNtAN8FElNCxrxJ-J_ZmwkH8wzz7tESF/view?usp=sharing)
3. Run `bash biomarkerTest.sh >> test.log`


