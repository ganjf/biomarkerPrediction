import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset

class tcgaSubtypePseudo(Dataset):
    def __init__(self, wsi_dir, data_csv, biomarker_txt, transform=None, biomarker_threshold=300, sampled=False, biomarker='TMB'):
        self.wsi_dir = wsi_dir
        self.data_csv = data_csv
        self.biomarker_txt = biomarker_txt
        self.transform = transform
        self.biomarker_threshold = biomarker_threshold
        self.sampled = sampled
        self.data_df = pd.read_csv(self.data_csv)
        self.biomarker = biomarker
        self.data_biomarker = pd.read_csv(biomarker_txt, sep='\t')
        self.data_biomarker = self.data_biomarker.fillna(0)
        
        if self.sampled:
            self.classes_for_all_imgs = []
            for index in range(len(self.data_df)):
                biomarker = self.data_df.iloc[index][biomarker]
                self.classes_for_all_imgs.append(biomarker)

    def __getitem__(self, index):
        annotation = self.data_df.iloc[index]
        name = annotation['name']
        biomarker = float(self.data_biomarker[self.data_biomarker['patient'] == name[0:12]][self.biomarker])
        biomarker = 1 if biomarker >= self.biomarker_threshold else 0
        
        name = os.path.join(self.wsi_dir, name)
        img = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        # cv2.imwrite('ori.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img = self.transform(image=img)['image']
        # cv2.imwrite('heJitter.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        biomarker = torch.tensor(biomarker)
        return img, biomarker

    def __len__(self):
        return len(self.data_df)
    
    def setData(self, dataCSV):
        self.data_df = pd.read_csv(dataCSV)
    
    def get_classes_for_all_imgs(self):
        return self.classes_for_all_imgs