import os
import re
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

def clean_medical_text(text):
    if not text or pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'\bs\d{5,}\b', '', text)
    text = re.sub(r'dicom[_\s]?id\s*[:=]?\s*\w+', '', text)
    text = re.sub(r'study\s+id\s*[:=]?\s*\w+', '', text)
    text = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}[,\s]+\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def augment_text_for_retrieval(text, drop_sentence_prob=0.1):
    if not text or len(text.strip()) == 0:
        return text
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) <= 1:
        return text
        
    if random.random() < 0.5:
        num_to_drop = min(2, max(1, len(sentences) // 4))
        indices_to_keep = sorted(random.sample(range(len(sentences)), len(sentences) - num_to_drop))
        sentences = [sentences[i] for i in indices_to_keep]
        
    augmented_text = '. '.join(sentences)
    if augmented_text and not augmented_text.endswith('.'):
        augmented_text += '.'
    return augmented_text

def get_transforms(img_size=224, is_train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])

class OTLGADataset(Dataset):
    def __init__(self, data_root, csv_path, split='train', img_size=224, is_multiview=True, max_txt_len=200):
        self.data_root = data_root
        self.split = split
        self.is_multiview = is_multiview
        self.max_txt_len = max_txt_len
        self.transform = get_transforms(img_size, is_train=(split=='train'))
        
        df_all = pd.read_csv(csv_path)
        self.df = df_all[df_all['split'] == split].reset_index(drop=True)
        
        self.df['study_id'] = self.df['filename'].apply(lambda x: x.split('.')[0])
        
        self.unique_studies = self.df['study_id'].unique()
        
        self.study_to_images = self.df.groupby('study_id')['filename'].apply(list).to_dict()
        self.study_to_data = self.df.groupby('study_id').first().to_dict('index')
        
        self.label_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                          'Enlarged Cardiomediastinum', 'Lung Lesion', 'Lung Opacity', 
                          'Normal', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']
        
        print(f"Dataset Split: {split}, Studies: {len(self.unique_studies)}, Samples: {len(self.df)}")

    def __len__(self):
        return len(self.unique_studies)

    def _find_image_path(self, filename, source='dataset1'):
        for sub in ['train', 'valid', 'test']:
            path = os.path.join(self.data_root, sub, filename)
            if os.path.exists(path):
                return path
        
        if source == 'dataset2':
            path = os.path.join(self.data_root, 'resized_images', '512', filename)
            if os.path.exists(path):
                return path
                
        return os.path.join(self.data_root, filename)

    def __getitem__(self, index):
        study_id = self.unique_studies[index]
        study_info = self.study_to_data[study_id]
        img_filenames = self.study_to_images[study_id]
        
        if self.is_multiview:
            if len(img_filenames) >= 2:
                selected = random.sample(img_filenames, 2)
            else:
                selected = [img_filenames[0], img_filenames[0]]
            
            imgs = []
            for fname in selected:
                img_path = self._find_image_path(fname, source=study_info.get('source', 'dataset1'))
                try:
                    img = Image.open(img_path).convert('RGB')
                except:
                    img = Image.new('RGB', (224, 224), color='black')
                imgs.append(self.transform(img))
            image_tensor = torch.stack(imgs)
        else:
            img_path = self._find_image_path(img_filenames[0], source=study_info.get('source', 'dataset1'))
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                img = Image.new('RGB', (224, 224), color='black')
            image_tensor = self.transform(img)

        raw_text = study_info.get('org_caption', study_info.get('label', ''))
        if pd.isna(raw_text):
            raw_text = str(study_info.get('label', ''))
        
        text = clean_medical_text(raw_text)
        if self.split == 'train':
            text = augment_text_for_retrieval(text)
        
        if len(text.split()) > self.max_txt_len:
            text = " ".join(text.split()[:self.max_txt_len])
            
        soft_labels = []
        for col in self.label_cols:
            val = study_info.get(col, 0.0)
            soft_labels.append(float(val))
        labels = torch.tensor(soft_labels, dtype=torch.float32)
            
        return {
            "image": image_tensor,
            "text": text,
            "labels": labels,
            "study_id": study_id
        }
