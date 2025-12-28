import os
import re
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# ==========================================
# 1. 文本清洗与增强 (保留研究方案特性)
# ==========================================
def clean_medical_text(text):
    if not text or pd.isna(text):
        return ''
    text = str(text).lower()
    # 彻底清洗医学报告文本 (参考 MVCM 逻辑并保留研究方案的噪音剔除)
    text = re.sub(r'\bs\d{5,}\b', '', text) # Study ID
    text = re.sub(r'dicom[_\s]?id\s*[:=]?\s*\w+', '', text)
    text = re.sub(r'study\s+id\s*[:=]?\s*\w+', '', text)
    # 删除日期
    text = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}[,\s]+\d{4}\b', '', text)
    text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def augment_text_for_retrieval(text, drop_sentence_prob=0.1):
    """
    针对检索任务的文本增强 (参考研究方案: 随机丢弃句子)
    """
    if not text or len(text.strip()) == 0:
        return text
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    
    if len(sentences) <= 1:
        return text
        
    # 随机丢弃一些句子（但至少保留一半）
    if random.random() < 0.5:
        num_to_drop = min(2, max(1, len(sentences) // 4))
        indices_to_keep = sorted(random.sample(range(len(sentences)), len(sentences) - num_to_drop))
        sentences = [sentences[i] for i in indices_to_keep]
        
    augmented_text = '. '.join(sentences)
    if augmented_text and not augmented_text.endswith('.'):
        augmented_text += '.'
    return augmented_text

# ==========================================
# 2. 图像增强 (参考 MVCM 增强配置，加强随机性)
# ==========================================
def get_transforms(img_size=224, is_train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        # 参考 MVCM 的强增强配置，确保多视角时有明显差异
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), interpolation=InterpolationMode.BICUBIC),  # 更大的裁剪范围
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),  # 提高概率
            transforms.RandomGrayscale(p=0.2),  # 提高概率
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),  # 添加旋转
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize
        ])

# ==========================================
# 3. 核心 Dataset 类 (参考 MVCM CustomMIMICDataset 结构)
# ==========================================
class OTLGADataset(Dataset):
    def __init__(self, data_root, csv_path, split='train', img_size=224, is_multiview=True, max_txt_len=200):
        self.data_root = data_root
        self.split = split
        self.is_multiview = is_multiview
        self.max_txt_len = max_txt_len
        self.transform = get_transforms(img_size, is_train=(split=='train'))
        
        # 1. 加载并根据 split 列划分 (参考 MVCM 逻辑)
        df_all = pd.read_csv(csv_path)
        self.df = df_all[df_all['split'] == split].reset_index(drop=True)
        
        # 2. 构建 Study 到 Images 的映射 (实现多视角)
        # 假设文件名中可以提取 Study ID
        self.df['study_id'] = self.df['filename'].apply(lambda x: x.split('.')[0])
        
        # 记录所有唯一的 Study ID
        self.unique_studies = self.df['study_id'].unique()
        
        # 构建映射字典
        self.study_to_images = self.df.groupby('study_id')['filename'].apply(list).to_dict()
        self.study_to_data = self.df.groupby('study_id').first().to_dict('index')
        
        self.label_cols = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 
                          'Enlarged Cardiomediastinum', 'Lung Lesion', 'Lung Opacity', 
                          'Normal', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax']
        
        print(f"Dataset Split: {split}, Studies: {len(self.unique_studies)}, Samples: {len(self.df)}")

    def __len__(self):
        return len(self.unique_studies)

    def _find_image_path(self, filename, source='dataset1'):
        """
        参考 MVCM 的路径查找逻辑
        """
        # 尝试在合并后的文件夹查找
        for sub in ['train', 'valid', 'test']:
            path = os.path.join(self.data_root, sub, filename)
            if os.path.exists(path):
                return path
        
        # 尝试在 dataset2 的 resized_images 查找
        if source == 'dataset2':
            path = os.path.join(self.data_root, 'resized_images', '512', filename)
            if os.path.exists(path):
                return path
                
        # 兜底
        return os.path.join(self.data_root, filename)

    def __getitem__(self, index):
        study_id = self.unique_studies[index]
        study_info = self.study_to_data[study_id]
        img_filenames = self.study_to_images[study_id]
        
        # --- 多视角支持 (研究方案核心) ---
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
            image_tensor = torch.stack(imgs) # [2, 3, H, W]
        else:
            img_path = self._find_image_path(img_filenames[0], source=study_info.get('source', 'dataset1'))
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                img = Image.new('RGB', (224, 224), color='black')
            image_tensor = self.transform(img) # [3, H, W]

        # --- 文本处理 ---
        raw_text = study_info.get('org_caption', study_info.get('label', ''))
        if pd.isna(raw_text):
            raw_text = str(study_info.get('label', ''))
        
        text = clean_medical_text(raw_text)
        if self.split == 'train':
            text = augment_text_for_retrieval(text)
        
        # 截断
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
