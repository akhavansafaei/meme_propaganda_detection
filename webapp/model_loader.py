"""
Model Loader for AdaBoost Ensemble Propaganda Detection

این فایل مدل آموزش‌دیده رو بارگذاری می‌کنه و پیش‌بینی می‌کنه.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json
import os
from transformers import CLIPProcessor, CLIPModel, RobertaTokenizer, RobertaModel


class SimplifiedPropagandaDetector:
    """
    نسخه ساده‌شده برای استفاده در webapp

    TODO: جایگزین کن با کد واقعی از نوت‌بوک AdaBoost
    """

    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize the model

        Args:
            model_path: مسیر فایل مدل ذخیره شده (.pth)
            device: 'cpu' یا 'cuda'
        """
        self.device = device

        # لیست تکنیک‌ها (22 تکنیک)
        self.technique_names = [
            'Appeal to (Strong) Emotions',
            'Appeal to Authority',
            'Appeal to Fear/Prejudice',
            'Bandwagon',
            'Black-and-White Fallacy',
            'Causal Oversimplification',
            'Doubt',
            'Exaggeration/Minimisation',
            'Flag-waving',
            'Glittering Generalities',
            'Loaded Language',
            'Misrepresentation (Straw Man)',
            'Name Calling/Labeling',
            'Obfuscation/Vagueness',
            'Red Herring',
            'Reductio ad Hitlerum',
            'Repetition',
            'Slogans',
            'Smears',
            'Thought-Terminating Cliche',
            'Transfer',
            'Whataboutism'
        ]

        # اگه مدل واقعی دارید، اینجا بارگذاری کنید
        if model_path and os.path.exists(model_path):
            print(f"🔧 بارگذاری مدل از {model_path}...")
            self.load_real_model(model_path)
            self.use_real_model = True
        else:
            print("⚠️  مدل واقعی پیدا نشد، از شبیه‌سازی استفاده می‌شود")
            self.use_real_model = False
            self.model = None

    def load_real_model(self, model_path):
        """
        بارگذاری مدل واقعی

        TODO: این قسمت رو با کد واقعی از نوت‌بوک جایگزین کنید
        """
        try:
            # مثال: بارگذاری checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)

            # TODO: مدل رو بسازید و weights رو بارگذاری کنید
            # self.model = YourModelClass(...)
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.model.eval()

            print("✅ مدل بارگذاری شد")

        except Exception as e:
            print(f"❌ خطا در بارگذاری مدل: {e}")
            self.use_real_model = False
            self.model = None

    def preprocess_image(self, image_path):
        """
        پیش‌پردازش تصویر

        TODO: این رو با پیش‌پردازش دقیق نوت‌بوک جایگزین کنید
        """
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)

        return image_tensor

    def extract_text_features(self, text):
        """
        استخراج ویژگی از متن

        TODO: اگه از RoBERTa استفاده می‌کنید، اینجا اضافه کنید
        """
        # فعلاً خالی
        return None

    def predict(self, image_path, text="", ground_truth_labels=None):
        """
        پیش‌بینی تکنیک‌های پروپاگاندا

        Args:
            image_path: مسیر تصویر
            text: متن میم (اختیاری)
            ground_truth_labels: لیبل‌های واقعی (اختیاری)

        Returns:
            dict: نتایج پیش‌بینی
        """

        if self.use_real_model and self.model is not None:
            # استفاده از مدل واقعی
            predictions = self._predict_real(image_path, text)
        else:
            # شبیه‌سازی برای تست
            predictions = self._predict_simulated(image_path, text)

        # ساخت نتیجه نهایی
        result = {
            'predictions': predictions,
            'technique_names': self.technique_names,
            'ground_truth': ground_truth_labels,
            'using_real_model': self.use_real_model
        }

        return result

    def _predict_real(self, image_path, text):
        """
        پیش‌بینی با مدل واقعی

        TODO: کد پیش‌بینی واقعی از نوت‌بوک
        """
        with torch.no_grad():
            # TODO: استخراج ویژگی و پیش‌بینی
            # image_features = self.preprocess_image(image_path)
            # text_features = self.extract_text_features(text)
            # outputs = self.model(image_features, text_features)
            # predictions = torch.sigmoid(outputs).cpu().numpy()[0]

            # فعلاً یک مثال ساده
            predictions = {tech: 0.0 for tech in self.technique_names}

        return predictions

    def _predict_simulated(self, image_path, text):
        """
        شبیه‌سازی برای تست (تا مدل واقعی رو وصل نکردید)
        """
        import random

        # تحلیل ساده متن برای شبیه‌سازی واقعی‌تر
        text_lower = text.lower()

        predictions = {}

        for tech in self.technique_names:
            # امتیاز پایه تصادفی
            base_score = random.uniform(0.05, 0.25)

            # اگه متن داشتیم، یکم هوشمندانه‌تر شبیه‌سازی کنیم
            if text:
                # مثال‌های ساده
                if tech == 'Loaded Language' and any(w in text_lower for w in ['corrupt', 'evil', 'destroy']):
                    base_score += 0.4
                elif tech == 'Appeal to (Strong) Emotions' and any(w in text_lower for w in ['!', 'fear', 'angry']):
                    base_score += 0.3
                elif tech == 'Bandwagon' and any(w in text_lower for w in ['everyone', 'everybody', 'all']):
                    base_score += 0.35
                elif tech == 'Doubt' and any(w in text_lower for w in ['allegedly', 'supposedly', 'claim']):
                    base_score += 0.3

            predictions[tech] = min(base_score, 1.0)

        return predictions


def load_ground_truth_labels(image_name, dataset_path='../data/labels.json'):
    """
    بارگذاری لیبل‌های واقعی از دیتاست (اگه داری)

    Args:
        image_name: نام فایل تصویر
        dataset_path: مسیر فایل JSON که لیبل‌ها رو داره

    Returns:
        list: لیست تکنیک‌های واقعی یا None
    """
    if not os.path.exists(dataset_path):
        return None

    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            labels_data = json.load(f)

        # فرض: labels_data یک dict هست که key هاش نام فایل هستن
        return labels_data.get(image_name, None)

    except Exception as e:
        print(f"خطا در بارگذاری ground truth: {e}")
        return None


# تست
if __name__ == "__main__":
    print("=" * 60)
    print("تست Model Loader")
    print("=" * 60)

    # ساخت detector
    detector = SimplifiedPropagandaDetector()

    # تست با یک متن نمونه
    test_text = "These corrupt politicians are destroying everything!"

    # شبیه‌سازی پیش‌بینی
    result = detector.predict(
        image_path="test.jpg",  # فرضی
        text=test_text
    )

    print(f"\n📝 متن: {test_text}")
    print(f"\n🔮 پیش‌بینی‌ها (Top 5):")

    sorted_preds = sorted(
        result['predictions'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:5]

    for tech, score in sorted_preds:
        print(f"  {tech}: {score:.3f}")

    print("\n" + "=" * 60)
    print("✅ تست موفقیت‌آمیز بود!")
    print("⚠️  یادت باشه مدل واقعی رو وصل کنی!")
    print("=" * 60)
