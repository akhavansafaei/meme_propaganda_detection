# 🔧 راهنمای اتصال مدل AdaBoost به وب‌سایت

این راهنما بهت کمک می‌کنه که مدل AdaBoost خودت رو به وب‌سایت وصل کنی.

## 📋 قدم به قدم

### قدم 1: مدل رو ذخیره کن

اول باید مدل آموزش‌دیده خودت رو از نوت‌بوک ذخیره کنی.

در نوت‌بوک `ali_write_adaboost_ensemble.ipynb` این کد رو اضافه کن:

```python
import torch
import pickle

# بعد از آموزش مدل، ذخیره‌اش کن:

# اگر مدل PyTorch هست:
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model_config,
    # هر چیز دیگه‌ای که لازمه
}, 'saved_models/adaboost_model.pth')

# اگر مدل sklearn هست:
with open('saved_models/adaboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ مدل ذخیره شد!")
```

### قدم 2: کد پیش‌پردازش رو کپی کن

کد استخراج ویژگی از نوت‌بوک رو کپی کن و در یک فایل جدا بذار:

```bash
# ساخت فایل جدید
touch webapp/feature_extraction.py
```

محتوای این فایل باید شامل تابع‌های پیش‌پردازش باشه:

```python
# webapp/feature_extraction.py

import torch
from PIL import Image
from torchvision import transforms
# هر چیز دیگه‌ای که توی نوت‌بوک استفاده کردی

def extract_features(image_path, text):
    """
    استخراج ویژگی از تصویر و متن
    این دقیقاً همون کدی هست که توی نوت‌بوک داری
    """
    # TODO: کپی کن از نوت‌بوک

    # مثال:
    # 1. بارگذاری تصویر
    image = Image.open(image_path).convert('RGB')

    # 2. پیش‌پردازش
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image)

    # 3. استخراج ویژگی متن
    # text_features = ...

    # 4. ترکیب ویژگی‌ها
    # combined_features = ...

    return combined_features
```

### قدم 3: کد بارگذاری مدل رو اضافه کن

فایل `webapp/model_loader.py` رو بساز:

```python
# webapp/model_loader.py

import torch
import pickle
from feature_extraction import extract_features

class PropagandaDetector:
    def __init__(self, model_path='../saved_models/adaboost_model.pth'):
        """بارگذاری مدل"""
        print(f"🔧 بارگذاری مدل از {model_path}...")

        # برای مدل PyTorch:
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = YourModelClass(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # یا برای sklearn:
        # with open(model_path, 'rb') as f:
        #     self.model = pickle.load(f)

        print("✅ مدل بارگذاری شد!")

        # لیست تکنیک‌ها
        self.technique_names = [
            'Appeal to (Strong) Emotions',
            'Appeal to Authority',
            # ... بقیه 22 تکنیک
        ]

    def predict(self, image_path, text=""):
        """تشخیص تکنیک‌های پروپاگاندا"""

        # 1. استخراج ویژگی
        features = extract_features(image_path, text)

        # 2. پیش‌بینی
        with torch.no_grad():
            predictions = self.model(features)

        # 3. تبدیل به دیکشنری
        results = {}
        for i, tech_name in enumerate(self.technique_names):
            results[tech_name] = float(predictions[i])

        return results

# تست
if __name__ == "__main__":
    detector = PropagandaDetector()
    results = detector.predict('test_image.jpg')
    print(results)
```

### قدم 4: اتصال به Flask

حالا فایل `app.py` رو ویرایش کن:

```python
# در ابتدای فایل app.py، اضافه کن:
from model_loader import PropagandaDetector

# بعد از kg_detector، اضافه کن:
print("🔧 Loading your AdaBoost model...")
model = PropagandaDetector('../saved_models/adaboost_model.pth')
print("✅ AdaBoost model loaded!")

# حالا تابع get_model_predictions رو تغییر بده:
def get_model_predictions(image_path, text):
    """Get predictions from your AdaBoost model"""
    return model.predict(image_path, text)
```

## 🚀 راه‌اندازی سایت

### نصب بسته‌ها

```bash
cd webapp
pip install -r requirements.txt
```

### اجرای سرور

```bash
python app.py
```

مرورگر رو باز کن و برو به:
```
http://localhost:5000
```

## 🔍 تست سریع

یک میم تست آپلود کن و ببین نتیجه رو درست نشون میده یا نه.

## 🐛 عیب‌یابی

### خطا: مدل پیدا نشد

```bash
# اطمینان حاصل کن که مسیر درسته
ls -la saved_models/
```

### خطا: ماژول پیدا نشد

```bash
# نصب بسته‌های گم‌شده
pip install [package-name]
```

### خطا: شکل تنسور اشتباه است

بررسی کن که:
1. پیش‌پردازش تصویر درست باشه (همون که توی نوت‌بوک هست)
2. شکل ورودی مدل با آموزش یکسان باشه

## 📝 نکات مهم

1. **پیش‌پردازش**: حتماً همون پیش‌پردازشی که موقع آموزش استفاده کردی رو اینجا هم استفاده کن
2. **نرمالیزاسیون**: اگر دیتا رو نرمال کردی، اینجا هم باید نرمال کنی
3. **Device**: مدل رو روی CPU بارگذاری کن (برای سرور معمولی)
4. **Cache**: می‌تونی مدل رو کش کنی تا هر بار بارگذاری نشه

## 🎯 مثال کامل

اگه توی نوت‌بوک این کد رو داری:

```python
# Training code
model = AdaBoostEnsemble(...)
model.train(train_loader)

# Prediction
output = model(image_features)
```

پس توی webapp باید:

```python
# model_loader.py
class PropagandaDetector:
    def __init__(self):
        self.model = AdaBoostEnsemble(...)
        self.model.load_state_dict(torch.load('model.pth'))

    def predict(self, image_path):
        image_features = extract_features(image_path)
        return self.model(image_features)
```

## ✅ چک‌لیست

- [ ] مدل رو ذخیره کردم (`.pth` یا `.pkl`)
- [ ] کد پیش‌پردازش رو کپی کردم
- [ ] `model_loader.py` رو ساختم
- [ ] `app.py` رو ویرایش کردم
- [ ] بسته‌ها رو نصب کردم
- [ ] سرور رو اجرا کردم
- [ ] با یک میم تست کردم

## 🆘 کمک

اگه گیر کردی، این فایل‌ها رو چک کن:
1. نوت‌بوک اصلیت: `ali_write_adaboost_ensemble.ipynb`
2. خطاهای Flask: در ترمینال نمایش داده میشه
3. خطاهای مرورگر: F12 > Console

موفق باشی! 🚀
