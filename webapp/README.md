# 🌐 وب‌سایت تشخیص پروپاگاندا در میم‌ها

یک وب‌اپلیکیشن ساده و قشنگ برای تشخیص تکنیک‌های پروپاگاندا در میم‌ها با استفاده از هوش مصنوعی.

## 🎨 امکانات

- ✅ آپلود میم (Drag & Drop یا Click)
- ✅ تشخیص 22 تکنیک پروپاگاندا
- ✅ نمایش نتایج با نمودار و امتیاز
- ✅ استفاده از مدل AdaBoost Ensemble شما
- ✅ بهبود نتایج با Knowledge Graph
- ✅ رابط کاربری فارسی/انگلیسی
- ✅ طراحی زیبا و Responsive

## 🚀 راه‌اندازی سریع

### روش 1: استفاده از اسکریپت (راحت‌ترین)

**Linux/Mac:**
```bash
cd webapp
./START_WEBAPP.sh
```

**Windows:**
```bash
cd webapp
START_WEBAPP.bat
```

### روش 2: دستی

```bash
# 1. رفتن به پوشه webapp
cd webapp

# 2. نصب بسته‌ها
pip install -r requirements.txt

# 3. اجرای سرور
python app.py
```

سپس مرورگر رو باز کن و برو به:
```
http://localhost:5000
```

## 📁 ساختار فایل‌ها

```
webapp/
├── app.py                      # سرور Flask (backend)
├── templates/
│   └── index.html             # رابط کاربری (frontend)
├── uploads/                    # پوشه موقت برای آپلود (ساخته میشه)
├── requirements.txt           # بسته‌های مورد نیاز
├── MODEL_INTEGRATION_GUIDE.md # راهنمای اتصال مدل
├── START_WEBAPP.sh            # اسکریپت راه‌اندازی (Linux/Mac)
├── START_WEBAPP.bat           # اسکریپت راه‌اندازی (Windows)
└── README.md                  # این فایل
```

## ⚙️ اتصال مدل AdaBoost خودت

**مهم:** هم‌اکنون سایت با پیش‌بینی‌های تصادفی کار می‌کنه. برای استفاده از مدل واقعی:

### قدم 1: ذخیره مدل

در نوت‌بوک `ali_write_adaboost_ensemble.ipynb`:

```python
import torch

# بعد از آموزش مدل:
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config
}, '../saved_models/adaboost_model.pth')
```

### قدم 2: ساخت Model Loader

فایل `model_loader.py` بساز:

```python
import torch

class PropagandaDetector:
    def __init__(self, model_path):
        # بارگذاری مدل
        checkpoint = torch.load(model_path)
        self.model = YourModelClass(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, image_path, text=""):
        # استخراج ویژگی و پیش‌بینی
        features = self.extract_features(image_path, text)
        with torch.no_grad():
            predictions = self.model(features)
        return predictions
```

### قدم 3: ویرایش app.py

در `app.py` پیدا کن:

```python
# TODO: Load your AdaBoost model here
model = None  # Placeholder
```

و جایگزین کن با:

```python
from model_loader import PropagandaDetector
model = PropagandaDetector('../saved_models/adaboost_model.pth')
```

**راهنمای کامل:** `MODEL_INTEGRATION_GUIDE.md`

## 🔧 تنظیمات

### تغییر پورت

در `app.py` آخرین خط:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # پورت رو تغییر بده
```

### حداکثر سایز فایل

در `app.py`:

```python
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
```

### فعال کردن LLM (اختیاری)

برای دقت بیشتر، می‌تونی LLM verification رو فعال کنی:

```python
# در app.py
kg_detector = ThreeStreamArchitecture(
    techniques_path="../propaganda_techniques.json",
    use_llm=True,
    llm_client=openai.OpenAI(api_key="your-key")
)
```

## 📊 چطور کار می‌کنه؟

```
1. کاربر میم رو آپلود می‌کنه
          ↓
2. سرور Flask تصویر رو دریافت می‌کنه
          ↓
3. مدل AdaBoost تحلیل می‌کنه → پیش‌بینی‌های اولیه
          ↓
4. Knowledge Graph بهبود می‌بخشه → حذف False Positives
          ↓
5. نتایج نهایی به کاربر نمایش داده میشه
```

## 🎨 تغییر ظاهر

فایل `templates/index.html` رو ویرایش کن:

### تغییر رنگ‌ها

```css
background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
/* رنگ دلخواه خودت رو بذار */
```

### تغییر فونت

```css
font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
/* فونت فارسی دلخواه: Vazir, Shabnam, etc. */
```

## 📱 Responsive Design

سایت روی همه دستگاه‌ها کار می‌کنه:
- 💻 Desktop
- 📱 Mobile
- 🖥️ Tablet

## 🐛 عیب‌یابی

### خطا: Port already in use

```bash
# پورت دیگه‌ای استفاده کن
python app.py  # و در app.py پورت رو عوض کن
```

### خطا: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### خطا: propaganda_techniques.json not found

```bash
# مطمئن شو از پوشه webapp اجرا می‌کنی
cd webapp
python app.py
```

### میم آپلود نمیشه

- سایز فایل رو چک کن (حداکثر 16MB)
- فرمت فایل باید JPG, PNG یا GIF باشه
- Console مرورگر رو چک کن (F12)

## 📈 بهبود عملکرد

### اضافه کردن OCR

```bash
pip install pytesseract easyocr
```

```python
# در app.py
import pytesseract

def extract_text_from_image(image_path):
    return pytesseract.image_to_string(image_path, lang='eng+fas')
```

### اضافه کردن Image Features

```python
# استفاده از CLIP برای استخراج ویژگی
from transformers import CLIPProcessor, CLIPModel

def extract_image_features(image_path):
    # استخراج entities, symbols, etc.
    return features
```

### کش کردن مدل

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_prediction(image_hash):
    # کش کردن نتایج برای سرعت بیشتر
    pass
```

## 🚀 استقرار Production

### استفاده از Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### استفاده از Docker

```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### Deploy روی Heroku/Railway

1. اضافه کردن `Procfile`:
```
web: gunicorn app:app
```

2. Push کردن:
```bash
git push heroku main
```

## 🔒 امنیت

برای Production:

1. **Debug mode رو خاموش کن:**
```python
app.run(debug=False)
```

2. **محدودیت Rate Limit:**
```python
from flask_limiter import Limiter
limiter = Limiter(app, default_limits=["200 per day", "50 per hour"])
```

3. **HTTPS استفاده کن**

4. **Input validation**

## 📊 مثال استفاده

1. مرورگر رو باز کن: `http://localhost:5000`
2. میم رو بکش توی باکس یا کلیک کن
3. دکمه "تحلیل میم" رو بزن
4. نتایج رو ببین:
   - تکنیک‌های شناسایی شده (سبز رنگ)
   - امتیاز هر تکنیک (نمودار)
   - جزئیات کامل

## ✅ چک‌لیست راه‌اندازی

- [ ] بسته‌ها نصب شدند
- [ ] سرور اجرا شد
- [ ] سایت در مرورگر باز شد
- [ ] یک میم تست آپلود کردم
- [ ] نتایج نمایش داده شد
- [ ] (اختیاری) مدل AdaBoost وصل شد
- [ ] (اختیاری) OCR اضافه شد
- [ ] (اختیاری) Image features اضافه شد

## 🆘 کمک و پشتیبانی

مشکل داری؟ این فایل‌ها رو چک کن:

1. **راهنمای اتصال مدل:** `MODEL_INTEGRATION_GUIDE.md`
2. **راهنمای کلی:** `../INTEGRATION_GUIDE.md`
3. **مستندات کامل:** `../README_KG_ARCHITECTURE.md`
4. **نوت‌بوک نمونه:** `../START_HERE_Integration.ipynb`

## 🎓 تکنولوژی‌ها

- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
- **AI Model:** PyTorch + AdaBoost Ensemble
- **Knowledge Graph:** NetworkX
- **Styling:** Custom CSS با طراحی Gradient

## 📄 License

همون لایسنسی که پروژه اصلی داره.

---

**ساخته شده با ❤️ برای تشخیص بهتر پروپاگاندا**

**موفق باشید! 🚀**
