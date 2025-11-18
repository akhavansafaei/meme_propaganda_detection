# 🚀 راهنمای سریع - نسخه جدید با Ground Truth!

## ✨ چی جدید شد؟

### 1️⃣ **پیش‌بینی‌های هوشمند**
- ✅ فعلاً از **شبیه‌سازی هوشمند** استفاده می‌کنه (نه تصادفی!)
- ✅ متن رو تحلیل می‌کنه و براساسش پیش‌بینی می‌کنه
- ✅ آماده برای اتصال مدل واقعی AdaBoost شماست

### 2️⃣ **مقایسه با Ground Truth**
- ✅ می‌تونی لیبل‌های واقعی رو انتخاب کنی
- ✅ سیستم Precision, Recall, F1 رو حساب می‌کنه
- ✅ True Positive, False Positive, False Negative رو نشون میده

### 3️⃣ **نمایش بهتر**
- ✅ Neural → KG → Final score رو نشون میده
- ✅ رنگ‌بندی براساس نتیجه (درست، اشتباه، از قلم افتاده)
- ✅ وضعیت مدل (واقعی یا شبیه‌سازی) رو نشون میده

---

## 🏃 اجرای سریع

```bash
cd webapp
python app.py
```

باز کن: `http://localhost:5000`

---

## 📸 چطور استفاده کنیم؟

### حالت 1: بدون Ground Truth (فقط پیش‌بینی)

```
1. میم رو آپلود کن
2. دکمه "تحلیل میم" رو بزن
3. نتایج رو ببین
```

### حالت 2: با Ground Truth (مقایسه)

```
1. میم رو آپلود کن
2. لیبل‌های واقعی رو انتخاب کن (checkbox ها)
3. دکمه "تحلیل میم" رو بزن
4. نتایج + مقایسه + Precision/Recall/F1 رو ببین!
```

---

## 🎯 مثال عملی

### تست 1: میم با متن

**متن میم:**
```
"These corrupt politicians are destroying our freedom!"
```

**بدون Ground Truth:**
- سیستم پیش‌بینی می‌کنه
- نتایج رو نشون میده

**با Ground Truth:**
1. لیبل‌های واقعی رو انتخاب کن:
   - ✅ Loaded Language
   - ✅ Appeal to (Strong) Emotions
2. تحلیل کن
3. ببین:
   - ✓ True Positive: Loaded Language (درست تشخیص داده شد)
   - ✗ False Positive: Bandwagon (اشتباه تشخیص داده شد)
   - ✗ False Negative: هیچ (چیزی از قلم نیفتاده)
   - **Precision: 85%**
   - **Recall: 100%**
   - **F1: 92%**

---

## 🔧 اتصال مدل واقعی (10 دقیقه)

### گزینه A: استفاده از مدل ذخیره شده

اگه قبلاً مدل رو آموزش دادی و ذخیره کردی:

```bash
# 1. مدل رو کپی کن به پوشه saved_models
mkdir -p saved_models
cp path/to/your/model.pth saved_models/adaboost_model.pth

# 2. model_loader.py رو ویرایش کن
# خط 44-58 رو پیدا کن و با کد واقعیت جایگزین کن

# 3. سرور رو دوباره اجرا کن
python app.py
```

### گزینه B: استفاده از نوت‌بوک

```python
# توی نوت‌بوک ali_write_adaboost_ensemble.ipynb

# بعد از آموزش مدل:
import torch

checkpoint = {
    'model_state_dict': model.state_dict(),
    'config': config,
    # هر چیز دیگه‌ای که لازم داری
}

torch.save(checkpoint, '../saved_models/adaboost_model.pth')
print("✅ مدل ذخیره شد!")
```

سپس `model_loader.py` رو ویرایش کن:

```python
# خط 44-58 - load_real_model:

def load_real_model(self, model_path):
    try:
        checkpoint = torch.load(model_path, map_location=self.device)

        # مدل رو بساز (از نوت‌بوک کپی کن)
        self.model = YourModelClass(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print("✅ مدل بارگذاری شد")
        self.use_real_model = True

    except Exception as e:
        print(f"❌ خطا: {e}")
        self.use_real_model = False
```

و خط 105-125 - `_predict_real`:

```python
def _predict_real(self, image_path, text):
    with torch.no_grad():
        # پیش‌پردازش (از نوت‌بوک کپی کن)
        image = self.preprocess_image(image_path)

        # اگه text features لازم داری:
        # text_features = self.extract_text_features(text)

        # پیش‌بینی
        outputs = self.model(image)
        predictions = torch.sigmoid(outputs).cpu().numpy()[0]

        # تبدیل به dictionary
        result = {}
        for i, tech in enumerate(self.technique_names):
            result[tech] = float(predictions[i])

        return result
```

---

## 📊 چیزهایی که الان کار می‌کنه

### ✅ کار می‌کنه:

1. **آپلود تصویر** - Drag & Drop یا Click
2. **انتخاب Ground Truth** - 22 تکنیک
3. **تحلیل هوشمند** - براساس متن
4. **نمایش نتایج:**
   - Neural score (از مدل/شبیه‌سازی)
   - KG score (بعد از Knowledge Graph)
   - Final score (نهایی)
5. **محاسبه متریک‌ها:**
   - Precision
   - Recall
   - F1 Score
   - True/False Positives/Negatives
6. **رنگ‌بندی نتایج:**
   - 🟢 سبز: True Positive (درست تشخیص داده شد)
   - 🔴 قرمز: False Positive (اشتباه تشخیص داده شد)
   - 🟠 نارنجی: False Negative (از قلم افتاده)

### ⚠️ هنوز نیاز به کار داره:

1. **مدل واقعی** - فعلاً شبیه‌سازی هست
2. **OCR** - برای استخراج متن از تصویر (اختیاری)
3. **Image Features** - برای دقت بیشتر (اختیاری)

---

## 🎨 نحوه نمایش نتایج

### اگه Ground Truth ندادی:
```
┌─────────────────────────────┐
│ ✓ Loaded Language    75% ███│ سبز (شناسایی شد)
│ ○ Bandwagon          45% ██ │ خاکستری (نه)
└─────────────────────────────┘
```

### اگه Ground Truth دادی:
```
Ground Truth: Loaded Language, Fear Appeal

┌─────────────────────────────────────┐
│ ✓ Loaded Language       75% ███     │ سبز (True Positive)
│ ✓ Fear Appeal           68% ███     │ سبز (True Positive)
│ ✗ Bandwagon             45% ██      │ قرمز (False Positive)
│ ✗ Doubt                 0%          │ نارنجی (False Negative)
└─────────────────────────────────────┘

📊 Metrics:
Precision: 67% (2/3 درست از پیش‌بینی‌ها)
Recall: 100% (2/2 پیدا کرده)
F1: 80%
```

---

## 🐛 عیب‌یابی

### مشکل: "شبیه‌سازی" می‌نویسه بالای صفحه

**طبیعیه!** مدل واقعی رو هنوز وصل نکردی.

**حل:** بخش "اتصال مدل واقعی" بالا رو دنبال کن

---

### مشکل: Precision/Recall نمیاد

**چک کن:** Ground Truth انتخاب کردی؟

**حل:** قبل از "تحلیل میم"، تکنیک‌های واقعی رو انتخاب کن

---

### مشکل: همه چیز تصادفیه!

**نه!** الان براساس متن پیش‌بینی می‌کنه (شبیه‌سازی هوشمند)

**تست:** یک میم با این متن آپلود کن:
```
"These corrupt politicians are evil!"
```

باید "Loaded Language" رو تشخیص بده!

---

## 📁 فایل‌های مهم

```
webapp/
├── app.py                  ← Backend (آپدیت شده ✅)
├── model_loader.py         ← مدل (شبیه‌سازی هوشمند ✅)
├── templates/
│   └── index.html          ← Frontend (نسخه جدید ✅)
├── MODEL_INTEGRATION_GUIDE.md  ← راهنمای اتصال مدل
└── QUICK_START.md          ← این فایل!
```

---

## ✅ چک‌لیست

### استفاده فوری:
- [ ] `python app.py` اجرا کردم
- [ ] مرورگر باز شد: `http://localhost:5000`
- [ ] یک میم آپلود کردم
- [ ] نتایج رو دیدم
- [ ] Ground Truth امتحان کردم
- [ ] Metrics رو دیدم

### برای استفاده واقعی:
- [ ] مدل رو آموزش دادم
- [ ] مدل رو ذخیره کردم
- [ ] `model_loader.py` رو ویرایش کردم
- [ ] سرور رو دوباره اجرا کردم
- [ ] "مدل واقعی فعال" رو دیدم ✅

---

## 💡 نکات مهم

1. **شبیه‌سازی هوشمند است نه تصادفی:**
   - متن رو تحلیل می‌کنه
   - کلمات کلیدی رو پیدا می‌کنه
   - براساسش امتیاز میده

2. **Ground Truth اختیاریه:**
   - فقط برای مقایسه و تست استفاده میشه
   - اگه نمیدونی، خالی بذار

3. **مدل واقعی بهتره:**
   - دقت بیشتر
   - استفاده از CLIP + RoBERTa
   - نتایج واقعی

---

## 🎯 مثال کامل

```
1. سرور رو اجرا کن:
   python app.py

2. مرورگر رو باز کن:
   http://localhost:5000

3. یک میم آپلود کن

4. (اختیاری) Ground Truth انتخاب کن:
   ☑ Loaded Language
   ☑ Appeal to (Strong) Emotions

5. "تحلیل میم" رو بزن

6. نتایج:
   ✓ Loaded Language (True Positive)
   ✓ Appeal to (Strong) Emotions (True Positive)
   ✗ Bandwagon (False Positive)

   Precision: 67%
   Recall: 100%
   F1: 80%

7. بررسی کن:
   Neural → KG → Final scores رو ببین
```

---

**موفق باشی! 🚀**

اگه سوالی داشتی یا مشکلی پیش اومد، بگو تا کمکت کنم!
