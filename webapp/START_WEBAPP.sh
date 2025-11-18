#!/bin/bash

# راه‌اندازی سریع وب‌اپلیکیشن تشخیص پروپاگاندا

echo "=================================="
echo "🚀 Meme Propaganda Detection WebApp"
echo "=================================="
echo ""

# رنگ‌ها برای خروجی
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# چک کردن Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 نصب نیست!${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python پیدا شد${NC}"

# نصب بسته‌ها
echo ""
echo "📦 نصب بسته‌های مورد نیاز..."
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ بسته‌ها نصب شدند${NC}"
else
    echo -e "${RED}❌ خطا در نصب بسته‌ها${NC}"
    exit 1
fi

# ساخت پوشه uploads
mkdir -p uploads
echo -e "${GREEN}✅ پوشه uploads ساخته شد${NC}"

# چک کردن فایل propaganda_techniques.json
if [ -f "../propaganda_techniques.json" ]; then
    echo -e "${GREEN}✅ فایل تکنیک‌ها پیدا شد${NC}"
else
    echo -e "${YELLOW}⚠️  فایل propaganda_techniques.json پیدا نشد${NC}"
    echo "   مطمئن شو که از پوشه webapp اجرا می‌کنی"
fi

echo ""
echo "=================================="
echo "📝 یادآوری مهم:"
echo "=================================="
echo ""
echo "⚠️  هنوز باید مدل AdaBoost خودت رو وصل کنی!"
echo ""
echo "   راهنما: MODEL_INTEGRATION_GUIDE.md"
echo ""
echo "   خلاصه قدم‌ها:"
echo "   1. مدل رو ذخیره کن: torch.save(...)"
echo "   2. model_loader.py بساز"
echo "   3. app.py رو ویرایش کن"
echo ""
echo "=================================="
echo ""

# اجرای سرور
echo "🌐 راه‌اندازی سرور..."
echo ""
echo -e "${GREEN}سایت در آدرس زیر در دسترس است:${NC}"
echo -e "${YELLOW}http://localhost:5000${NC}"
echo ""
echo "برای متوقف کردن سرور: Ctrl+C"
echo ""
echo "=================================="
echo ""

# اجرای Flask
python app.py
