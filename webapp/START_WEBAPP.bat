@echo off
REM راه‌اندازی سریع وب‌اپلیکیشن تشخیص پروپاگاندا (Windows)

echo ==================================
echo 🚀 Meme Propaganda Detection WebApp
echo ==================================
echo.

REM چک کردن Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python نصب نیست!
    pause
    exit /b 1
)

echo ✅ Python پیدا شد
echo.

REM نصب بسته‌ها
echo 📦 نصب بسته‌های مورد نیاز...
pip install -q -r requirements.txt

if errorlevel 1 (
    echo ❌ خطا در نصب بسته‌ها
    pause
    exit /b 1
)

echo ✅ بسته‌ها نصب شدند
echo.

REM ساخت پوشه uploads
if not exist "uploads" mkdir uploads
echo ✅ پوشه uploads ساخته شد
echo.

echo ==================================
echo 📝 یادآوری مهم:
echo ==================================
echo.
echo ⚠️  هنوز باید مدل AdaBoost خودت رو وصل کنی!
echo.
echo    راهنما: MODEL_INTEGRATION_GUIDE.md
echo.
echo ==================================
echo.

REM اجرای سرور
echo 🌐 راه‌اندازی سرور...
echo.
echo سایت در آدرس زیر در دسترس است:
echo http://localhost:5000
echo.
echo برای متوقف کردن سرور: Ctrl+C
echo.
echo ==================================
echo.

python app.py

pause
