@echo off
title PhotoGIMP Crack Annotation Studio
chcp 65001 >nul
echo ======================================================================
echo    KHỞI ĐỘNG PHOTOGIMP CRACK ANNOTATION STUDIO
echo ======================================================================
echo [*] Đang kích hoạt GIMP Guard (chống Color Picker khi dùng Ctrl)...
start /b "" python "d:\truong\SpecialSubjectTTNT\scripts\gimp_guard.py"

echo [*] Đang mở PhotoGIMP...
"C:\Users\ASUS\AppData\Local\Programs\GIMP 3\bin\gimp-3.exe"

echo.
echo [*] GIMP đã đóng. Đang tắt GIMP Guard...
powershell -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like '*gimp_guard.py*' } | Stop-Process -Force"
echo [*] Hoàn tất dọn dẹp.
exit
