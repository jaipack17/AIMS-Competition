@echo off
rem Ridge stage: 1) fast exact solver (float64, ~10-15 min), 2) sklearn RidgeCV float32 check (~2 h).
rem Both save every finished fit (with a DONE marker) and skip finished fits, so after a shutdown
rem just double-click this file again and it continues where it stopped.
cd /d "%~dp0"
python pipeline\ridge_train.py >> logs\ridge_fast.log 2>&1
if errorlevel 1 (
  echo fast ridge failed, see logs\ridge_fast.log >> logs\ridge_sklearn.log
  exit /b 1
)
python pipeline\ridge_sklearn.py >> logs\ridge_sklearn.log 2>&1
