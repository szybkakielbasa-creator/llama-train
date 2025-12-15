@echo off
setlocal enabledelayedexpansion
REM -------------------------
REM SSOMAR AI PRO - fixed run.bat
REM -------------------------

REM Root (folder, w którym leży ten plik)
set "ROOT=%~dp0"
REM usuń końcowy backslash jeśli potrzebujesz: (opcjonalne)
REM set "ROOT=%ROOT:~0,-1%"

REM GLOBAL CONFIG (dostosuj jeśli trzeba)
set "MODEL_URL=hf.co/speakleash/Bielik-4.5B-v3.0-Instruct-GGUF:Q8_0"
set "OLLAMA_MODEL=bielik-ssomar"
set "PYTHON=python"

REM FOLDERS
mkdir "%ROOT%data" 2>nul
mkdir "%ROOT%models" 2>nul
mkdir "%ROOT%scripts" 2>nul
mkdir "%ROOT%tools" 2>nul
mkdir "%ROOT%logs" 2>nul

REM Ustaw kodowanie konsoli na UTF-8 by lepiej wyświetlać PL znaki
chcp 65001 >nul

:MENU
cls
echo.
echo =============================
echo SSOMAR AI PRO (fixed)
echo =============================
echo 1 - Train model (validate, make_train, finetune, create)
echo 2 - Test model (autotest)
echo 3 - Update dataset (validate + split)
echo 4 - Rebuild llama.cpp (clone ^& build)
echo 5 - Scrape and create dataset
echo 6 - Reduce dataset
echo 0 - Exit
echo.
set /p CHOICE= Wybierz opcje (0-4) ^> 

if "%CHOICE%"=="1" goto TRAIN
if "%CHOICE%"=="2" goto TEST
if "%CHOICE%"=="3" goto DATA
if "%CHOICE%"=="4" goto BUILD
if "%CHOICE%"=="5" goto SCRAPE
if "%CHOICE%"=="6" goto REDUCE
if "%CHOICE%"=="0" exit /b 0
goto MENU

:DATA
echo [DATA] Walidacja + split
if not exist "%ROOT%scripts\validate_split.py" (
  echo ERROR: brak scripts\validate_split.py - prosze utworzyc plik zgodnie z instrukcja.
  pause
  goto MENU
)
"%PYTHON%" "%ROOT%scripts\validate_split.py"
echo [DATA] zakonczone.
pause
goto MENU

:BUILD
echo [BUILD] llama.cpp (clone & build)
if not exist "%ROOT%tools\llama.cpp\.git" (
  echo Cloning llama.cpp...
  git clone https://github.com/ggerganov/llama.cpp "%ROOT%tools\llama.cpp"
) ELSE (
  echo llama.cpp already present.
)
echo Creating build folder...
cd /d "%ROOT%tools\llama.cpp"
if not exist "build" mkdir build
cd build
echo Running cmake...
cmake .. -DLLAMA_CUBLAS=ON
if errorlevel 1 (
  echo CMake failed. Sprawdz logi.
  cd /d "%ROOT%"
  pause
  goto MENU
)
cmake --build . --config Release
cd /d "%ROOT%"
echo [BUILD] done.
pause
goto MENU

:SCRAPE
echo [SCRAPE] Scraping and creating dataset
if not exist "%ROOT%scripts\scrape.py" (
  echo ERROR: brak scripts\scrape.py
  pause
  goto MENU
)
echo Podaj URL-e do scrapowania (oddzielone spacjami, np. https://example.com https://example2.com):
set /p URLS=
if "%URLS%"=="" (
  echo Brak URL-i, używam domyślnych.
  set URLS=https://github.com/ssomar/ExecutableItems/wiki https://github.com/LuckPerms/LuckPerms/wiki
)
echo Podaj max stron per seed (domyślnie 200):
set /p MAX_PAGES=
if "%MAX_PAGES%"=="" set MAX_PAGES=200
"%PYTHON%" "%ROOT%scripts\scrape.py" --urls %URLS% --out "%ROOT%raw_sources" --make-dataset --dataset-out "%ROOT%data\sft_dataset.jsonl" --max-pages-per-seed %MAX_PAGES%
echo [SCRAPE] done.
pause
goto MENU

:REDUCE
echo [REDUCE] Reducing dataset
if not exist "%ROOT%scripts\reduce_dataset.py" (
  echo ERROR: brak scripts\reduce_dataset.py
  pause
  goto MENU
)
echo Podaj maksymalną liczbę linii (domyślnie 800):
set /p MAX_LINES=
if "%MAX_LINES%"=="" set MAX_LINES=800
echo Czy losowe próbkowanie? (y/n, domyślnie n):
set /p RANDOM_CHOICE=
if "%RANDOM_CHOICE%"=="y" set RANDOM_FLAG=--random
"%PYTHON%" "%ROOT%scripts\reduce_dataset.py" --input "%ROOT%data\sft_dataset.jsonl" --output "%ROOT%data\sft_dataset_reduced.jsonl" --max_lines %MAX_LINES% %RANDOM_FLAG%
echo [REDUCE] done. Użyj zmniejszonego pliku jeśli potrzebujesz.
pause
goto MENU

:TRAIN
echo [TRAIN] START
if not exist "%ROOT%scripts\validate_split.py" (
  echo ERROR: brak scripts\validate_split.py
  pause
  goto MENU
)
if not exist "%ROOT%scripts\make_train.py" (
  echo ERROR: brak scripts\make_train.py
  pause
  goto MENU
)

echo 1) Validate + split
"%PYTHON%" "%ROOT%scripts\validate_split.py"

echo 2) Make train.txt
"%PYTHON%" "%ROOT%scripts\make_train.py"

echo 3) Pull base model into Ollama (moze wymaga logowania)
echo ollama pull %MODEL_URL%
ollama pull %MODEL_URL%

echo 4) Download base GGUF if not present
if not exist "%ROOT%models\base.gguf" (
  echo Downloading base model...
  set "GGUF_URL=https://huggingface.co/speakleash/Bielik-4.5B-v3.0-Instruct-GGUF/resolve/main/Bielik-4.5B-v3.0-Instruct.Q8_0.gguf"
  set "GGUF_FILE=%ROOT%models\base.gguf"
  "%PYTHON%" "%ROOT%scripts\download_gguf.py" "%GGUF_URL%" "%GGUF_FILE%"
  if %errorlevel% neq 0 (
    echo Download failed.
    pause
    goto MENU
  )
)

REM 5) Run finetune if finetune.exe available
set "FINETUNE_EXE=%ROOT%tools\llama.cpp\build\bin\Release\finetune.exe"
if exist "%FINETUNE_EXE%" (
  echo 5) Running finetune (llama.cpp finetune.exe) - obserwuj logi/wyjscie
  "%FINETUNE_EXE%" ^
   --model-base "%ROOT%models\base.gguf" ^
   --train-data "%ROOT%data\train.txt" ^
   --lora-out "%ROOT%models\lora-ssomar.gguf" ^
   --lora-r 8 ^
   --lora-alpha 16 ^
   --ctx 2048 ^
   --batch 1 ^
   --grad-accum 8 ^
   --epochs 3 ^
   --gpu-layers 20 ^
   --sample-start "<s>"
  if errorlevel 1 echo Finetune zakończony z błędem. Sprawdź logi.
) ELSE (
  echo WARN: finetune.exe not found, pomijam etap finetune. Zbuduj llama.cpp lub użyj innego trenera.
)

REM 6) Create Ollama model with adapter if available
if exist "%ROOT%models\lora-ssomar.gguf" (
  echo 6) Tworze Modelfile i rejestruje w Ollama...
  > "%ROOT%Modelfile" (
    echo FROM %MODEL_URL%
    echo ADAPTER ./models/lora-ssomar.gguf
    echo SYSTEM "Ekspert SSOMAR. ZACHOWUJ FORMAT."
  )
  ollama create %OLLAMA_MODEL% -f "%ROOT%Modelfile"
) ELSE (
  echo Brak adaptera models\lora-ssomar.gguf - pomijam create.
)

echo TRAIN finished.
pause
goto MENU

:TEST
echo [TEST] Auto-test
if not exist "%ROOT%scripts\autotest.py" (
  echo ERROR: brak scripts\autotest.py - prosze utworzyc plik.
  pause
  goto MENU
)
"%PYTHON%" "%ROOT%scripts\autotest.py"
echo Wynik testu zapisany w logs\test_output.txt (jesli skrypt dziala)
type "%ROOT%logs\test_output.txt" 2>nul || echo Brak logs\test_output.txt
pause
goto MENU
