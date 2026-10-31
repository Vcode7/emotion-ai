@echo off
REM ============================================================
REM  setup_avatar.bat — [DEPRECATED]
REM  Notice: Visual Chat now runs real-time 3D FaceCap avatar directly
REM  in the browser (Three.js / ARKit morph targets). SadTalker and
REM  LivePortrait are no longer required or used by Emotion AI.
REM ============================================================

setlocal EnableDelayedExpansion

SET SCRIPT_DIR=%~dp0
SET THIRD_PARTY=%SCRIPT_DIR%third_party
SET WITH_LIVEPORTRAIT=0
SET CPU_ONLY=0

REM Parse args
:parse_args
IF "%~1"=="--with-liveportrait" (
    SET WITH_LIVEPORTRAIT=1
    SHIFT
    GOTO parse_args
)
IF "%~1"=="--cpu-only" (
    SET CPU_ONLY=1
    SHIFT
    GOTO parse_args
)

echo.
echo ================================================================
echo   Emotion AI — Avatar Setup (SadTalker + LivePortrait)
echo ================================================================
echo.

REM Check prerequisites
where git >nul 2>&1 || (echo [ERROR] git is not installed. Please install git. && exit /b 1)
where python >nul 2>&1 || (echo [ERROR] Python not found. && exit /b 1)
where ffmpeg >nul 2>&1 || (echo [WARNING] FFmpeg not found in PATH. SadTalker requires FFmpeg.)

mkdir "%THIRD_PARTY%" 2>nul

REM ============================================================
REM  1. Clone SadTalker
REM ============================================================
IF NOT EXIST "%THIRD_PARTY%\SadTalker" (
    echo [1/5] Cloning SadTalker...
    git clone https://github.com/OpenTalker/SadTalker.git "%THIRD_PARTY%\SadTalker"
    IF ERRORLEVEL 1 (echo [ERROR] Failed to clone SadTalker. && exit /b 1)
) ELSE (
    echo [1/5] SadTalker already cloned — skipping.
)

REM ============================================================
REM  2. Install SadTalker requirements
REM ============================================================
echo [2/5] Installing SadTalker Python requirements...
pip install -r "%THIRD_PARTY%\SadTalker\requirements.txt"
IF ERRORLEVEL 1 (echo [ERROR] SadTalker pip install failed. && exit /b 1)

REM ============================================================
REM  3. Download SadTalker checkpoints
REM ============================================================
echo [3/5] Downloading SadTalker pretrained weights...
IF NOT EXIST "%THIRD_PARTY%\SadTalker\checkpoints" (
    mkdir "%THIRD_PARTY%\SadTalker\checkpoints"
    REM Download from HuggingFace
    python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='vinthony/SadTalker', local_dir='%THIRD_PARTY%\SadTalker\checkpoints', local_dir_use_symlinks=False)"
    IF ERRORLEVEL 1 (
        echo [WARNING] HuggingFace download failed. Trying direct script...
        IF EXIST "%THIRD_PARTY%\SadTalker\scripts\download_models.sh" (
            echo   Use: bash scripts/download_models.sh from the SadTalker directory.
        )
    )
) ELSE (
    echo [3/5] SadTalker checkpoints already present — skipping.
)

REM Download GFPGAN weights (face enhancer)
echo [3b/5] Downloading GFPGAN weights...
IF NOT EXIST "%THIRD_PARTY%\SadTalker\gfpgan\weights" (
    mkdir "%THIRD_PARTY%\SadTalker\gfpgan\weights"
    python -c "import urllib.request; urllib.request.urlretrieve('https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth', '%THIRD_PARTY%\SadTalker\gfpgan\weights\GFPGANv1.4.pth')"
)

REM ============================================================
REM  4. Verify portrait image
REM ============================================================
echo [4/5] Checking avatar portrait...
IF NOT EXIST "%SCRIPT_DIR%assets\avatar_portrait.jpg" (
    echo [ERROR] Portrait image not found at backend\assets\avatar_portrait.jpg
    echo         Please ensure the portrait image is present before running.
    exit /b 1
) ELSE (
    echo [4/5] Avatar portrait found.
)

REM ============================================================
REM  5. LivePortrait (optional)
REM ============================================================
IF %WITH_LIVEPORTRAIT%==1 (
    echo [5/5] Setting up LivePortrait...
    IF NOT EXIST "%THIRD_PARTY%\LivePortrait" (
        git clone https://github.com/KwaiVGI/LivePortrait.git "%THIRD_PARTY%\LivePortrait"
        IF ERRORLEVEL 1 (echo [ERROR] Failed to clone LivePortrait. && exit /b 1)
        pip install -r "%THIRD_PARTY%\LivePortrait\requirements.txt"
        python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='KwaiVGI/LivePortrait', local_dir='%THIRD_PARTY%\LivePortrait\pretrained_weights', local_dir_use_symlinks=False)"
    ) ELSE (
        echo [5/5] LivePortrait already present — skipping.
    )
) ELSE (
    echo [5/5] Skipping LivePortrait (use --with-liveportrait to enable).
    echo       For maximum quality, re-run:  setup_avatar.bat --with-liveportrait
)

echo.
echo ================================================================
echo   Avatar setup complete!
echo.
echo   SadTalker:     %THIRD_PARTY%\SadTalker
IF %WITH_LIVEPORTRAIT%==1 (
echo   LivePortrait:  %THIRD_PARTY%\LivePortrait
)
echo   Portrait:      %SCRIPT_DIR%assets\avatar_portrait.jpg
echo   Output dir:    %SCRIPT_DIR%static\avatar\
echo.
echo   Start the backend with:  uvicorn main:app --reload
echo ================================================================
echo.
