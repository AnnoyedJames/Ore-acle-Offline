@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  Ore-acle Offline - Demo Launcher
echo ============================================
echo.

REM ---------- 1. Start Ollama (if not already running) ----------
echo [1/5] Checking Ollama...

REM Check if Ollama is already listening on port 11434
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 2 -ErrorAction Stop; exit 0 } catch { exit 1 }" >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo        Ollama is already running.
    goto :skip_ollama
)

echo        Starting Ollama...
start "Ollama" cmd /c "ollama serve"

REM Wait for Ollama to come online (poll up to 20 seconds)
echo        Waiting for Ollama to initialize...
set OLLAMA_READY=0
for /L %%i in (1,1,20) do (
    powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -TimeoutSec 2 -ErrorAction Stop; exit 0 } catch { exit 1 }" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        set OLLAMA_READY=1
        goto :ollama_ready
    )
    timeout /t 1 /nobreak >nul
)
echo        WARNING: Ollama did not start within 20s. Continuing anyway...
goto :skip_ollama

:ollama_ready
echo        Ollama is ready.

:skip_ollama
echo.

REM ---------- 2. Free port 8000 (stale backends) ----------
echo [2/5] Checking port 8765...
powershell -Command "$p=netstat -ano|Select-String ':8765\s'|Select-String LISTENING;if($p){$pid=$p.Line.Trim().Split()[-1];Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue;Start-Sleep 1}" >nul 2>&1
echo.

REM ---------- 3. Start Backend ----------
echo [3/5] Starting Backend Server...
start "Backend" cmd /c "python -m backend.api.server"

REM ---------- 4. Start Frontend ----------
echo [4/5] Starting Frontend Dev Server...
start "Frontend" cmd /c "cd frontend && npm run dev"

REM ---------- 5. Wait & Open Browser ----------
echo [5/5] Waiting for servers to initialize...
timeout /t 5 /nobreak > nul

echo        Opening Ore-acle in your default web browser...
start http://localhost:5173

echo.
echo ============================================
echo  All services started!
echo  Close the command windows to stop.
echo ============================================
pause
