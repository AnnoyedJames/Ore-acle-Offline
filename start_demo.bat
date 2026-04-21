@echo off
echo Starting Backend Server...
start "Backend" cmd /c "python -m backend.api.server"

echo Starting Frontend Dev Server...
start "Frontend" cmd /c "cd frontend && npm run dev"

echo Waiting a few seconds for servers to initialize...
timeout /t 5 /nobreak > nul

echo Opening Ore-acle in your default web browser...
start http://localhost:5173

echo All services started! Close the command windows to stop the servers.
