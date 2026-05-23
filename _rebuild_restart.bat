@echo off
timeout /t 2 /nobreak > nul
cd /d "C:\Users\admin\Desktop\optimsolution"
cmake --build build --config Release
if %ERRORLEVEL% neq 0 (
    echo Rebuild needed > "C:\Users\admin\Desktop\optimsolution\.rebuild_pending"
)
start "" "C:\Users\admin\Desktop\optimsolution\build\Release\optimsolution_gui.exe" C:\Users\admin\Desktop\optimsolution\build\Release\optimsolution_gui.exe
