@echo off
echo Starting Big Data Cyber Security Analytics Project...

echo.
echo Starting Log Generator in a new window...
start "Log Generator" cmd /k "D:\Python\python.exe modules\module1_log_generator\log_generator.py"

echo.
echo Starting Streamlit Dashboard in a new window...
start "Dashboard" cmd /k "streamlit run modules\module6_dashboard\dashboard.py"

echo.
echo All components started!
echo Press any key to exit this launcher (the components will keep running in their new windows).
pause > nul
