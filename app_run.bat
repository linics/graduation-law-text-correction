@echo off
cd /d "D:\deeplearning\The graduation design"
echo Activating Conda environment...
call conda activate gradesign
if errorlevel 1 (
    echo Failed to activate Conda environment. Please check if Conda is installed and the environment exists.
    pause
    exit
)
echo Starting Streamlit app...
streamlit run app.py
echo Streamlit exited. Press any key to close...
pause
