@echo off
setlocal enableextensions enabledelayedexpansion
chcp 65001

set URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe
set "THIS_DIR=%cd%"
set PYTHON_VERSION=3.10

:build
call:miniconda.sh
call:venv
goto:my_end

:miniconda.sh
if exist miniconda.done goto:eof
echo Download mininconda
powershell Invoke-WebRequest -Uri '%URL%' -OutFile 'miniconda3.exe'
if !errorlevel! neq 0 exit /b !errorlevel!
echo.>miniconda.done
goto:eof

:venv
if exist venv.done goto:eof
echo Install Virtual Env
start /wait "" miniconda3.exe /InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=%THIS_DIR%\venv
rem similar to activate in bash
call %THIS_DIR%\venv\condabin\activate.bat
start /wait "" cmd /c conda install -y setuptools -c anaconda
start /wait "" cmd /c conda install -y pip -c anaconda
start /wait "" cmd /c conda update -y conda
start /wait "" cmd /c conda install -y python=%PYTHON_VERSION%
start /wait "" cmd /c conda install -y numpy
start /wait "" cmd /c conda install -y matplotlib
start /wait "" cmd /c conda install -y jupyterlab
start /wait "" cmd /c conda clean -f -y
start /wait "" cmd /c conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
start /wait "" cmd /c pip install -r ..\requirements.txt
if !errorlevel! neq 0 exit /b !errorlevel!
echo.>venv.done
goto:eof


:clean
@RD /S /Q %THIS_DIR%\venv
del *.done
del miniconda3.exe

:my_end
echo install.bat Done.
endlocal
