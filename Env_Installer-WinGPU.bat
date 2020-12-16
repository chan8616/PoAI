@echo off
setlocal
SET ENV_NAME=poai
IF NOT "%1"=="" SET ENV_NAME=%1
%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "&'C:\Users\%USERNAME%\Miniconda3\shell\condabin\conda-hook.ps1'; conda activate 'C:\Users\%USERNAME%\Miniconda3'; conda create -y -n %ENV_NAME% python=3.7; conda activate %ENV_NAME%; conda install -y tensorflow-gpu==2.3.0 keras==2.3.1 scikit-learn matplotlib tensorboard Cython pandas colorama; conda install -y -c pytorch pytorch=1.5.1 torchvision cudatoolkit=10.1; conda install -y -c conda-forge gooey; conda install -y -c cyclus java-jdk; pip install konlpy; pip install -U gensim; pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"
endlocal