setlocal
SET ENV_NAME=poai_gpu
IF NOT "%1"=="" SET ENV_NAME=%1

%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "&'C:\Users\%USERNAME%\Anaconda3\shell\condabin\conda-hook.ps1'; conda activate 'C:\Users\%USERNAME%\Anaconda3'; conda create -y -n %ENV_NAME% python=3.7;  conda activate %ENV_NAME%; conda install -y tensorflow-gpu==2.1; conda install -y pytorch=1.5.1 torchvision cudatoolkit=10.1 -c pytorch; conda install -y -c conda-forge wxpython; conda install -y -c conda-forge gooey==1.0.4; pip install keras==2.3.1; conda install -y scikit-learn; conda install -y Cython; conda install -y pandas; conda install -y colorama; conda install -y java-jdk -c cyclus; conda install -y pip; pip install konlpy; pip install -U gensim; pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"
endlocal

