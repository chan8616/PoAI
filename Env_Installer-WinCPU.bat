setlocal
SET ENV_NAME=TensorflowGUI_cpu
IF NOT "%1"=="" SET ENV_NAME=%1

%windir%\System32\WindowsPowerShell\v1.0\powershell.exe -ExecutionPolicy ByPass -NoExit -Command "&'C:\Users\%USERNAME%\Anaconda3\shell\condabin\conda-hook.ps1'; conda activate 'C:\Users\%USERNAME%\Anaconda3'; conda create -y -n %ENV_NAME% python=3.6;  conda activate %ENV_NAME%; conda install -y tensorflow==1.11; conda install -y -c conda-forge wxpython; conda install -y -c conda-forge treelib; conda install -y -c conda-forge gooey==1.0.2; pip install keras==2.2.3; conda install -y scikit-learn matplotlib; conda install -y Cython; conda install -y pandas; conda install -y xlsxwriter; conda install -y -c conda-forge imgaug; conda install -y colorama; conda install -y pip; pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI"
endlocal

