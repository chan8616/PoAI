apt-get update
apt-get install -y dpkg-dev \
    build-essential \
    freeglut3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libgtk-3-dev \
    libjpeg-dev \
    libnotify-dev \
    libpng-dev \
    libsdl2-dev \
    libsm-dev \
    libtiff5-dev \
    libwebkit2gtk-4.0-dev \
    libxtst-dev \
    libcanberra-gtk3-module

conda create -n TensorflowGUI_CPU -c anaconda python=3.6 tensorflow==1.11 keras=2.2.4 scikit-learn pandas xlsxwriter colorama 
conda install -n TensorflowGUI_CPU -c conda-forge wxpython treelib gooey==1.0.2 pycocotools Cython imgaug
