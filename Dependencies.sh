if [ "$#" -ne 1 ]; then
  echo "Give only one argument, the name of the virtual environment to create"
  exit
fi

python3 -m venv $1

source $1/bin/activate

pip install --default-timeout=100 \
  setuptools \
	numpy \
	opencv-python \
	pygame \
	scipy \
	jupyter \
  numba \
  nose \
  parameterized \
  scikit-learn \
  Pillow \
  typeguard