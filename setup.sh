#!bin/bash

# TeamProjML_env
python3 -m venv env

source env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

python -m ipykernel install --user --name=env --display-name "TeamProjML_env"

echo DONE
 ``