#!/bin/sh

PYTHON_EXEC=$(which python)
PYTHON_VER=$($PYTHON_EXEC --version)

echo "Found Python @ $PYTHON_EXEC"
echo "Found Python Version: $PYTHON_VER"

if [$PYTHON_VER -lt 3.0] ; 
then
    echo "Python Version Required 3.0, Please install Python 3"
    exit 1
fi
echo "Installing Dependencies"

python pip install -r ./requirements.txt

echo "$INSTALL_RESULT"