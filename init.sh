#!/usr/bin/bash

if [ ! -d 'ven' ]; then

    virtualenv ven
fi
source ven/bin/activate
pip install -r requirements.txt
python3 main.py
