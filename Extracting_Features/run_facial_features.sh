#!/bin/bash
chmod +x run.sh
cd ./3DDFA_V2/
# ls -s
python3 parallel_run.py $1 > output.txt