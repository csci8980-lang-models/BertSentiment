#!/bin/bash
python script.py --train --dp --sst --plF --portion=50 --epoch=20 && python script.py --train --dp --plF --sst --portion=75 --epoch=20