#!/bin/bash
python script.py --train --dp --sst --paramF --portion=50 --epoch=20 && python script.py --train --dp --paramF --sst --portion=75 --epoch=20