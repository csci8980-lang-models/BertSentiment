#!/bin/bash
python script.py --train --dp --sst --epoch=20 && python script.py --freeze --train --dp --sst --epoch=20