#!/bin/bash
python script.py --train --dp --sst --epoch=20 --ptune &&  python script.py --train --sst --epoch=20 --ptune