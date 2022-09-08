#!/bin/bash
cd ..
tar -cf indicator.xz indicator
scp indicator.xz signals2://home/bot
echo Done

