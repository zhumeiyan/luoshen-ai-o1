#!/bin/bash
# Wait for process 1276 to finish, then cat the output file
while ps -p 1276 > /dev/null 2>&1; do
    sleep 30
done
date
echo "=== FILE ==="
cat /mnt/data/catpaw/home/workspace/洛神赋1.1/luoshen_envrev_26_35.txt 2>&1 || echo NOFILE
echo "=== END ==="
