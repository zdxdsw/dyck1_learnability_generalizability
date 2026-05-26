#!/bin/bash

start=29541

for i in {1..6}; do
    port=$((start + i))
    d=$((i + 2))
    cuda=$((port % 4))

    echo "Launching: python run.py --port $port --cuda $cuda --task S3_d_trulyleq${d}"
    
    python run.py \
        --port "$port" \
        --cuda "$cuda" \
        --task "S3_d_trulyleq${d}" \
        > /dev/null 2>&1 &
done