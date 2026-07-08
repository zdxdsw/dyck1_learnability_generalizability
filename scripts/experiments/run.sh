#!/bin/bash

start=29500

for i in {0..6}; do
    port=$((start + i))
    d=$((i + 2))
    cuda=$((port % 4))

    echo "Launching: python run.py --port $port --cuda $cuda --task S1_d_trulyleq${d}"

    python run.py \
        --port "$port" \
        --cuda "$cuda" \
        --task "S1_d_trulyleq${d}" \
        > /dev/null 2>&1 &
done