# python run.py --port 29514 --cuda 3 --task S1_d_trulyleq2 > /dev/null 2>&1 &
# python run.py --port 29515 --cuda 1 --task S1_d_trulyleq3 > /dev/null 2>&1 &
# python run.py --port 29516 --cuda 2 --task S1_d_trulyleq4 > /dev/null 2>&1 &
# python run.py --port 29500 --cuda 0 --task S1_d_4 > /dev/null 2>&1 &
# python run.py --port 29518 --cuda 3 --task S1_d_trulyleq6 > /dev/null 2>&1 &
# python run.py --port 29519 --cuda 1 --task S1_d_trulyleq7 > /dev/null 2>&1 &
# python run.py --port 29520 --cuda 2 --task S1_d_trulyleq8 > /dev/null 2>&1 &

CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 29500 --num_processes 1 trainer.py --date 0616_185628 --task S1_d_5 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_[2-8]_4C/S1_d_5' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_[2-8]_4C/S1_d_5/0616_185628/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_[2-8]_4C/S1_d_5/0616_185628/terminal.txt' >/dev/null &
wait