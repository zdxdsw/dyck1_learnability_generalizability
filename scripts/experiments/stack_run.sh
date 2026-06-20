# python run.py --port 29514 --cuda 3 --task S1_d_trulyleq2 > /dev/null 2>&1 &
# python run.py --port 29515 --cuda 1 --task S1_d_trulyleq3 > /dev/null 2>&1 &
# python run.py --port 29516 --cuda 2 --task S1_d_trulyleq4 > /dev/null 2>&1 &
# python run.py --port 29500 --cuda 0 --task S1_d_4 > /dev/null 2>&1 &
# python run.py --port 29518 --cuda 3 --task S1_d_trulyleq6 > /dev/null 2>&1 &
# python run.py --port 29519 --cuda 1 --task S1_d_trulyleq7 > /dev/null 2>&1 &
# python run.py --port 29520 --cuda 2 --task S1_d_trulyleq8 > /dev/null 2>&1 &

# CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 29500 --num_processes 1 trainer.py --date 0619_203905 --task S1_d_trulyleq8 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_2P/S1_d_trulyleq8' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_2P/S1_d_trulyleq8/0619_203905/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_2P/S1_d_trulyleq8/0619_203905/terminal.txt'  >/dev/null &
# CUDA_VISIBLE_DEVICES="3" accelerate launch --main_process_port 29501 --num_processes 1 trainer.py --date 0619_172018 --task S1_d_trulyleq3 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_2P/S1_d_trulyleq3' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_2P/S1_d_trulyleq3/0619_172018/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_2P/S1_d_trulyleq3/0619_172018/terminal.txt'  >/dev/null &
# wait

python standalone_inference.py --cuda 0 --task_suites 'S1_d_trulyleq[2-8]_1B' > /dev/null 2>&1 &
python standalone_inference.py --cuda 1 --task_suites 'S1_d_trulyleq[2-8]_1C' > /dev/null 2>&1 &
python standalone_inference.py --cuda 3 --task_suites 'S1_d_trulyleq[2-8]_1D' > /dev/null 2>&1 &
wait


