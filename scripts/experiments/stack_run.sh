# CUDA_VISIBLE_DEVICES="1" accelerate launch --main_process_port 29549 --num_processes 1 trainer.py --date 0623_014740 --task S1_d_trulyleq7 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq7' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq7/0623_014740/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq7/0623_014740/terminal.txt'  >/dev/null &
# CUDA_VISIBLE_DEVICES="2" accelerate launch --main_process_port 29550 --num_processes 1 trainer.py --date 0623_020523 --task S1_d_trulyleq8 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq8' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq8/0623_020523/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq8/0623_020523/terminal.txt'  >/dev/null &
# wait

# python standalone_inference.py --cuda 0 --task_suites 'S1_d_trulyleq[2-8]_2I' > /dev/null 2>&1 &
# python standalone_inference.py --cuda 0 --task_suites 'S1_d_trulyleq[2-8]_2J' > /dev/null 2>&1 &
# wait

# python run.py --port 29508 --cuda 0 --task S2_d_trulyleq3 > /dev/null 2>&1 &
# python run.py --port 29509 --cuda 0 --task S2_d_trulyleq4 > /dev/null 2>&1 &
# python run.py --port 29510 --cuda 0 --task S2_d_trulyleq2 > /dev/null 2>&1 &
# python run.py --port 29511 --cuda 1 --task S2_d_trulyleq2 > /dev/null 2>&1 &
# python run.py --port 29512 --cuda 1 --task S2_d_trulyleq3 > /dev/null 2>&1 &
# python run.py --port 29513 --cuda 1 --task S2_d_trulyleq4 > /dev/null 2>&1 &
# python run.py --port 29514 --cuda 1 --task S2_d_trulyleq3 > /dev/null 2>&1 &
# python run.py --port 29515 --cuda 2 --task S2_d_trulyleq4 > /dev/null 2>&1 &
# python run.py --port 29516 --cuda 2 --task S2_d_trulyleq5 > /dev/null 2>&1 &
# python run.py --port 29517 --cuda 2 --task S2_d_trulyleq2 > /dev/null 2>&1 &
# python run.py --port 29518 --cuda 3 --task S2_d_trulyleq3 > /dev/null 2>&1 &
# python run.py --port 29519 --cuda 3 --task S2_d_trulyleq4 > /dev/null 2>&1 &
# python run.py --port 29520 --cuda 3 --task S2_d_trulyleq5 > /dev/null 2>&1 &
# python run.py --port 29521 --cuda 3 --task S2_d_trulyleq2 > /dev/null 2>&1 &
# python run.py --port 29507 --cuda 3 --task S2_d_trulyleq2 > /dev/null 2>&1 &
python run.py --port 29510 --cuda 2 --task S2_d_trulyleq3 > /dev/null 2>&1 &
# python run.py --port 29509 --cuda 1 --task S2_d_trulyleq4 > /dev/null 2>&1 &
