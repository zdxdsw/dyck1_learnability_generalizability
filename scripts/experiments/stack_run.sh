# CUDA_VISIBLE_DEVICES="1" accelerate launch --main_process_port 29549 --num_processes 1 trainer.py --date 0623_014740 --task S1_d_trulyleq7 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq7' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq7/0623_014740/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq7/0623_014740/terminal.txt'  >/dev/null &
# CUDA_VISIBLE_DEVICES="2" accelerate launch --main_process_port 29550 --num_processes 1 trainer.py --date 0623_020523 --task S1_d_trulyleq8 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq8' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq8/0623_020523/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S1_d_trulyleq[2-8]_1I/S1_d_trulyleq8/0623_020523/terminal.txt'  >/dev/null &
# wait

python standalone_inference.py --cuda 0 --task_suites 'S1_d_trulyleq[2-8]_2H' > /dev/null 2>&1 &
python standalone_inference.py --cuda 0 --task_suites 'S1_d_trulyleq[2-8]_2I' > /dev/null 2>&1 &
python standalone_inference.py --cuda 0 --task_suites 'S1_d_trulyleq[2-8]_2J' > /dev/null 2>&1 &
python standalone_inference.py --cuda 0 --task_suites 'S1_d_trulyleq[2-8]_2K' > /dev/null 2>&1 &
python standalone_inference.py --cuda 0 --task_suites 'S1_d_trulyleq[2-8]_2L' > /dev/null 2>&1 &
python standalone_inference.py --cuda 1 --task_suites 'S1_d_trulyleq[2-8]_2M' > /dev/null 2>&1 &
python standalone_inference.py --cuda 1 --task_suites 'S1_d_trulyleq[2-8]_2h' > /dev/null 2>&1 &
python standalone_inference.py --cuda 1 --task_suites 'S1_d_trulyleq[2-8]_2i' > /dev/null 2>&1 &
python standalone_inference.py --cuda 1 --task_suites 'S1_d_trulyleq[2-8]_2j' > /dev/null 2>&1 &
python standalone_inference.py --cuda 2 --task_suites 'S1_d_trulyleq[2-8]_2k' > /dev/null 2>&1 &
python standalone_inference.py --cuda 2 --task_suites 'S1_d_trulyleq[2-8]_2l' > /dev/null 2>&1 &
python standalone_inference.py --cuda 2 --task_suites 'S1_d_trulyleq[2-8]_2m' > /dev/null 2>&1 &
python standalone_inference.py --cuda 2 --task_suites 'S1_d_trulyleq[2-8]_1H' > /dev/null 2>&1 &
python standalone_inference.py --cuda 3 --task_suites 'S1_d_trulyleq[2-8]_1I' > /dev/null 2>&1 &
python standalone_inference.py --cuda 3 --task_suites 'S1_d_trulyleq[2-8]_1J' > /dev/null 2>&1 &
python standalone_inference.py --cuda 3 --task_suites 'S1_d_trulyleq[2-8]_1K' > /dev/null 2>&1 &
python standalone_inference.py --cuda 3 --task_suites 'S1_d_trulyleq[2-8]_1L' > /dev/null 2>&1 &
python standalone_inference.py --cuda 3 --task_suites 'S1_d_trulyleq[2-8]_1M' > /dev/null 2>&1 &
wait
