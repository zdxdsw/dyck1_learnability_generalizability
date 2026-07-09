# CUDA_VISIBLE_DEVICES="1" accelerate launch --main_process_port 29517 --num_processes 1 trainer.py --date 0626_130513 --task S2_d_trulyleq2 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_rnn/S2_d_trulyleq[2-8]_F/S2_d_trulyleq2' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_rnn/S2_d_trulyleq[2-8]_F/S2_d_trulyleq2/0626_130513/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_rnn/S2_d_trulyleq[2-8]_F/S2_d_trulyleq2/0626_130513/terminal.txt'  >/dev/null &
# CUDA_VISIBLE_DEVICES="3" accelerate launch --main_process_port 29523 --num_processes 1 trainer.py --date 0626_192455 --task S1_d_6 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_rnn/S1_d_[2-8]_F/S1_d_6' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_rnn/S1_d_[2-8]_F/S1_d_6/0626_192455/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_rnn/S1_d_[2-8]_F/S1_d_6/0626_192455/terminal.txt'  >/dev/null &
# wait

# python standalone_inference.py --cuda 1 --task_suites 'S2_d_trulyleq[2-8]_B' > /dev/null 2>&1 &
# wait

python stress_test_inference.py --cuda 1 --task_suites 'S1_d_trulyleq[2-8]_1A' 'S1_d_trulyleq[2-8]_1B' 'S1_d_trulyleq[2-8]_1C' > /dev/null 2>&1 &
python stress_test_inference.py --cuda 1 --task_suites 'S1_d_trulyleq[2-8]_1D' 'S1_d_trulyleq[2-8]_1E' 'S1_d_trulyleq[2-8]_1F' > /dev/null 2>&1 &
python stress_test_inference.py --cuda 2 --task_suites 'S1_d_trulyleq[2-8]_1G' 'S1_d_trulyleq[2-8]_2a' 'S1_d_trulyleq[2-8]_2b' > /dev/null 2>&1 &
python stress_test_inference.py --cuda 2 --task_suites 'S1_d_trulyleq[2-8]_2c' 'S1_d_trulyleq[2-8]_2d' 'S1_d_trulyleq[2-8]_2e' > /dev/null 2>&1 &
python stress_test_inference.py --cuda 3 --task_suites 'S1_d_trulyleq[2-8]_2f' 'S1_d_trulyleq[2-8]_2g' 'S1_d_trulyleq[2-8]_1a' > /dev/null 2>&1 &
python stress_test_inference.py --cuda 3 --task_suites 'S1_d_trulyleq[2-8]_1b' 'S1_d_trulyleq[2-8]_1c' 'S1_d_trulyleq[2-8]_1d' > /dev/null 2>&1 &
python stress_test_inference.py --cuda 0 --task_suites 'S1_d_trulyleq[2-8]_1e' 'S1_d_trulyleq[2-8]_1f' 'S1_d_trulyleq[2-8]_1g' > /dev/null 2>&1 &
wait

# python run.py --port 29510 --cuda 2 --task S2_d_trulyleq3 > /dev/null 2>&1 &
