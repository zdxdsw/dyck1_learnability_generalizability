# CUDA_VISIBLE_DEVICES="3" accelerate launch --main_process_port 29500 --num_processes 1 trainer.py --date 0622_011009 --task S3_d_trulyleq8 --output_dir '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S3_d_trulyleq[2-8]_2G/S3_d_trulyleq8' 2> '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S3_d_trulyleq[2-8]_2G/S3_d_trulyleq8/0622_011009/stderr.txt' | tee '/home/yingshac/workspace/tmlr/scripts/experiments/output_transformer/S3_d_trulyleq[2-8]_2G/S3_d_trulyleq8/0622_011009/terminal.txt' >/dev/null &
# wait

python standalone_inference.py --cuda 0 --task_suites 'S2_d_trulyleq[2-8]_2E' > /dev/null 2>&1 &
python standalone_inference.py --cuda 0 --task_suites 'S2_d_trulyleq[2-8]_2F' > /dev/null 2>&1 &
python standalone_inference.py --cuda 1 --task_suites 'S2_d_trulyleq[2-8]_2G' > /dev/null 2>&1 &
python standalone_inference.py --cuda 2 --task_suites 'S3_d_trulyleq[2-8]_2E' > /dev/null 2>&1 &
python standalone_inference.py --cuda 3 --task_suites 'S3_d_trulyleq[2-8]_2F' > /dev/null 2>&1 &
python standalone_inference.py --cuda 3 --task_suites 'S3_d_trulyleq[2-8]_2G' > /dev/null 2>&1 &
wait
