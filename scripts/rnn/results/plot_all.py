import sys, os

task_suite = "S2a_d_trulyleq[2-8]_A"
output_dir = f"/home/yingshac/workspace/tmlr/scripts/rnn/output/{task_suite}/"
figure_dir = f"./{task_suite}/"

# create figure_dir if not exist
os.makedirs(figure_dir, exist_ok=True)

os.system("python plot.py --task_suite {} --output_dir {} --figure_dir {}".format(
    task_suite,
    output_dir,
    figure_dir,
))

os.system("python plot_best_same_d_val_epc.py --task_suite {} --output_dir {} --figure_dir {}".format(
    task_suite,
    output_dir,
    figure_dir,
))