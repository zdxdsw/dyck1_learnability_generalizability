import os

TASK_SUITE = [
    "S1_d_trulyleq[2-8]_4b",
    "S1_d_trulyleq[2-8]_4c",
    "S1_d_trulyleq[2-8]_2b",
    "S1_d_trulyleq[2-8]_2c", 
    "S1_d_trulyleq[2-8]_2D", 
    "S1_d_trulyleq[2-8]_2E",
    "S1_d_trulyleq[2-8]_2F",
    "S1_d_trulyleq[2-8]_4B",
    "S1_d_[2-8]_4B",
    "S1_d_[2-8]_2G",
]
model = "transformer"

if isinstance(TASK_SUITE, str): TASK_SUITE = [TASK_SUITE]
for task_suite in TASK_SUITE:
    output_dir = f"/home/yingshac/workspace/tmlr/scripts/experiments/output_{model}/{task_suite}/"
    figure_dir = f"../results_{model}/{task_suite}/"

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