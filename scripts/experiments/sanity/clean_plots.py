import os

MODEL = "rnn"
PLOT_DIR = f"../single_metric_plot_{MODEL}"

# loop through all metrics, and delete all files with "ACDEF" in filename
for metric in os.listdir(PLOT_DIR):
    plot_for_metric_dir = os.path.join(PLOT_DIR, metric)
    if not os.path.isdir(plot_for_metric_dir): continue
    for file in os.listdir(plot_for_metric_dir):
        if "ACDEF" in file:
            # print(f"Deleting file: {os.path.join(plot_for_metric_dir, file)}")
            ### BE CAREFUL!
            os.remove(os.path.join(plot_for_metric_dir, file))