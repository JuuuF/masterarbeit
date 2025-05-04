import os, pickle
from ma_darts.ai.callbacks import HistoryPlotter

raw_hists = [
    pickle.load(open(f"dump/{f}", "rb"))
    for f in sorted(
        os.listdir("dump")
    )
    if f.startswith("training_history_long") and f.endswith(".pkl")
]

total_hist = raw_hists[0]
for t_ in raw_hists[1:]:
    for k, v in t_.items():
        for k_, v_ in v.items():
            total_hist[k][k_] += v_

hp = HistoryPlotter("dump/test.png", log_scale=True, dark_mode=False)
hp.train_logs = total_hist["train_logs"]
hp.val_logs = total_hist["val_logs"]

hp.plot_losses()
