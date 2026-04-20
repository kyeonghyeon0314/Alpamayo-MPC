# check_labels.py
import h5py, numpy as np, glob

paths = sorted(glob.glob("alpamayo_dataset/data/prepare/train/*.h5"))[:10]
for path in paths:
    with h5py.File(path, "r") as f:
        if "labels/mpc_weights" not in f:
            print(f"{path.split('/')[-1]}: NO LABEL")
            continue
        w    = f["labels/mpc_weights"][:]
        ade  = float(f["labels/ade"][()])
        valid = bool(f["labels/valid"][()])
        print(f"{path.split('/')[-1]}  ADE={ade:.3f}m  valid={valid}")
        print(f"  long={w[0]:.2f}  lat={w[1]:.2f}  hdg={w[2]:.2f}  "
              f"acc={w[3]:.3f}  steer_r={w[4]:.2f}  acc_r={w[5]:.2f}")
