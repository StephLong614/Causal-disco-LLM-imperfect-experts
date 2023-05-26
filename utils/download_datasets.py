import os

def download_datasets():
    DATA_URL = \
    [
        # Small
        "https://www.bnlearn.com/bnrepository/sachs/sachs.bif.gz",
        "https://www.bnlearn.com/bnrepository/asia/asia.bif.gz",
        "https://www.bnlearn.com/bnrepository/cancer/cancer.bif.gz",
        
        # Medium
        "https://www.bnlearn.com/bnrepository/alarm/alarm.bif.gz",
        "https://www.bnlearn.com/bnrepository/barley/barley.bif.gz",
        "https://www.bnlearn.com/bnrepository/child/child.bif.gz",
        "https://www.bnlearn.com/bnrepository/insurance/insurance.bif.gz",
        "https://www.bnlearn.com/bnrepository/mildew/mildew.bif.gz",
        "https://www.bnlearn.com/bnrepository/water/water.bif.gz",
        
        # Large
        "https://www.bnlearn.com/bnrepository/hailfinder/hailfinder.bif.gz",
        "https://www.bnlearn.com/bnrepository/hepar2/hepar2.bif.gz",
        "https://www.bnlearn.com/bnrepository/win95pts/win95pts.bif.gz"
    ]
    DATA_URL = {os.path.basename(u).replace(".bif.gz", ""): u for u in DATA_URL}

    os.makedirs("_raw_bayesian_nets", exist_ok=True)
    for name, u in DATA_URL.items():
        os.system(f"wget {u} -q -O _raw_bayesian_nets/{os.path.basename(u)} && gunzip -f _raw_bayesian_nets/{os.path.basename(u)} && echo '{name}: success' || echo '{name}': FAILED!")

    SUPPORTED_DATASETS = {name: f"_raw_bayesian_nets/{name}.bif" for name in DATA_URL}
