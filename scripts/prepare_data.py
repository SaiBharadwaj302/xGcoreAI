from pathlib import Path
import shutil
import sys

LOCAL_PROCESSED = Path(__file__).resolve().parents[1] / "Data" / "processed"
REMOTE_PROCESSED = Path("/mount/src/xgcoreai/data/processed")

if not LOCAL_PROCESSED.exists():
    print(f"Local processed folder not found: {LOCAL_PROCESSED}")
    sys.exit(1)

REMOTE_PROCESSED.mkdir(parents=True, exist_ok=True)

def copy_file(name: str):
    src = LOCAL_PROCESSED / name
    if not src.exists():
        print(f"Skipping {name}; source does not exist")
        return
    dst = REMOTE_PROCESSED / name
    shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")

COPY_FILES = [
    "shots_final.csv",
    "player_stats_final.csv",
]

for file_name in COPY_FILES:
    copy_file(file_name)

print("Data prep complete.")
