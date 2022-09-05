from huggingface_hub import Repository
from huggingface_hub import create_repo

import argparse
import subprocess
parser = argparse.ArgumentParser()
# parser.add_argument("--local_dir", type=str, required=True)
# parser.add_argument("--remote_dir", type=str, required=True)
parser.add_argument("--commit_message", type=str, default="Push to HF hub")
args = parser.parse_args()

args.local_dir = "/users/zyong2/data/zyong2/bigscience/data/processed/024/bloom-350m_az_continual-pretrain-reinit_100000samples_-1vocab_original"
args.remote_dir = "bs-la/bloom-350m_az_continual-pretrain-reinit_100000samples_-1vocab_original"

# git init
subprocess.run(f"cd {args.local_dir} && git init", shell=True)

# ignore checkpoints, and pilot experiments (eval downstream)
subprocess.run(f"cd {args.local_dir} && echo 'checkpoint-*/' >> .gitignore", shell=True)
subprocess.run(f"cd {args.local_dir} && touch .gitignore", shell=True)
subprocess.run(f"cd {args.local_dir} && echo '*/pilot_*/' >> .gitignore", shell=True)
subprocess.run(f"cd {args.local_dir} && echo 'pilot_*/' >> .gitignore", shell=True)

# create remote repo
try:
    create_repo(args.remote_dir)
except Exception as e:
    print("=== Encounter following error when creating remote repo ===")
    print(e)
    print("===============================")

# add remote origin
subprocess.run(f"cd {args.local_dir} && git remote add origin https://huggingface.co/{args.remote_dir}", shell=True)

# git pull remote origin
subprocess.run(f"cd {args.local_dir} && git pull origin main", shell=True)

# git add 
repo = Repository(
    local_dir=args.local_dir,
    clone_from=args.remote_dir
)
repo.git_add(auto_lfs_track=True)

# git commit
subprocess.run(f"cd {args.local_dir} && git commit -m '{args.commit_message}'", shell=True)

# git push to main branch on remote
branch_name = subprocess.run(f"cd {args.local_dir} && git symbolic-ref --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()
subprocess.run(f"cd {args.local_dir} && git push origin {branch_name}:main", shell=True)
