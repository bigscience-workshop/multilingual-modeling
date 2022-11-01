from huggingface_hub import Repository
from huggingface_hub import create_repo, move_repo

import argparse
import subprocess
import pathlib
parser = argparse.ArgumentParser()
# parser.add_argument("--local_dir", type=str, required=True)
# parser.add_argument("--remote_dir", type=str, required=True)
# parser.add_argument("--parent_folder", type=str, required=True, 
#                     help="the parent directory that contains all the model repos. "
#                          "we will iterate through the model repo (`local_dir`) "
#                          "and create the `remote_dir` on HF hub.")
parser.add_argument("--commit_message", type=str, default="Push to HF hub")
args = parser.parse_args()

def add_to_gitignore(local_dir, f, strf):
    if f.is_file():
        subprocess.run(f"cd {local_dir} && echo '{strf}' >> .gitignore", shell=True)
    else:
        subprocess.run(f"cd {local_dir} && echo '{strf}/' >> .gitignore", shell=True)


def upload_to_hub(local_dir, remote_dir, commit_message, model_type):
    # git init
    subprocess.run(f"cd {local_dir} && git init", shell=True)

    # git reset origin
    subprocess.run(f"cd {local_dir} && git remote set-url origin https://huggingface.co/{remote_dir}", shell=True)

    # model-specific ignored:
    subprocess.run(f"cd {local_dir} && touch .gitignore", shell=True)
    subprocess.run(f"cd {local_dir} && > .gitignore", shell=True)


    if model_type == "full-model":
        files_to_keep = [".git", ".gitignore", ".gitattributes",
                         "config.json", "optimizer.pt", "rng_state.pth", "scheduler.pt", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "trainer_state.json", "training_args.bin"]
        for f in pathlib.Path(local_dir).glob("*"):
            # get relative path
            strf = str(f.relative_to(pathlib.Path(local_dir)))
            if strf not in files_to_keep and "pytorch_model" not in strf:
                add_to_gitignore(local_dir, f, strf)

    elif model_type == "adapter":
        files_to_keep = [".git", ".gitignore", ".gitattributes"]
        for f in pathlib.Path(local_dir).glob("*"):
            # get relative path
            strf = str(f.relative_to(pathlib.Path(local_dir)))
            if strf not in files_to_keep and not strf.startswith("oscar"): #FIXME: replace dataset-specific keyword with sth else.
                add_to_gitignore(local_dir, f, strf)
    
    elif model_type == "mask":
        files_to_keep = [".git", ".gitignore", ".gitattributes",
                         "config.json", "optimizer.pt", "rng_state.pth", "scheduler.pt", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "trainer_state.json", "training_args.bin",
                         # sft:
                         "pytorch_diff.bin",
                         ]
        for f in pathlib.Path(local_dir).glob("*"):
            # get relative path
            strf = str(f.relative_to(pathlib.Path(local_dir)))
            if strf not in files_to_keep and "pytorch_model" not in strf: #FIXME: replace dataset-specific keyword with sth else.
                add_to_gitignore(local_dir, f, strf)

    else:
        # ignore checkpoints, and pilot experiments (eval downstream)
        files_to_keep = [".git", ".gitignore", ".gitattributes"]
        subprocess.run(f"cd {args.local_dir} && echo 'checkpoint-*/' >> .gitignore", shell=True)
        subprocess.run(f"cd {args.local_dir} && echo '*/pilot_*/' >> .gitignore", shell=True)
        subprocess.run(f"cd {args.local_dir} && echo 'pilot_*/' >> .gitignore", shell=True)
    
    # create remote repo
    try:
        create_repo(remote_dir)
    except Exception as e:
        print("=== Encounter following error when creating remote repo ===")
        print(e)
        print("===============================")

    # add remote origin
    subprocess.run(f"cd {local_dir} && git remote add origin https://huggingface.co/{remote_dir}", shell=True)

    # git pull remote origin
    subprocess.run(f"cd {local_dir} && git pull origin main", shell=True)

    # git add 
    repo = Repository(
        local_dir=local_dir,
        clone_from=remote_dir
    )
    repo.git_add(auto_lfs_track=True)
    subprocess.run(f"cd {local_dir} && git status", shell=True)

    # git commit
    subprocess.run(f"cd {local_dir} && git commit -m '{commit_message}'", shell=True)

    # git push to main branch on remote
    branch_name = subprocess.run(f"cd {local_dir} && git symbolic-ref --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(f"cd {local_dir} && git push -f origin {branch_name}:main", shell=True)

def upload_ckpt_to_hub(local_dir, checkpoint_dir, remote_dir, commit_message, model_type):
    # create checkpoint branch
    subprocess.run(f"cd {local_dir} && git switch --orphan {checkpoint_dir}", shell=True)

    # git add
    repo = Repository(
        local_dir=local_dir,
        clone_from=remote_dir
    )
    repo.git_add(checkpoint_dir, auto_lfs_track=True)
    subprocess.run(f"cd {local_dir} && git status", shell=True)

     # git commit
    subprocess.run(f"cd {local_dir} && git commit -m '{commit_message}'", shell=True)

    # git push to checkpoint branch on remote
    branch_name = subprocess.run(f"cd {local_dir} && git symbolic-ref --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(f"cd {local_dir} && git push origin {branch_name}:{branch_name}", shell=True)

def rm_one(local_dir, remote_dir, rm_file, commit_message="rm --cached"):
    """
    Delete one file / folder from remote repo while keeping local repo intact.
    if rm_file = "*", then we remove everything from the remote repo.

    E.g.:
    local_dir = "/users/zyong2/data/zyong2/bigscience/data/processed/024/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original"
    remote_dir = f"bs-la/{pathlib.Path(local_dir).name}".replace("+", "_") 
    rm_file = "*"
    rm_one(local_dir, remote_dir, rm_file)
    """

    subprocess.run(f"cd {local_dir} && git rm -r --cached '{rm_file}'", shell=True)
    subprocess.run(f"cd {local_dir} && git commit -m '{commit_message} {rm_file}'", shell=True)
    branch_name = subprocess.run(f"cd {local_dir} && git symbolic-ref --short HEAD", shell=True, capture_output=True, text=True).stdout.strip()
    subprocess.run(f"cd {local_dir} && git push origin {branch_name}:main", shell=True)

# #### main ####
# for folder in pathlib.Path(args.parent_folder).glob("*"):
#     # skip empty directory:
#     if any(folder.iterdir()) is False:
#         continue

folder = "/home/zhengxinyong/outputs/bloom-7b1_de_continual-pretrain_100000samples_-1vocab_original"
local_dir = str(folder)
remote_dir = f"bs-la/{pathlib.Path(folder).name}".replace("+", "_")  # affects "pfeiffer+inv"

if "continual" in local_dir or "bitfit" in local_dir:
    # rm_file = "*"
    # rm_one(local_dir, remote_dir, rm_file)
    upload_to_hub(local_dir, remote_dir, commit_message=args.commit_message, model_type="full-model")
    
    # upload_ckpt_to_hub(local_dir, "checkpoint-5000", remote_dir, commit_message=args.commit_message, model_type="full-model")
    # upload_ckpt_to_hub(local_dir, "checkpoint-10000", remote_dir, commit_message=args.commit_message, model_type="full-model")
    # upload_ckpt_to_hub(local_dir, "checkpoint-15000", remote_dir, commit_message=args.commit_message, model_type="full-model")
    # upload_ckpt_to_hub(local_dir, "checkpoint-20000", remote_dir, commit_message=args.commit_message, model_type="full-model")
    # upload_ckpt_to_hub(local_dir, "checkpoint-25000", remote_dir, commit_message=args.commit_message, model_type="full-model")
elif "pfeiffer" in local_dir or "lora" in local_dir or "ia3" in local_dir:
    upload_to_hub(local_dir, remote_dir, commit_message=args.commit_message, model_type="adapter")
elif "fish" in local_dir or "sft" in local_dir:
    upload_to_hub(local_dir, remote_dir, commit_message=args.commit_message, model_type="mask")
else:
    print("❌ skip", local_dir)



