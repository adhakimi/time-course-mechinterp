import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, snapshot_download

# Directory where you want to save the downloads
download_dir = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di35pop/revisions_temp"

# Create the directory if it doesn't exist
os.makedirs(download_dir, exist_ok=True)

api = HfApi()
refs = api.list_repo_refs("EleutherAI/pythia-6.9b")

# Instead of filtering, download all branches.
branches_to_download = refs.branches

# Optionally, if you also want to download tags, uncomment the following line:
# revisions_to_download = refs.branches + refs.tags
# Otherwise, use branches_to_download for only branch revisions.

# Define a function to download a single revision
def download_revision(revision):
    name = revision.name
    print(f"Downloading {name}...")
    snapshot_download(repo_id="EleutherAI/pythia-6.9b", revision=name, cache_dir=download_dir)
    return name

# Use ThreadPoolExecutor to download revisions in parallel
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = [executor.submit(download_revision, rev) for rev in branches_to_download]
    for future in as_completed(futures):
        try:
            result = future.result()
            if result:
                print(f"Completed: {result}")
        except Exception as exc:
            print(f"Generated an exception: {exc}")

print(f"Total revisions processed: {len(branches_to_download)}")


# import torch
# from transformer_lens import HookedTransformer

# def load_models(model_name, device='cuda:0', checkpoint=""):
#     model = HookedTransformer.from_pretrained_no_processing(
#         model_name,
#         dtype=torch.bfloat16,
#         center_unembed=True,
#         center_writing_weights=True,
#         #fold_ln=True,
#         device=device,
#         trust_remote_code=True,
#         cache_dir="/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di35pop/revisions_temp",
#         checkpoint_value=checkpoint,
#     )

#     return model


# # Load the models
# model_name = "EleutherAI/pythia-6.9b"

# base_model = load_models(model_name, device='cuda:0', checkpoint='main')


# import os
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from huggingface_hub import HfApi, snapshot_download

# # Directory where you want to save the downloads
# download_dir = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di35pop/revisions_temp"

# # Create the directory if it doesn't exist
# os.makedirs(download_dir, exist_ok=True)

# # Define the filtering parameter
# start_step = 5000

# api = HfApi()
# refs = api.list_repo_refs("EleutherAI/pythia-6.9b")

# # Filter and sort branches for the Pythia model:
# filtered_and_sorted_branches = sorted(
#     [branch for branch in refs.branches
#      if branch.name not in ('main', 'nitro')
#      and int(branch.name.split('step')[1]) >= start_step
#      and (int(branch.name.split('step')[1]) == 5000 or int(branch.name.split('step')[1]) % 10000 == 0)],
#     key=lambda x: int(x.name.split('step')[1])
# )

# # Slice the list to get the last 5 revisions (the ones immediately preceding main)
# #last_five_branches = filtered_and_sorted_branches[-5:]

# filtered_and_sorted_branches.append("step143000")

# # Define a function to download a single branch
# def download_branch(branch):
#     name = branch.name
#     step_number = int(name.split('step')[1])
#     print(f"Downloading {name}...")
#     # Create a subdirectory for this branch
#     #local_dir = os.path.join(download_dir, name)
#     snapshot_download(repo_id="EleutherAI/pythia-6.9b", revision=name, cache_dir=download_dir)
#     return name

# # Use ThreadPoolExecutor to download branches in parallel
# with ThreadPoolExecutor(max_workers=16) as executor:
#     futures = [executor.submit(download_branch, branch) for branch in filtered_and_sorted_branches]
#     for future in as_completed(futures):
#         try:
#             result = future.result()
#             if result:
#                 print(f"Completed: {result}")
#         except Exception as exc:
#             print(f"Generated an exception: {exc}")

# print(f"Total branches processed: {len(filtered_and_sorted_branches)}")


# from huggingface_hub import HfApi, snapshot_download

# # Set the base download directory (change to your desired path)
# download_base = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di35pop/revisions_temp"

# api = HfApi()
# refs = api.list_repo_refs("EleutherAI/pythia-6.9b")

# # Filter branches: include step5000 and every nonzero multiple of 10000 (exclude step0)
# branches_to_download = []
# for branch in refs.branches:
#     if branch.name.startswith("step"):
#         try:
#             step = int(branch.name.replace("step", ""))
#             if (step == 5000) or (step != 0 and step % 10000 == 0):
#                 branches_to_download.append(branch.name)
#         except ValueError:
#             continue

# print("Branches to download:", branches_to_download)

# # Download each branch revision into its own subdirectory
# for branch_name in branches_to_download:
#     local_dir = f"{download_base}/{branch_name}"
#     print(f"Downloading branch '{branch_name}' into '{local_dir}'...")
#     snapshot_download(repo_id="EleutherAI/pythia-6.9b", revision=branch_name, local_dir=local_dir)
    
    

# # Directory where you want to save the downloads
# download_dir = "/dss/dssfs04/lwp-dss-0002/pn25ho/pn25ho-dss-0001/di35pop/revisions_temp"

# # Create the directory if it doesn't exist
# os.makedirs(download_dir, exist_ok=True)

# # # Filter and sort branches
# # filtered_and_sorted_branches = sorted(
# #     [branch for branch in refs.branches 
# #      if branch.name not in ('main', 'nitro') 
# #      and start_step <= int(branch.name.split('step')[1].split('-')[0]) <= max_step 
# #      and int(branch.name.split('step')[1].split('-')[0]) % step_increment == 0],
# #     key=lambda x: int(x.name.split('step')[1].split('-')[0])
# # )

# # # Define a function to download a single branch
# def download_branch(branch):
#     name = branch.name
#     step_number = int(name.split('step')[1].split('-')[0])

#     if start_step <= step_number <= max_step and step_number % step_increment == 0:
#         print(f"Downloading {name}...")
#         snapshot_download(repo_id="allenai/OLMo-7B-0424-hf", revision=name, cache_dir=download_dir)
#         return name

# # # for branch in filtered_and_sorted_branches:
# # #     download_branch(branch)

# # # Use ThreadPoolExecutor to download branches in parallel
# # with ThreadPoolExecutor(max_workers=32) as executor:  # Adjust max_workers based on your system's capacity
# #     futures = [executor.submit(download_branch, branch) for branch in filtered_and_sorted_branches]
# #     for future in as_completed(futures):
# #         try:
# #             result = future.result()
# #             if result:
# #                 print(f"Completed: {result}")
# #         except Exception as exc:
# #             print(f"Generated an exception: {exc}")

# # print(f"Total branches processed: {len(futures)}")

# filtered_and_sorted_branches = sorted(
#     [branch for branch in refs.branches
#      if branch.name not in ('main', 'nitro')
#      and int(branch.name.split('step')[1].split('-')[0]) >= start_step
#      and int(branch.name.split('step')[1].split('-')[0]) % step_increment == 0],
#     key=lambda x: int(x.name.split('step')[1].split('-')[0])
# )

# # Slice the list to get the last 5 revisions (the ones immediately preceding main)
# last_five_branches = filtered_and_sorted_branches[-5:]

# # Download each branch sequentially
# for branch in last_five_branches:
#     print(f"Downloading {branch.name}...")
#     download_branch(branch)
#     print(f"Completed: {branch.name}")

# print(f"Total branches processed: {len(last_five_branches)}")
#                                                                                                                                                                                         38,1          Bot
