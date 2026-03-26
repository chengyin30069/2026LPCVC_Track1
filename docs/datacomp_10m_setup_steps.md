DataComp 10M Setup For This Repository

Goal
- Keep benchmark assets: coco2017 and raw_datasets
- Build a new Track1 training dataset from DataComp with at least 10M image text pairs
- Rebuild embedding caches after dataset refresh

1) Confirm cleanup state
- Required to keep:
  - coco2017
  - raw_datasets
- Local training folders removed by assistant in this workspace:
  - dataset
  - dataset_image_only

2) Clone DataComp tools
- Commands:
  - cd raw_datasets
  - git clone https://github.com/mlfoundations/datacomp.git datacomp_repo
  - cd datacomp_repo

3) Install DataComp download dependencies
- Recommended inside your conda lpcv env:
  - conda activate lpcv
  - pip install -U pip
  - pip install huggingface_hub img2dataset cloudpathlib pyarrow webdataset fasttext-wheel fsspec pandas tqdm

4) Download CommonPool small scale (12.8M)
- Note: small scale data is large on disk. Reserve hundreds of GB.
- Use img2dataset files output so conversion is simple.
- Commands:
  - cd raw_datasets/datacomp_repo
  - python download_upstream.py --scale small --data_dir ../datacomp_small --output_format files --processes_count 16 --thread_count 128 --retries 2

5) Convert DataComp files to Track1 CSV + image layout
- This repository now includes:
  - prepare_datacomp_track1_dataset.py
- Build 10M pairs into dataset:
  - cd /home/lpcv2026/2026LPCVC_Track1
  - python prepare_datacomp_track1_dataset.py --source-roots raw_datasets/datacomp_small/shards --output-dir dataset --max-pairs 10000000 --link-mode symlink --image-prefix datacomp --overwrite

6) Rebuild image only dataset helper
- Command:
  - python prepare_image_only_distill_dataset.py --output-dir dataset_image_only --overwrite

7) Recompute caches after dataset refresh
- Student text embeddings:
  - python cache_text_embeddings.py --txt-list dataset/txt_list.csv --output artifacts/text_embeddings.npz

- Teacher text embeddings (SigLIP2):
  - python cache_text_embeddings.py --backend transformers --model-id google/siglip2-so400m-patch16-512 --txt-list dataset/txt_list.csv --output artifacts/text_embeddings_teacher_siglip2.npz

- Teacher image embeddings (SigLIP2):
  - python precompute_teacher_embeddings.py --backend transformers --model-id google/siglip2-so400m-patch16-512 --img-list dataset/img_list.csv --image-folder dataset/images --output artifacts/teacher_image_embeddings_siglip2.npz

- Optional hard negatives refresh:
  - python mine_hard_negatives.py --embeddings artifacts/text_embeddings.npz --output artifacts/hard_negatives.npz --csv-output artifacts/hard_negatives.csv --top-k 10

8) Sanity checks before training
- Pair count:
  - python -c "import json;print(json.load(open('dataset/summary.json'))['kept_pairs'])"
- CSV row counts:
  - python -c "import pandas as pd;print('img',len(pd.read_csv('dataset/img_list.csv')),'txt',len(pd.read_csv('dataset/txt_list.csv')))"
- Cache dimensions:
  - python -c "import numpy as np;print('text',np.load('artifacts/text_embeddings.npz')['embeddings'].shape,'teacher_text',np.load('artifacts/text_embeddings_teacher_siglip2.npz')['embeddings'].shape,'teacher_img',np.load('artifacts/teacher_image_embeddings_siglip2.npz')['embeddings'].shape)"

Notes
- If download bandwidth is limited, start with max_pairs 2M to validate the end to end flow first, then scale to 10M.
- If symlink mode is not desired, use --link-mode copy.
