# prepare_data.py

import os
import pickle


def build_cad_recode_pkl(dataset_dir: str, split: str):
    """
    Scan cad-recode-v1.5/{split} and write {split}.pkl with entries:
      - mesh_path (for .stl files)
      - py_path   (for .py scripts)
      - uid
    """
    print(f"Building cad-recode-v1.5 {split}.pkl")
    data = []
    split_dir = os.path.join(dataset_dir, split)

    if split == "train":
        for batch in sorted(os.listdir(split_dir)):
            batch_path = os.path.join(split_dir, batch)
            if not os.path.isdir(batch_path):
                continue
            for fname in os.listdir(batch_path):
                if not fname.endswith(".stl"):
                    continue
                uid = fname[:-4]
                data.append({
                    "mesh_path": os.path.join(split, batch, fname),
                    "py_path":   os.path.join(split, batch, uid + ".py"),
                    "uid": uid
                })

    elif split == "val":
        for fname in os.listdir(split_dir):
            if not fname.endswith(".stl"):
                continue
            uid = fname[:-4]
            data.append({
                "mesh_path": os.path.join(split, fname),
                "py_path":   os.path.join(split, uid + ".py"),
                "uid": uid
            })

    out_path = os.path.join(dataset_dir, f"{split}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {len(data)} records to {out_path}")


def build_mesh_list_pkl(dataset_dir: str):
    """
    For a folder of standalone .stl files, write train.pkl listing each mesh_path and uid.
    """
    print(f"Building mesh list for {dataset_dir}")
    data = []
    for fname in os.listdir(dataset_dir):
        if fname.endswith(".stl"):
            data.append({
                "mesh_path": fname,
                "uid": fname[:-4]
            })
    out_path = os.path.join(dataset_dir, "train.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {len(data)} records to {out_path}")

def build_cadquery_pkl(dataset_dir: str, split: str):
    """
    Scan CadQuery/{split} and write {split}.pkl with entries:
      - mesh_path (for .stl files)
      - py_path   (for .py scripts)
      - uid
    """
    print(f"Building CadQuery {split}.pkl")
    data = []
    split_dir = os.path.join(dataset_dir, split)

    if split == "train":
        for batch in sorted(os.listdir(split_dir)):
            batch_path = os.path.join(split_dir, batch)
            if not os.path.isdir(batch_path):
                continue
            for fname in os.listdir(batch_path):
                if not fname.endswith(".stl"):
                    continue
                uid = fname[:-4]
                data.append({
                    "mesh_path": os.path.join(split, batch, fname),
                    "py_path":   os.path.join(split, batch, uid + ".py"),
                    "uid": uid
                })

    elif split == "val":
        for fname in os.listdir(split_dir):
            if not fname.endswith(".stl"):
                continue
            uid = fname[:-4]
            data.append({
                "mesh_path": os.path.join(split, fname),
                "py_path":   os.path.join(split, uid + ".py"),
                "uid": uid
            })

    out_path = os.path.join(dataset_dir, f"{split}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {len(data)} records to {out_path}")

if __name__ == "__main__":
    root = "./data"

    build_cad_recode_pkl(os.path.join(root, "cad-recode-v1.5"), "train")
    build_cad_recode_pkl(os.path.join(root, "cad-recode-v1.5"), "val")
    build_cadquery_pkl(os.path.join(root, "CadQuery"), "train")
    build_cadquery_pkl(os.path.join(root, "CadQuery"), "val")
    # text2cad already provides its own train/val/test .pkl, so we skip it here

    build_mesh_list_pkl(os.path.join(root, "fusion360_test_mesh"))
    build_mesh_list_pkl(os.path.join(root, "deepcad_test_mesh"))

    print("All pickle files created.")
