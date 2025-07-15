# prepare_data.py (tailored for raw git clone structure)

import os
import pickle


def build_cad_recode_pkl(dataset_dir, split, out_path):
    print(f"Building CAD-Recode PKL for split: {split}")
    data = []

    if split == "train":
        split_dir = os.path.join(dataset_dir, "train")
        for folder in sorted(os.listdir(split_dir)):
            batch_path = os.path.join(split_dir, folder)
            if os.path.isdir(batch_path):
                for fname in os.listdir(batch_path):
                    if fname.endswith(".stl"):
                        uid = fname.replace(".stl", "")
                        entry = {
                            "mesh_path": os.path.join("train", folder, fname),
                            "py_path": os.path.join("train", folder, uid + ".py"),
                            "uid": uid
                        }
                        data.append(entry)

    elif split == "val":
        split_dir = os.path.join(dataset_dir, "val")
        for fname in os.listdir(split_dir):
            if fname.endswith(".py"):
                uid = fname.replace(".py", "")
                entry = {
                    "py_path": os.path.join("val", fname),
                    "uid": uid
                    # No mesh_path available
                }
                data.append(entry)

    os.makedirs(dataset_dir, exist_ok=True)
    with open(os.path.join(dataset_dir, f"{split}.pkl"), "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {len(data)} records to {split}.pkl")


def build_mesh_list_pkl(dataset_dir):
    print(f"Indexing STL files in: {dataset_dir}")
    data = []
    for fname in os.listdir(dataset_dir):
        if fname.endswith(".stl"):
            data.append({
                "mesh_path": fname,
                "uid": fname.replace(".stl", "")
            })
    with open(os.path.join(dataset_dir, "train.pkl"), "wb") as f:
        pickle.dump(data, f)
    print(f"Wrote {len(data)} records to {dataset_dir}/train.pkl")


if __name__ == "__main__":
    root = "./data"
    build_cad_recode_pkl(os.path.join(root, "cad-recode-v1.5"), "train", os.path.join(root, "cad-recode-v1.5", "train.pkl"))
    build_cad_recode_pkl(os.path.join(root, "cad-recode-v1.5"), "val", os.path.join(root, "cad-recode-v1.5", "val.pkl"))

    build_mesh_list_pkl(os.path.join(root, "deepcad_test_mesh"))
    build_mesh_list_pkl(os.path.join(root, "fusion360_test_mesh"))

    print("All pickle files created.")
