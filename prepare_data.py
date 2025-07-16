# prepare_data.py
import os
import pickle

def build_cad_recode_pkl(root_dir, split):
    """
    cad-recode-v1.5 has:
      root_dir/train/batch_XX/*.py
      root_dir/val/*.py
    """
    split_dir = os.path.join(root_dir, split)
    records = []
    for dirpath, _, filenames in os.walk(split_dir):
        rel_dir = os.path.relpath(dirpath, root_dir)
        for fname in filenames:
            uid = fname[:-4]
            records.append({
                "uid": uid,
                # "mesh_path": os.path.join(rel_dir, fname),
                "py_path":   os.path.join(rel_dir, uid + ".py")
            })
    out_path = os.path.join(root_dir, f"{split}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(f"Wrote {len(records)} records to {out_path}")

def build_text2cad_pkl(root_dir):
    """
    text2cad has train/val/test splits each with:
      cadquery/*.py
      meshes/*.stl
    """
    for split in ("train", "val", "test"):
        split_dir = os.path.join(root_dir, split)
        cad_dir = os.path.join(split_dir, "cadquery")
        mesh_dir = os.path.join(split_dir, "meshes")
        if not os.path.isdir(split_dir):
            print(f"Skipping missing {split_dir}")
            continue

        records = []
        for py_file in sorted(os.listdir(cad_dir)):
            if not py_file.endswith(".py"):
                continue
            uid = py_file[:-3]
            stl_file = uid + ".stl"
            stl_path = os.path.join(mesh_dir, stl_file)
            if not os.path.exists(stl_path):
                print(f"  warning: missing mesh for {uid}")
                continue
            records.append({
                "uid": uid,
                "description": "Generate cadquery code",
                "py_path": os.path.join(split, "cadquery", py_file),
                "mesh_path": os.path.join(split, "meshes", stl_file)
            })

        out_path = os.path.join(root_dir, f"{split}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(records, f)
        print(f"Wrote {len(records)} records to {out_path}")

def build_mesh_only_pkl(root_dir):
    """
    For fusion360_test_mesh/ and deepcad_test_mesh/ (just .stl files).
    """
    records = []
    for fname in sorted(os.listdir(root_dir)):
        if fname.endswith(".stl"):
            records.append({
                "uid": fname[:-4],
                "mesh_path": fname
            })
    out_path = os.path.join(root_dir, "train.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(records, f)
    print(f"Wrote {len(records)} records to {out_path}")

if __name__ == "__main__":
    data_root = "./data"

    # cad-recode
    cad_root = os.path.join(data_root, "cad-recode-v1.5")
    build_cad_recode_pkl(cad_root, "train")
    build_cad_recode_pkl(cad_root, "val")

    # text2cad
    text_root = os.path.join(data_root, "text2cad")
    build_text2cad_pkl(text_root)

    # fusion360 & deepcad
    build_mesh_only_pkl(os.path.join(data_root, "fusion360_test_mesh"))
    build_mesh_only_pkl(os.path.join(data_root, "deepcad_test_mesh"))

    print("All pickle files created.")