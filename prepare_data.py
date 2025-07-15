# prepare_data.py
import os
import pickle

def make_text2cad_records(split_dir):
    """
    Creates records for the Text2CAD dataset
    """
    cad_dir = os.path.join(split_dir, 'cadquery')
    mesh_dir = os.path.join(split_dir, 'meshes')

    py_files = sorted(f for f in os.listdir(cad_dir) if f.endswith('.py'))
    records = []

    for py_file in py_files:
        uid = py_file.replace('.py', '')
        stl_file = f"{uid}.stl"
        if not os.path.exists(os.path.join(mesh_dir, stl_file)):
            print(f"Skipping {uid} — STL file not found.")
            continue

        record = {
            "uid": uid,
            "description": "Generate cadquery code",
            "py_path": os.path.join('cadquery', py_file),
            "mesh_path": os.path.join('meshes', stl_file)
        }
        records.append(record)

    return records

def dump(records, out_path):
    with open(out_path, 'wb') as f:
        pickle.dump(records, f)
    print(f"Wrote {len(records)} records to {out_path}")


if __name__ == "__main__":
    root = "./data/text2cad"

    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            print(f"Skipping missing split: {split_dir}")
            continue
        records = make_text2cad_records(split_dir)
        out_path = os.path.join(root, f"{split}.pkl")
        dump(records, out_path)
