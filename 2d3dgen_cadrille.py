import argparse
import os
import numpy as np
import trimesh

import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoProcessor

import re

import cadquery as cq
from cadquery.vis import show
import open3d as o3d

from pytorch3d.ops import sample_farthest_points

import evaluate

from dataset import Text2CADDataset, CadRecodeDataset
from cadrille import Cadrille, collate

def load_model_and_processor(model_name, mode): # can also use path to checkpoint
    """
    Load the pretrained Cadrille model and processor from Hugging Face.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model and processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28,
        use_fast=True
    )
    model = Cadrille.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16, # or auto
        attn_implementation='sdpa',
        device_map='auto'
    )
    return processor, model

# generate a point cloud from a mesh file
def process_mesh_to_point_cloud(mesh_path, n_points, n_pre_points=8192, mode='pc'):
    mesh = trimesh.load_mesh(mesh_path, force='mesh')
    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    return np.asarray(vertices[ids]) # this is the point cloud itself

def view_point_cloud(point_cloud_data, mesh_path):
    # create an o3d PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    o3d.visualization.draw_geometries([pcd],
                                      window_name=f"{len(point_cloud_data)} points - {mesh_path}",
                                      width=1280,
                                      height=720
                                      )

def generate_cadquery_script_from_3d(processor, model, token_size, mesh_path, output_path, n_points):
    print(f"Generating Cadquery script for mesh: {mesh_path}")

    mesh = process_mesh_to_point_cloud(mesh_path, n_points)

    example = {
      "point_cloud": mesh,
      "description": "Generate CadQuery code",
      "file_name":    os.path.splitext(os.path.basename(mesh_path))[0],
      "answer":         ""
    }

    mesh = collate(
        [example], 
        processor, 
        n_points, 
        eval=True
    )

    generated_ids = model.generate(
        input_ids=mesh['input_ids'].to(model.device),
        attention_mask=mesh['attention_mask'].to(model.device),
        point_clouds=mesh['point_clouds'].to(model.device),
        is_pc=mesh['is_pc'].to(model.device),
        is_img=mesh['is_img'].to(model.device), # putting here even if not required
        max_new_tokens=token_size, # was 768
        
        # FINE TUNING PARAMETERS
        # temperature=0.7, # higher for more creative outputs
        # top_p=0.3,
        # top_k=10,
        # num_beams=1
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(mesh.input_ids, generated_ids)
    ]
    py_strings = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    stem = os.path.splitext(os.path.basename(mesh_path))[0]

    for idx, code in enumerate(py_strings):
        # if you only have one string per mesh, idx will be 0 anyway
        py_script_name = f"cadquery_{stem}.py"
        out_file = os.path.join(output_path, py_script_name)
        with open(out_file, "w",encoding="utf-8") as f:
            f.write(code)

    print(f"Cadquery script written to {out_file}\n")

def view_script_mesh(pyfile):
    return 0

# taken from cadrecode2mesh.py
def compound_to_mesh(compound):
    vertices, faces = compound.tessellate(0.001, 0.1)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in vertices], faces)

def py_file_to_mesh_file(pyfile):
    print(f"Converting {pyfile} to mesh file...")
    try:
        with open(pyfile, 'r') as f:
            py_string = f.read()
        exec(py_string, globals())
        compound = globals()['r'].val()
        mesh = compound_to_mesh(compound)
        mesh.export(pyfile[:-3] + '.stl')
    except:
        pass

# TODO
# show metrics like cd, iou
# built from evaluate.py
def get_metrics(gt_mesh_path, stlfile, n_points):
    gt_mesh = trimesh.load_mesh(gt_mesh_path, force='mesh')
    generated_mesh = trimesh.load_mesh(stlfile, force='mesh') # Also load the generated STL for evaluation

    iou = evaluate.compute_iou(gt_mesh, generated_mesh)
    cd = evaluate.compute_chamfer_distance(gt_mesh, generated_mesh, n_points)
    print(f"Metrics for {stlfile}:")
    print(f"  IoU: {iou}")
    print(f"  Chamfer Distance: {cd}")
    return iou, cd

def validate(pyfile):
    class InvalidScript(Exception):
        pass
    print(f"Validating {pyfile} ")
    try:
        # 1) Read the generated script into a string:
        with open(pyfile, 'r', encoding='utf-8') as f:
            source = f.read()

        # 2) Compile the *contents*, passing the real filename for better error messages:
        compile(source, pyfile, 'exec')
        print(f"Script is syntactically valid.")

    except SyntaxError as e:
        # This will catch real syntax errors in the file you generated
        print(f"Syntax Error in {pyfile}:")
        print(f"  Line {e.lineno}, offset {e.offset}: {e.text.strip()}")
        print(f"  {e.msg}")
        raise InvalidScript()

def main():
    parser = argparse.ArgumentParser(description="2d3dgen: text or batch 3D mesh mode.",
                                     epilog="NOTE: N_points can be adjusted, but 256 is optimal. Any more and the outputs are in a different language :)")
    parser.add_argument("--model", type=str, default='maksimko123/cadrille',
                        help="Model: Specify which model to use (default maksimko123/cadille).\n"
                        "When using a local path, format as: './path/to/model/checkpoint/'.")
    parser.add_argument("--mode", choices=["text", "3d"], required=True,
                        help="Mode: 'text' for prompt, '3d' for mesh/batch mode.")
    parser.add_argument("--prompt", type=str,
                        help="Text description for CadQuery generation (text mode).")
    parser.add_argument("--mesh", type=str,
                        help="Single mesh file path for CadQuery generation (3d mode).")
    parser.add_argument("--mesh_dir", type=str,
                        help="Directory of mesh files (PLY/STL) to batch process (3d mode).")
    parser.add_argument("--output_dir", type=str, default="./generated_scripts",
                        help="Directory to save generated CadQuery scripts.")
    parser.add_argument("--n_points", type=int, default=256,
                        help='Specify n sample points (default 256)')
    parser.add_argument("--max_retries", type=int, default=5,
                        help='Specify max retries (default 5)')
    parser.add_argument("--token_size", type=int, default=768,
                        help='Specify token size (default 768)')
    args = parser.parse_args()
    print("Clearing CUDA cache...")
    torch.cuda.empty_cache()
    print("Cleared CUDA cache.")
    
    processor, model = load_model_and_processor(args.model, args.mode)
    os.makedirs(args.output_dir, exist_ok=True)
    if(args.token_size < args.n_points):
        print(f"[**WARNING**]: token_size {args.token_size} is less than n_points {args.n_points}. "
              "This may lead to unexpected results. Consider increasing token_size or reducing n_points.")
    if args.mode == "text":
        user_prompt = args.prompt or input("Enter text prompt: ")
        # # When in text mode, the original prompt is `build_prompt_text(user_prompt)`.
        # original_prompt = build_prompt_text(user_prompt)
        # raw = generate_cadquery_script(processor, model, "text", user_prompt)
        # # final = postprocess_script(raw, original_prompt_text=original_prompt)
        # final = generate_cadquery_script(processor, model, "text", user_prompt)
        # print("\nGenerated CadQuery script:\n")
        # print(final)
    else:  # 3d mode
        if args.mesh_dir:
            for fname in os.listdir(args.mesh_dir):
                if fname.lower().endswith(('.ply', '.stl')):
                    mesh_path = os.path.join(args.mesh_dir, fname)
                    generate_cadquery_script_from_3d(processor, 
                                                    model, 
                                                    args.token_size,
                                                    mesh_path, 
                                                    args.output_dir, 
                                                    args.n_points)
          
        elif args.mesh:
            # uncomment below to view the point cloud
            view_point_cloud(process_mesh_to_point_cloud(args.mesh, args.n_points), args.mesh)
            for retry in range(1, args.max_retries + 1):
                try:
                    print(f"------------- Attempt {retry} -------------\n")
                    generate_cadquery_script_from_3d(processor, 
                                                    model, 
                                                    args.token_size,
                                                    args.mesh, 
                                                    args.output_dir, 
                                                    args.n_points)
                    pyfile = os.path.join(args.output_dir, f"cadquery_{os.path.splitext(os.path.basename(args.mesh))[0]}.py")
                    stlfile = os.path.join(args.output_dir, f"cadquery_{os.path.splitext(os.path.basename(args.mesh))[0]}.stl")
                    validate(pyfile)
                    py_file_to_mesh_file(pyfile)
                    # view_script_mesh(pyfile)
                    cd, iou = get_metrics(args.mesh, stlfile, args.n_points)
                    if cd is not None and iou is not None:
                        break
                    break
                except Exception as e:
                    print(f"Attempt {retry}: Error occurred - {e}. Retrying...\n")
        else:
            print("Error: In 3d mode, specify --mesh or --mesh_dir")

if __name__ == "__main__":
    main()