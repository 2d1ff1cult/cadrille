# torch imports
import torch
from torch.utils.data import DataLoader, ConcatDataset
from transformers import AutoProcessor
from pytorch3d.ops import sample_farthest_points

# tools and system imports
from tqdm import tqdm
from functools import partial
import argparse
import os
import evaluate

# 3d imports
import cadquery as cq
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import trimesh
import numpy as np

# cadrille imports
from dataset import Text2CADDataset, CadRecodeDataset
from cadrille import Cadrille, collate
import dataset

class InvalidScript(Exception):
    pass
class InvalidMesh(Exception):
    pass

def load_model_and_processor(model_name): # can also use path to checkpoint
    """
    Load the pretrained Cadrille model and processor from Hugging Face.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model and processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=256 * 28 * 28, 
        max_pixels=1280 * 28 * 28,
        padding_side='left',
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
def process_mesh_to_point_cloud(mesh_path, n_points, n_pre_points=8192):
    """
    Processes a mesh file into a normalized point cloud.
    """
    # See dataset.py
    # At training we normalize by 100, but at inference we do not
    normalize_std = 100

    mesh = trimesh.load_mesh(mesh_path, force='mesh')
    # mesh = dataset._augment_pc(mesh)
    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    
    # Use PyTorch3D for farthest point sampling
    tensor_vertices = torch.tensor(vertices, dtype=torch.float32).unsqueeze(0)
    _, ids = sample_farthest_points(tensor_vertices, K=n_points)
    points = tensor_vertices[0, ids[0]].numpy()

    # place center of mass at origin
    centroid = points.mean(axis=0)

    # as of 7/30/2025, found out that this *may* be required
    # resulting meshes are HUGE
    points = points - centroid
    # See dataset.py:
    # During training, it seems we do this:
    points = points / normalize_std

    # But at inference (i.e. anything other than 'train' or 'val'):
    # However, please note that class CadRecodeDataset in dataset.py
    # seems to do this already. Therefore, we MAY NOT need to do this here.
    # since effectively we're doing things TWICE

    points = (points - 0.5) * 2

    return np.asarray(points)

def view_point_cloud(point_cloud_data, mesh_path):
    # create an o3d PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    o3d.visualization.draw_geometries([pcd],
                                      window_name=f"{len(point_cloud_data)} points - {mesh_path}",
                                      width=1280,
                                      height=720
                                      )

def OLD_generate_cadquery_script_from_3d(processor, model, token_size, mesh_path, output_path, n_points):
    # old
    print(f"Generating Cadquery script for mesh: {mesh_path}")

    mesh = process_mesh_to_point_cloud(mesh_path, n_points)

    # the reason for poor metrics might stem from here...
    example = {
      "point_cloud": mesh,
      "description": "", # Generate CadQuery code
      "file_name":    os.path.splitext(os.path.basename(mesh_path))[0],
      "answer":         ""
    }

    # ...and here
    mesh = collate(
        [example], 
        processor, 
        n_points, 
        eval=True
    )
    
    # tensor size sanity check
    # print({ k: v.shape for k,v in mesh.items() if isinstance(v, torch.Tensor) })

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

# does not regenerate, do not use until fixed
def WORKING_generate_cadquery_script_from_3d(processor,
                                     model,
                                     token_size,
                                     mesh_folder,
                                     output_path,
                                     n_points,
                                     max_retries):
    dataset = CadRecodeDataset(
        root_dir='./',
        split=mesh_folder,
        n_points=n_points,
        normalize_std_pc=100,
        noise_scale_pc=0.01,
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode='pc')
    batch_size = 32

    n_samples = 1
    dataloader = DataLoader(
        dataset=ConcatDataset([dataset] * n_samples),
        batch_size=batch_size,
        num_workers=24,
        collate_fn=partial(collate, processor=processor, n_points=256, eval=True))

    for batch in tqdm(dataloader):
        generated_ids = model.generate(
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device),
            point_clouds=batch['point_clouds'].to(model.device),
            is_pc=batch['is_pc'].to(model.device),
            is_img=batch['is_img'].to(model.device),
            pixel_values_videos=batch['pixel_values_videos'].to(model.device) if batch.get('pixel_values_videos', None) is not None else None,
            video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get('video_grid_thw', None) is not None else None,
            max_new_tokens=token_size)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(batch.input_ids, generated_ids)
        ]
        py_strings = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        py_dest = output_path
        # need to validate each script at the given index
        for stem, py_string in zip(batch['file_name'], py_strings):
            for retry in range(1, max_retries+1):
                py_name = f'cadquery_{stem}.py'
                py_path = os.path.join(py_dest, py_name)
                # handle different mesh file types
                # output an stl anyway
                for file in os.listdir(mesh_folder):
                    name, ext = os.path.splitext(file)
                    if name == stem:
                        gtmesh = os.path.join(mesh_folder, file)
                        generated_stl = os.path.join(output_path, f"cadquery_{stem}.stl")
                        if ext in ['.glb', '.gltf', '.off']:
                            print(f"Unsupported file type for {file}, skipping...")
                            continue
                script = os.path.join(py_dest, py_name)
                print(f"------------------------- Attempt {retry}: {gtmesh} -------------------------\n")
                with open(script, 'w') as f:
                    f.write(py_string)
                    if not py_string.strip():
                        print(f"Empty script generated for {py_name}. Retrying... ({retry}/{max_retries})")
                        continue
                py_file_to_mesh_file(py_path)
                validate(py_path)
                iou, cd = get_metrics(gtmesh, generated_stl, n_points)
                if(validate(py_path) is False):
                    print(f"Invalid script generated: {py_name}. Retrying... ({retry}/{max_retries})")
                    continue
                if cd <= 1:# and iou >= 0.2:
                    view_script_mesh(generated_stl, gtmesh)
                    print(f"Valid mesh. Cadquery script written to {py_name}\n")
                    break
                else:
                    # for debugging purposes, comment below line for speed
                    view_script_mesh(generated_stl, gtmesh)
                    print((f"Low metrics: CD={cd}, IoU={iou}. Retrying..."))
                    continue
                    # raise InvalidMesh(f"Low metrics: CD={cd}, IoU={iou}")
        print(f"Finished. All files located in {output_path}\n")
    # except InvalidMesh as i:
    #     print(f"Invalid mesh generated: {i}")
    #     continue
    # except Exception as e:
    #     print(f"Error occurred - {e}. Retrying...\n")
    #     continue

def TEST_generate_cadquery_script_from_3d(processor,
                                     model,
                                     token_size,
                                     mesh_folder,
                                     output_path,
                                     n_points,
                                     max_retries):
    dataset = CadRecodeDataset(
        root_dir='./',
        split=mesh_folder,
        n_points=n_points,
        normalize_std_pc=100,
        noise_scale_pc=0.01,
        img_size=128,
        normalize_std_img=200,
        noise_scale_img=-1,
        num_imgs=4,
        mode='pc')
    batch_size = 32

    n_samples = 1
    dataloader = DataLoader(
        dataset=ConcatDataset([dataset] * n_samples),
        batch_size=batch_size,
        num_workers=24,
        collate_fn=partial(collate, processor=processor, n_points=256, eval=True))

    for batch in tqdm(dataloader):
        # Track which items in the batch still need successful generation
        pending_items = list(range(len(batch['file_name'])))
        successful_items = set()
        
        # Store batch items that need regeneration
        items_to_retry = {}
        for i, stem in enumerate(batch['file_name']):
            items_to_retry[i] = {
                'stem': stem,
                'retry_count': 0,
                'max_retries': max_retries
            }
        
        while pending_items and any(items_to_retry[i]['retry_count'] < items_to_retry[i]['max_retries'] 
                                   for i in pending_items):
            
            # Increment retry count for pending items
            for i in pending_items:
                items_to_retry[i]['retry_count'] += 1
                retry_num = items_to_retry[i]['retry_count']
                stem = items_to_retry[i]['stem']
                print(f"------------------------- Attempt {retry_num}: {stem} -------------------------\n")
            
            # Generate new scripts for all pending items
            generated_ids = model.generate(
                input_ids=batch['input_ids'].to(model.device),
                attention_mask=batch['attention_mask'].to(model.device),
                point_clouds=batch['point_clouds'].to(model.device),
                is_pc=batch['is_pc'].to(model.device),
                is_img=batch['is_img'].to(model.device),
                pixel_values_videos=batch['pixel_values_videos'].to(model.device) if batch.get('pixel_values_videos', None) is not None else None,
                video_grid_thw=batch['video_grid_thw'].to(model.device) if batch.get('video_grid_thw', None) is not None else None,
                max_new_tokens=token_size)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(batch['input_ids'], generated_ids)
            ]
            
            py_strings = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            
            # Process each pending item
            newly_successful = []
            
            for i in pending_items:
                stem = batch['file_name'][i]
                py_string = py_strings[i]
                retry_num = items_to_retry[i]['retry_count']
                
                # Check if script is empty
                if not py_string.strip():
                    print(f"Empty script generated for {stem}. Retrying... ({retry_num}/{max_retries})")
                    continue
                
                py_name = f'cadquery_{stem}.py'
                py_path = os.path.join(output_path, py_name)
                
                # Write the newly generated script
                with open(py_path, 'w') as f:
                    f.write(py_string)
                
                # Find ground truth mesh file
                gtmesh = None
                generated_stl = os.path.join(output_path, f"cadquery_{stem}.stl")
                
                for file in os.listdir(mesh_folder):
                    name, ext = os.path.splitext(file)
                    if name == stem:
                        if ext in ['.glb', '.gltf', '.off']:
                            print(f"Unsupported file type for {file}, skipping...")
                            break
                        gtmesh = os.path.join(mesh_folder, file)
                        break
                
                if gtmesh is None:
                    print(f"No supported ground truth mesh found for {stem}")
                    continue
                
                # Validate and test the script
                try:
                    py_file_to_mesh_file(py_path)
                    
                    if not validate(py_path):
                        print(f"Invalid script generated: {py_name}. Retrying... ({retry_num}/{max_retries})")
                        continue
                    
                    iou, cd = get_metrics(gtmesh, generated_stl, n_points)
                    
                    if cd <= 1:  # and iou >= 0.2:
                        view_script_mesh(generated_stl, gtmesh)
                        print(f"Valid mesh. Cadquery script written to {py_name}\n")
                        newly_successful.append(i)
                    else:
                        # for debugging purposes, comment below line for speed
                        view_script_mesh(generated_stl, gtmesh)
                        print(f"Low metrics: CD={cd}, IoU={iou}. Retrying... ({retry_num}/{max_retries})")
                        continue
                        
                except Exception as e:
                    print(f"Error processing {py_name}: {str(e)}. Retrying... ({retry_num}/{max_retries})")
                    continue
            
            # Remove successful items from pending list
            for i in newly_successful:
                pending_items.remove(i)
                successful_items.add(i)
            
            # Remove items that have exceeded max retries
            pending_items = [i for i in pending_items 
                           if items_to_retry[i]['retry_count'] < items_to_retry[i]['max_retries']]
        
        # Report final status
        for i, stem in enumerate(batch['file_name']):
            if i not in successful_items:
                print(f"Failed to generate valid script for {stem} after {max_retries} attempts")
    
    print(f"Finished. All files located in {output_path}\n")

def view_script_mesh(stlfile, gtmesh):
    """
    View generated mesh against ground truth mesh using Open3D.
    """
    mesh = o3d.io.read_triangle_mesh(stlfile)
    mesh.paint_uniform_color([0.9, 0.2, 0.2]) # red for generated mesh
    mesh = mesh.compute_vertex_normals()
    mesh = mesh.scale(0.005, center=(0.0, 0.0, 0.0)) # scale and center


    gt = o3d.io.read_triangle_mesh(gtmesh)
    gt.paint_uniform_color([0.4, 0.9, 0.4]) # green for ground truth
    gt = gt.compute_vertex_normals()
    # gt = gt.translate((0.0, 1.0, 0.0))

    o3d.visualization.gui.Application.instance.initialize()

    # Create window and scene
    window = o3d.visualization.gui.Application.instance.create_window(
        f"{stlfile} (Red Generated) vs. {gtmesh} (Green Ground Truth)", 1024, 768)
    scene = o3d.visualization.gui.SceneWidget()
    scene.scene = o3d.visualization.rendering.Open3DScene(window.renderer)

    # Create materials
    # Semi-transparent material for generated mesh (red)
    generated_material = o3d.visualization.rendering.MaterialRecord()
    #generated_material.base_color = [0.9, 0.4, 0.4, 0.1]  # Semi-transparent red
    generated_material.shader = "defaultLitTransparency"

    # Semi-transparent material for ground truth mesh (green)
    gt_material = o3d.visualization.rendering.MaterialRecord()
    #gt_material.base_color = [0.1, 0.9, 0.1, 1.0]  # Solid green
    gt_material.shader = "defaultLit"

    # Add geometries to scene
    scene.scene.add_geometry("generated_mesh", mesh, generated_material)
    scene.scene.add_geometry("ground_truth_mesh", gt, gt_material)

    # Set up the camera to view both meshes
    # Combine bounding boxes to get the full scene bounds
    bounds = mesh.get_axis_aligned_bounding_box()
    gt_bounds = gt.get_axis_aligned_bounding_box()
    bounds += gt_bounds  # Combine bounding boxes
    
    scene.setup_camera(60, bounds, bounds.get_center())

    # Add scene to window
    window.add_child(scene)

    # Run the application
    o3d.visualization.gui.Application.instance.run()

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
        stl = mesh.export(pyfile[:-3] + '.stl')
        print(f"Mesh file saved to {pyfile[:-3] + '.stl'}")
        return stl
    except:
        pass

def get_metrics(gt_mesh_path, stlfile, n_points):
    gt_mesh = trimesh.load_mesh(gt_mesh_path, force='mesh')
    generated_mesh = trimesh.load_mesh(stlfile, force='mesh') # Also load the generated STL for evaluation

    generated_mesh = generated_mesh.apply_scale(0.005)
    generated_mesh.apply_translation(-generated_mesh.centroid)
    gt_mesh.apply_translation(-gt_mesh.centroid)

    iou = evaluate.compute_iou(gt_mesh, generated_mesh)
    cd = evaluate.compute_chamfer_distance(gt_mesh, generated_mesh, n_points)
    print(f"Metrics for {stlfile}:")
    print(f"  IoU: {iou}")
    print(f"  Chamfer Distance: {cd}")
    iou = float(iou)
    cd = float(cd)
    return iou, cd

def validate(pyfile):
    print(f"Validating {pyfile} ")
    try:
        # 1) Read the generated script into a string:
        with open(pyfile, 'r', encoding='utf-8') as f:
            source = f.read()

        # 2) Compile the *contents*, passing the real filename for better error messages:
        compile(source, pyfile, 'exec')
        print(f"Script is syntactically valid.")
        return True

    except SyntaxError as e:
        # This will catch real syntax errors in the file you generated
        print(f"Syntax Error in {pyfile}:")
        print(f"  Line {e.lineno}, offset {e.offset}: {e.text.strip()}")
        print(f"  {e.msg}")
        # raise InvalidScript()
        

def main():
    parser = argparse.ArgumentParser(description="2d3dgen: text or batch 3D mesh mode.",
                                     usage="\n\tpy -3.10 2d3dgen_cadrille.py --mesh_dir <folder of meshes>",
                                     epilog="NOTE: N_points can be adjusted, but 256 is optimal. Any more and the outputs are in a different language :)")
    parser.add_argument("--model", type=str, default='maksimko123/cadrille',
                        help="Model: Specify which model to use (default maksimko123/cadille).\n"
                        "When using a local path, format as: './path/to/model/checkpoint/'.")
    parser.add_argument("--mesh", type=str,
                        help="Single mesh file path for CadQuery generation (3d mode).")
    parser.add_argument("--mesh_dir", type=str, required=True,
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

    processor, model = load_model_and_processor(args.model)
    os.makedirs(args.output_dir, exist_ok=True)
    if(args.token_size < args.n_points):
        print(f"[**WARNING**]: token_size {args.token_size} is less than n_points {args.n_points}. "
              "This may lead to unexpected results. Consider increasing token_size or reducing n_points.")
    
    if args.mesh_dir:
        TEST_generate_cadquery_script_from_3d(processor, 
                                        model, 
                                        args.token_size,
                                        args.mesh_dir, 
                                        args.output_dir, 
                                        args.n_points,
                                        args.max_retries
        )

if __name__ == "__main__":
    main()