# file: chat_cadrille.py
import os
import sys
import torch
import trimesh
from transformers import AutoProcessor
from cadrille import Cadrille

sys.path.append(".")

# Function to load a mesh and convert to tensors
def load_mesh_as_tensor(file_path):
    mesh = trimesh.load(file_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Loaded object is not a mesh")
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)  # (N, 3)
    faces = torch.tensor(mesh.faces, dtype=torch.long)           # (F, 3)
    return vertices, faces

# Load model and processor
print("Loading model...")
model = Cadrille.from_pretrained(
    "maksimko123/cadrille",
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto"
)


print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-32B-Instruct",
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28,
    padding_side="left",
    use_fast=True
)

model.eval()

print("Model and processor loaded.\n")

# Prompt loop
print("Enter a prompt and optional mesh path (ex: 'design a cube | cube.obj'). Type 'quit' to exit.")
while True:
    user_input = input(">>> ")
    if user_input.lower() in ("quit", "exit"):
        break

    # Parse text and optional mesh path
    if "|" in user_input:
        prompt, mesh_path = map(str.strip, user_input.split("|", 1))
        if not os.path.exists(mesh_path):
            print(f"Mesh file not found: {mesh_path}. Continuing without mesh input.")
            mesh_path = None

    # Compose prompt
    full_prompt = (
        f"Generate a complete and runnable CadQuery Python script for the following:\n\n{prompt}\n\n"
        f"Make sure to include 'import cadquery as cq' and 'show_object(result)' at the end. Append newlines in the form of '\'."
    )

    # Tokenize text
    inputs = processor(text=full_prompt, return_tensors="pt", padding=True).to(model.device)
    input_ids = inputs.input_ids

    # Prepare mesh input if provided
    if mesh_path:
        verts, faces = load_mesh_as_tensor(mesh_path)
        mesh_dict = {
            "vertices": verts.unsqueeze(0).to(model.device),  # (1, N, 3)
            "faces": faces.unsqueeze(0).to(model.device)      # (1, F, 3)
        }
        is_mesh = torch.tensor([True], device=model.device)
    else:
        mesh_dict = None
        is_mesh = torch.tensor([False], device=model.device)

    # Generate script
    max_tokens_to_generate = 10000  # Large for full code output

    out_ids = model.generate(
        input_ids=input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_tokens_to_generate,
        point_clouds=None,
        is_pc=torch.tensor([False], device=model.device),
        is_img=torch.tensor([False], device=model.device),
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )[0]

    # Decode and clean output
    new_tokens = out_ids[input_ids.shape[-1]:]
    code = processor.decode(new_tokens, skip_special_tokens=True).strip()

    if not code.startswith("import cadquery as cq"):
        code = "import cadquery as cq\n\n" + code

    if "show_object(" not in code and "result =" in code:
        code += "\nshow_object(result)"
    elif "show_object(" not in code:
        code += "\n# If the above code produces an object named 'result', uncomment and run:\n# show_object(result)"

    # Print and save
    print("\n=== Generated CadQuery script ===\n")
    print(code)
    print("\n=================================\n")

    with open("generated_cadquery_script.py", "w") as f:
        f.write(code)
    print("Script saved to generated_cadquery_script.py")
