  # file: chat_cadrille.py
import torch
# from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoProcessor
from cadrille import Cadrille
import sys
import time

sys.path.append(".")

# load the model & processor
print("Loading model")
model = Cadrille.from_pretrained(
    # Uncomment the following line to use a different model
    # "Qwen/Qwen2.5-VL-32B-Instruct", # broken for some reason?
    # "Qwen/Qwen2.5-VL-7B-Instruct", # broken for some reason?
    # "Qwen/Qwen2.5-VL-3B-Instruct", # broken for some reason?
    # "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    # "maksimko123/Cadrille", # pretrained from Hugging Face as stated in paper
    attn_implementation='sdpa', # comment when using torch.nn.attention.SDPBackend.MATH
    device_map="auto"
).to("cuda")

print("Loading processor...")
processor = AutoProcessor.from_pretrained(
    # Uncomment the following line to use a different model
    # "Qwen/Qwen2.5-VL-32B-Instruct", # broken for some reason?
    # "Qwen/Qwen2.5-VL-7B-Instruct", # broken for some reason?
    # "Qwen/Qwen2.5-VL-3B-Instruct", # broken for some reason?
    # "Qwen/Qwen2-VL-7B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
    min_pixels=256 * 28 * 28, 
    max_pixels=1280 * 28 * 28,
    padding_side='left',
    # use_fast=True
)
print(f"Processor: {processor}")
print("Model loaded")

model.eval()

print("Enter a text prompt (or ‘quit’ to exit).")
while True:
    prompt = input(">>> ")
    if prompt.lower() in ("quit", "exit"):
        break

    # --- Improvement 1: Add a strong prompt prefix for CadQuery code generation ---
    # REALLY LONG PROMPT
    full_prompt = f"Generate a complete and runnable CadQuery Python script using only the following functions: Sketch.edge(), Sketch.segment(), Sketch.arc(), Sketch.spline(), Sketch.close(), Sketch.assemble(), Sketch.constrain(), Sketch.solve(), Workplane.all(), Workplane.size(), Workplane.vals(), Workplane.add(), Workplane.val(), Workplane.first(), Workplane.item(), Workplane.last(), Workplane.end(), Workplane.vertices(), Workplane.faces(), Workplane.edges(), Workplane.wires(), Workplane.solids(), Workplane.shells(), Workplane.compounds(), NearestToPointSelector(), BoxSelector(), BaseDirSelector(), ParallelDirSelector(), DirectionSelector(), DirectionNthSelector(), LengthNthSelector(), AreaNthSelector(), RadiusNthSelector(), PerpendicularDirSelector(), TypeSelector(), DirectionMinMaxSelector(), CenterNthSelector(), BinarySelector(), AndSelector(), SumSelector(), SubtractSelector(), InverseSelector(), StringSyntaxSelector(), BoundBox.init(), BoundBox.add(), BoundBox.enlarge(), BoundBox.findOutsideBox2D(), BoundBox.isInside(), Color.toTuple(), Location.toTuple(), Matrix.transposed_list(), Shape.BoundingBox(), Shape.Center(), Shape.CenterOfBoundBox(), Shape.Closed(), Shape.CombinedCenter(), Shape.CombinedCenterOfBoundBox(), Shape.CompSolids(), Shape.Compounds(), Shape.Edges(), Shape.Faces(), Shape.Shells(), Shape.Solids(), Shape.Vertices(), Shape.Volume(), Shape.Wires(), Shape.add(), Shape.eq(), Compound.ancestors(), Compound.cut(), Compound.fuse(), Compound.intersect(), Compound.makeCompound(), Compound.makeText(), Compound.remove(), Compound.siblings(), Workplane.copyWorkplane(), Workplane.cskHole(), Workplane.cutBlind(), Workplane.cutEach(), Workplane.cutThruAll(), Workplane.cylinder(), Workplane.each(), Workplane.eachpoint(), Workplane.ellipse(), Workplane.ellipseArc(), Workplane.export(), Workplane.exportSvg(), Workplane.extrude(), Workplane.fillet(), Workplane.filter(), Workplane.findFace(), Workplane.findSolid(), Workplane.hLine(), Workplane.hLineTo(), Workplane.hole(), Workplane.interpPlate(), Workplane.intersect(), Workplane.invoke(), Workplane.item(), Workplane.rect(), Workplane.circle(), Workplane.threePointArc(), Workplane.center(), Workplane.polygon(), Workplane.pushPoints(), Workplane.mirrorY(), Workplane.spline(), Workplane.close(), Workplane.mirrorX(), Workplane.moveTo(), Workplane.mirror(), Workplane.union(), Workplane.rotate(), Workplane.line(), Workplane.workplane(), Workplane.box(), Workplane.hole(), CQ(), Workplane(), Shape(), Vertex(), Edge(), Wire(), Face(), Shell(), Solid(), Compound(), Selector(), NearestToPointSelector(), BoxSelector(), BaseDirSelector(), ParallelDirSelector(), DirectionSelector(), DirectionNthSelector(), PerpendicularDirSelector(), TypeSelector(), DirectionMinMaxSelector(), BinarySelector(), AndSelector(), SumSelector(), SubtractSelector(), InverseSelector(), StringSyntaxSelector(), add(), chamfer(), combineSolids(), union(), findSolid(), rotate(), split(), translate(), toSvg(), each(), eachpoint(), all(), size(), vals(), add(), val(), first(), item(), last(), end(), rect(), fillet2D(), loft(), sweep(), parametricCurve(), polyline(), revolve(), text(), shell(), fillet(), chamfer(), split(), rotateAboutCenter(). Make sure to make the distinction between a Sketch, Workplane, BoundBox, and Location. Satisfy the prompt given:\n\n{prompt}\n\nMake sure to include 'import cadquery as cq' and a workplane initialization for 'w0 = cq.Workplane('XY')'. Add 'show_object(result)' at the end. Append newlines in the form of '\'."

    # prepare inputs
    inputs = processor(
        text=full_prompt,
        return_tensors="pt",
        padding=True
    ).to(model.device)
    input_ids = inputs.input_ids

    max_tokens_to_generate = 768

    # --- NEW: FlashAttention Context Wrapper ---
    # Only applies if the model supports torch's native SDPA interface.
    start_time = time.perf_counter()
    # with sdpa_kernel(SDPBackend.MATH): # or FLASH_ATTENTION
    out_ids = model.generate(
        input_ids=inputs.input_ids,
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

    # strip off the prompt tokens
    new_tokens = out_ids[input_ids.shape[-1]:]
    code = processor.decode(new_tokens, skip_special_tokens=True)

    # post processing the generated code
    cleaned_code = code.strip()
    # if not cleaned_code.startswith("import cadquery as cq"):
    #     cleaned_code = "import cadquery as cq\n\n w0 = cq.Workplane('XY')\n\n" + cleaned_code

    if "show_object(" not in cleaned_code and "result =" in cleaned_code:
        cleaned_code += "\nshow_object(result)"
    elif "show_object(" not in cleaned_code and "result =" not in cleaned_code:
        cleaned_code += "\n# If the above code produces an object named 'result', uncomment and run:\n# show_object(result)"

    print("\n=== Generated CadQuery script ===\n")
    print(cleaned_code)
    print("\n=================================\n")

    with open("generated_cadquery_script.py", "w") as f:
        f.write(cleaned_code)
    print("Script saved to generated_cadquery_script.py. You can open this in CQ-editor.")
    end_time = time.perf_counter()
    duration = end_time - start_time
    print(f"\nResponse generated in {duration:.2f} seconds\n")
