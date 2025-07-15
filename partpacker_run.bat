@echo off

cd PartPacker
env PYTHONPATH=. python /vae/scripts/infer.py --ckpt_path pretrained/vae.pt --input assets/meshes/ --output_dir output/

# flow 3D generation from images
env PYTHONPATH=. python /flow/infer.py --ckpt_path pretrained/flow.pt --input assets/images/ --output_dir output/

# Prevent fragmentation errors
python ..\imports.py

# change launch() in gradio app
powershell -Command "(Get-Content app.py) -replace 'block\.launch\(\)', 'block.launch(share=True)' | Set-Content app.py"

# open local gradio app (single GPU)
python app.py

# open local gradio app with multi-GPU support
# python app.py --multi --share