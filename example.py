import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn', 'sdpa'
os.environ["XFORMERS_NO_TRITON"] = "1"
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import warnings
warnings.filterwarnings("ignore")

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("E:\CheckPoint\TRELLIS-image-large")
pipeline.cuda()
print("load model success!")

# Load an image
image = Image.open(r"D:\MyProject\3d\TRELLIS_new\assets\example_multi_image\yoimiya_1.png")

# Run the pipeline
outputs = pipeline.run(
    image,
    seed=1,
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 4,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 4,
        "cfg_strength": 3,
    },
)

print("----------------------------run model success!")
print("Type:", type(outputs["mesh"]))
print("Type:", type(outputs["mesh"][0]))
print("----------------------------outputs end")
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['gaussian']: a list of 3D Gaussians
# - outputs['radiance_field']: a list of radiance fields
# - outputs['mesh']: a list of meshes


# GLB files can be extracted from the outputs
glb = postprocessing_utils.to_glb(
    app_rep=outputs['mesh'][0],
    mesh=outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export(r"D:\MyProject\3d\TRELLIS_new\RESULT\yoimiya000.glb")
path = r"D:\MyProject\3d\TRELLIS_new\RESULT"
print(f"export glb success to", path)

