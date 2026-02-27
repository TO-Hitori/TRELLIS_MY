import os
os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn', 'sdpa'
os.environ["XFORMERS_NO_TRITON"] = "1"
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import warnings
warnings.filterwarnings("ignore", message="A matching Triton is not available")

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

# Load a pipeline from a model folder or a Hugging Face model hub.
pipeline = TrellisImageTo3DPipeline.from_pretrained("E:\CheckPoint\TRELLIS-image-large")
pipeline.cuda()

# Load an image
images = [
    Image.open("D:/MyProject/3d/TRELLIS_new/assets/My_Image/003_azi180_ele0.png"),
    Image.open("D:/MyProject/3d/TRELLIS_new/assets/My_Image/004_azi270_ele0.png"),
    Image.open("D:/MyProject/3d/TRELLIS_new/assets/My_Image/005_azi315_ele0.png"),
    Image.open("D:/MyProject/3d/TRELLIS_new/assets/My_Image/006_azi10_ele10.png"),
]

# Run the pipeline
outputs = pipeline.run_multi_image(
    images,
    seed=1,
    mode='multidiffusion', # ['stochastic', 'multidiffusion']
    # Optional parameters
    sparse_structure_sampler_params={
        "steps": 8,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 8,
        "cfg_strength": 7,
    },
)
# outputs is a dictionary containing generated 3D assets in different formats:
# - outputs['mesh']: a list of meshes


glb = postprocessing_utils.to_glb(
    outputs['mesh'][0],
    outputs['mesh'][0],
    # Optional parameters
    simplify=0.95,          # Ratio of triangles to remove in the simplification process
    texture_size=1024,      # Size of the texture used for the GLB
)
glb.export(r"D:\MyProject\3d\TRELLIS_new\RESULT\multi_multidiffusion_sample_1111.glb")
print("export glb success!")