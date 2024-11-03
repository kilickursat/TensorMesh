from torch.utils.cpp_extension import load
import os 
spsolve_cpp = load(
    name="spsolve", sources=[
        os.path.join(__file__,"spsolve.cpp")], 
        verbose=True)
