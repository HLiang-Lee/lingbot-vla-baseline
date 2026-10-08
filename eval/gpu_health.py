"""GPU admission checks and runtime failure classification; no resets or ECC suppression."""
import csv,io,json,subprocess,time

def inventory():
    p=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,ecc.errors.uncorrected.volatile.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=20)
    return {int(row[0]):{'index':int(row[0]),'uuid':row[1].strip(),'pci_bus_id':row[2].strip(),'uncorrected_ecc_volatile':int(row[3]) if row[3].strip().isdigit() else None} for row in csv.reader(io.StringIO(p.stdout))}

def failure_kind(text):
    lower=text.lower()
    if any(s in lower for s in ['vk::devicelosterror','errordevicelost','uncorrectable ecc','uncorrected ecc','cuda error: device is lost','cuda error: unknown error','cuda error: unspecified launch failure','cuda-capable device(s) is/are busy or unavailable']):
        return 'gpu_fault'
    if any(s in text for s in ['ModuleNotFoundError:','ImportError:','cuRobo planner initialization failed']):
        return 'dependency_error'
    return None

def probe(gpu,sim_python,env,cwd):
    code="""import torch
x=torch.ones((512,512),device='cuda'); y=x@x; torch.cuda.synchronize()
assert y[0,0].item()==512
from curobo.wrap.reacher.motion_gen import MotionGen
from envs.robot.planner import CuroboPlanner
print('CUDA_AND_CUROBO_OK',torch.cuda.get_device_name())
"""
    started=time.time()
    try:
        p=subprocess.run([sim_python,'-c',code],env=env,cwd=cwd,capture_output=True,text=True,timeout=120)
        return {'gpu':gpu,'ok':p.returncode==0,'returncode':p.returncode,'stdout':p.stdout[-2000:],'stderr':p.stderr[-4000:],'seconds':time.time()-started}
    except subprocess.TimeoutExpired:
        return {'gpu':gpu,'ok':False,'error':'CUDA/cuRobo preflight timed out','seconds':time.time()-started}
