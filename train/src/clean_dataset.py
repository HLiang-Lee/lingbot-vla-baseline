"""CPU-only raw RoboTwin adapter. q[t] is recorded state; supervision starts at q[t+1]."""
import collections,json,os,pathlib,io
import cv2,h5py,numpy as np,torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Resize
from lingbotvla.data.vla_data.utils import FeatureTransform

class CleanDataset(Dataset):
 def __init__(self,dataset_config,config,processor,use_depth_align=False,**kwargs):
  self.doc=json.loads(pathlib.Path(dataset_config.train_path).read_text())
  assert self.doc['setting']=='clean' and self.doc['embodiment']=='aloha-agilex'
  self.episodes=[e for t in self.doc['tasks'] for e in t['episodes']]
  self.starts=np.cumsum([0]+[e['samples'] for e in self.episodes]);self.num_frames=int(self.starts[-1]);self.num_episodes=len(self.episodes)
  self.task_names=[t['task'] for t in self.doc['tasks']]
  self.by_task=[[i for i,e in enumerate(self.episodes) if e['task']==name] for name in self.task_names]
  self.mode=os.environ.get('ROUND1_SAMPLING','frame');self.augment=os.environ.get('ROUND1_AUGMENT','0')=='1';self.epoch=0
  self.task_prob=None
  if self.mode=='task_weighted':
   spec=json.loads(pathlib.Path(os.environ['ROUND1_TASK_WEIGHTS']).read_text())
   weights=spec['weights'] if isinstance(spec,dict) and 'weights' in spec else spec
   raw=np.array([float(weights[name]) for name in self.task_names],dtype=np.float64)
   assert np.all(raw>0),raw
   self.task_prob=raw/raw.sum()
  self.chunk=dataset_config.chunk_size;self.future=dataset_config.use_future_image;self.resize=Resize((dataset_config.img_size,dataset_config.img_size));self.cache=collections.OrderedDict()
  self.feature_transform=FeatureTransform(str(pathlib.Path(dataset_config.robot_config_root)/'robotwin.yaml'),dataset_config,config,processor,False,True,chunk_size=self.chunk,image_augment=False,use_depth_align=use_depth_align,use_future_image=self.future)
 def __len__(self):return self.num_frames
 def set_epoch(self,epoch):self.epoch=epoch
 def locate(self,idx):
  if not 0<=idx<len(self):raise IndexError(idx)
  if self.mode=='task_weighted':
   rng=np.random.default_rng(np.random.SeedSequence([42,self.epoch,int(idx)]));ti=int(rng.choice(len(self.by_task),p=self.task_prob));ids=self.by_task[ti];ei=int(rng.choice(ids));t=int(rng.integers(0,self.episodes[ei]['samples']))
  elif self.mode=='task_episode':
   rng=np.random.default_rng(np.random.SeedSequence([42,self.epoch,int(idx)]));ids=self.by_task[idx%len(self.by_task)];ei=int(rng.choice(ids));t=int(rng.integers(self.episodes[ei]['samples']))
  else:ei=int(np.searchsorted(self.starts,idx,side='right')-1);t=int(idx-self.starts[ei])
  return ei,t
 def raw(self,idx):
  ei,t=self.locate(idx);e=self.episodes[ei]
  if ei not in self.cache:
   f=open(e['rgb_path'],'rb')
   with np.load(e['arrays_path']) as a:arrays={k:a[k] for k in a.files}
   self.cache[ei]=(f,arrays)
   if len(self.cache)>4:self.cache.popitem(last=False)[1][0].close()
  self.cache.move_to_end(ei);f,arrays=self.cache[ei];q=arrays['q'];n=len(q);ix=t+1+np.arange(self.chunk);pad=ix>=n;ix=np.minimum(ix,n-1)
  texts=e['instructions_seen'];text=texts[(idx+self.epoch)%len(texts)]
  state=torch.from_numpy(q[t].copy());action=torch.from_numpy(q[ix].copy())
  raw={'observation.state':state,'action':action,'action_is_pad':torch.from_numpy(pad),'task':text,'timestamp':torch.tensor(t*15/250),'frame_index':torch.tensor(t),'episode_index':torch.tensor(ei),'task_index':torch.tensor(ei//50)}
  brightness=contrast=saturation=1.
  if self.augment:
   brightness=float(torch.empty(()).uniform_(0.75,1.25));contrast=float(torch.empty(()).uniform_(0.75,1.25));saturation=float(torch.empty(()).uniform_(0.6,1.4))
   # Proprioception only. Action labels stay the recorded clean joints.
   raw['observation.state']=state+0.01*torch.randn_like(state)
  for ci,(camera,key) in enumerate([('head_camera','cam_high'),('left_camera','cam_left_wrist'),('right_camera','cam_right_wrist')]):
   images=[]
   for frame in ([t,min(t+self.chunk-1,n-1)] if self.future else [t]):
    payload=os.pread(f.fileno(),int(arrays['sizes'][ci,frame]),int(arrays['offsets'][ci,frame]))
    arr=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
    if arr is None:raise ValueError(f'JPEG decode failed {e["path"]} {camera} {frame}')
    # RoboTwin encodes RGB arrays directly with cv2.imencode; decoding restores RGB channel order.
    image=torch.from_numpy(arr.copy()).permute(2,0,1).float()/255
    if self.augment:image=self.colorize(image,brightness,contrast,saturation);image=self.occlude(image)
    images.append(self.resize(image))
   raw['observation.images.'+key]=torch.stack(images) if self.future else images[0]
  return raw,e,t
 def colorize(self,image,brightness,contrast,saturation):
  gray=image.mean(dim=0,keepdim=True)
  image=gray+(image-gray)*saturation
  image=((image-.5)*contrast+.5).mul(brightness)
  return image.clamp(0,1)
 def occlude(self,image):
  if torch.rand(())>0.5:return image
  _,h,w=image.shape
  rh=int(torch.randint(8,max(9,h//10),()).item());rw=int(torch.randint(8,max(9,w//10),()).item())
  y=int(torch.randint(0,h-rh+1,()).item());x=int(torch.randint(0,w-rw+1,()).item())
  image=image.clone();image[:,y:y+rh,x:x+rw]=image.mean(dim=(1,2),keepdim=True)
  return image
 def __getitem__(self,idx):
  raw,e,t=self.raw(idx)
  try:out=self.feature_transform.apply(raw)
  except Exception as exc:raise RuntimeError(f'{e["task"]}/episode{e["episode"]}/frame{t}') from exc
  out['rep_id']=e['task'];return out

def build_clean_dataset(dataset_config,model_config,config,processor,**kwargs):
 return CleanDataset(dataset_config,config,processor,**kwargs)
