import json,os,pathlib,time,math
import torch

def apply_backbone_lr(optimizer,model):
 scale=float(os.environ.get('ROUND1_BACKBONE_LR_SCALE','1'))
 names={id(p):n for n,p in model.named_parameters()};summary=[]
 for opt in getattr(optimizer,'optimizers',[optimizer]):
  groups=[]
  for g in opt.param_groups:
   bins={True:[],False:[]}
   for p in g['params']:bins['.qwenvl.' in names[id(p)]].append(p)
   for backbone,params in bins.items():
    if not params:continue
    h=dict(g);h['params']=params;h['lr']=g['lr']*(scale if backbone else 1);groups.append(h)
    summary.append({'backbone':backbone,'lr':h['lr'],'tensors':len(params),'numel':sum(p.numel() for p in params)})
  opt.param_groups=groups
 expert_only=os.environ.get('ROUND1_TRAIN_EXPERT_ONLY','0') in ('1','true','True')
 if not any(x['backbone'] for x in summary) and not expert_only:
  raise RuntimeError('No VLM backbone parameters matched')
 if int(os.environ.get('RANK','0'))==0:
  p=pathlib.Path(os.environ['ROUND1_RUN_DIR'])/'optimizer_groups.json';p.write_text(json.dumps(summary,indent=2))

def log_step(args,step,**values):
 if args.train.global_rank!=0:return
 row={'time':time.time(),'step':int(step),'pid':os.getpid(),'global_batch_size':args.train.global_batch_size}
 for k,v in values.items():
  if torch.is_tensor(v):v=v.detach().float().item()
  row[k]=float(v)
 row['samples_per_second']=args.train.global_batch_size/row['step_seconds'] if row['step_seconds']>0 else None
 row['cuda_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
 row['cuda_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
 root=pathlib.Path(args.train.output_dir)
 with (root/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 p=root/'heartbeat.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(p)
 if not math.isfinite(row['loss']):raise FloatingPointError('Non-finite training loss; see steps.jsonl')


def log_phase(args,step,micro,phase):
 root=pathlib.Path(args.train.output_dir)
 row={'time':time.time(),'rank':args.train.global_rank,'pid':os.getpid(),'step':int(step),'micro':int(micro),'phase':phase}
 p=root/f'phase_rank{args.train.global_rank}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(row));tmp.replace(p)
