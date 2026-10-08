#!/usr/bin/env python3
"""Reproducible pretrained baseline scheduler. No optimization or checkpoint updates."""
import argparse, concurrent.futures, datetime, fcntl, hashlib, json, os, pathlib, queue, re, shutil, signal, socket, subprocess, sys, threading, time
HERE=pathlib.Path(__file__).resolve().parent

def need(name):
    value=os.environ.get(name)
    if not value:
        raise SystemExit(f'Missing environment variable {name}')
    return value

MODEL=pathlib.Path(need('ROUND1_EVAL_MODEL'))
REPO=pathlib.Path(need('LINGBOT_REPO'))
SIM=pathlib.Path(need('ROBOTWIN_ROOT'))
IPY=need('VLA_TRAIN_PYTHON')
SPY=need('VLA_SIM_PYTHON')
ALL_TASKS=[line.strip() for line in (HERE/'tasks.txt').read_text().splitlines() if line.strip()]
assert len(ALL_TASKS)==50 and len(set(ALL_TASKS))==50
TASKS=os.environ.get('ROUND1_EVAL_TASKS',','.join(ALL_TASKS)).split(',')
assert set(TASKS)<=set(ALL_TASKS) and len(TASKS)==len(set(TASKS))

def py_site(python):
    hits=sorted((pathlib.Path(python).resolve().parent.parent/'lib').glob('python*/site-packages'))
    if not hits:
        raise SystemExit(f'No site-packages next to {python}')
    return hits[-1]
LOCK=threading.Lock()
CHILDREN=set()
STOP=threading.Event()
FATAL=threading.Event()
GPU_INFO={}
from gpu_health import inventory, probe, failure_kind

class GPUQuarantined(RuntimeError):
    pass

def quarantine(run,gpu,reason):
    with LOCK:
        p=run/'gpu_quarantine.json'
        data=json.loads(p.read_text()) if p.exists() else {}
        data[str(gpu)]={'time':time.time(),'reason':reason,'gpu':GPU_INFO.get(gpu,{})}
        atomic(p,data)


def atomic(path,obj):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False));temp.replace(path)

def digest(path):
    with open(path,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def terminate(p):
    if p.poll() is None:
        try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        except ProcessLookupError:pass
    with LOCK:CHILDREN.discard(p)

def spawn(cmd,env,cwd,log):
    f=open(log,'a')
    try:p=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    finally:f.close()
    with LOCK:CHILDREN.add(p)
    return p

def summarize(run):
    manifest=json.loads((run/'manifest.json').read_text())
    items=[]
    for setting in manifest['settings']:
        for task in manifest['tasks']:
            attempts=sorted((run/'evaluation'/setting/task).glob('attempt_*'))
            base=attempts[-1]/task if attempts else None
            records=[]
            if base and (base/'episodes.jsonl').exists():
                for line in (base/'episodes.jsonl').read_text().splitlines():
                    try:records.append(json.loads(line))
                    except json.JSONDecodeError:pass # writer may be mid-line
            result=json.loads((base/'result.json').read_text()) if base and (base/'result.json').exists() else None
            count=len(records);success=sum(x['success'] for x in records)
            item={'task':task,'setting':setting,'attempts':count,'successes':success,
                  'success_rate':success/count if count else None,
                  'complete':bool(result and count==manifest['episodes'] and result['attempts']==count and result['successes']==success),
                  'videos':sum(bool(x['video'] and pathlib.Path(x['video']).is_file()) for x in records),
                  'artifact_dir':str(base) if base else None}
            items.append(item)
    totals={}
    for setting in manifest['settings']:
        subset=[x for x in items if x['setting']==setting]
        n=sum(x['attempts'] for x in subset);s=sum(x['successes'] for x in subset)
        totals[setting]={'attempts':n,'successes':s,'success_rate':s/n if n else None,
                         'completed_tasks':sum(x['complete'] for x in subset),'expected_tasks':len(subset),
                         'expected_episodes':len(subset)*manifest['episodes'],'videos':sum(x['videos'] for x in subset)}
    result={'schema':'local_baseline_v1 (not official submission template)',
            'updated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'complete':all(x['complete'] for x in items), 'settings':totals,'tasks':items}
    atomic(run/'results.json',result)
    lines=['# Round1 checkpoint 仿真评测指标', '', '状态：'+('全部完成' if result['complete'] else '进行中或未完成；不是最终指标'), '',
           '| Setting | 完成回合 / 目标回合 | 成功数 | 已完成回合成功率 | 完成任务 | 视频数 |',
           '|---|---:|---:|---:|---:|---:|']
    for setting,v in totals.items():
        rate=f"{v['success_rate']:.2%}" if v['success_rate'] is not None else '未产生'
        lines.append(f"| {setting} | {v['attempts']} / {v['expected_episodes']} | {v['successes']} | {rate} | {v['completed_tasks']} / {v['expected_tasks']} | {v['videos']} |")
    lines += ['', '模型：'+manifest['model'] if 'model' in manifest else '',
              '详细逐任务指标：results.json；逐回合记录及视频：evaluation/<setting>/<task>/attempt_XX/<task>/',
              '架构、归一化来源和评测协议见 manifest.json 与 snapshot/。这是本地可复现基线，官方评测模板与渲染规范仍待核对。']
    report=run/'metrics.md';temp=run/'metrics.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(report)
    ref=os.environ.get('VLA_COMPARE_RUN')
    if ref and (pathlib.Path(ref)/'manifest.json').exists():
        from compare import compare_runs
        compare_runs(ref, run)
    return result

def environment(gpu,run):
    env=os.environ.copy()
    system_glx=pathlib.Path('/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0').exists()
    # The sysroot copies include Vulkan 1.3. When the driver GL stack is installed
    # under /usr, prepending sysroot makes OptiX create a black framebuffer.
    if system_glx:
        vk_icd='/usr/share/vulkan/icd.d/nvidia_icd.json'
        egl_icd='/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
        lib_prefix=''
    else:
        vk_icd=os.environ.get('VLA_VK_ICD','')
        egl_icd=os.environ.get('VLA_EGL_ICD','')
        lib_prefix=os.environ.get('VLA_GL_LIBDIR','')
        if lib_prefix and not lib_prefix.endswith(':'):
            lib_prefix+=':'
        if not vk_icd or not egl_icd:
            raise SystemExit('NVIDIA Vulkan/EGL vendor files are missing. Install the driver userspace stack or set VLA_VK_ICD and VLA_EGL_ICD.')
    env.update(CUDA_VISIBLE_DEVICES=GPU_INFO.get(gpu,{}).get('uuid',str(gpu)),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONNOUSERSITE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',
               HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME=os.environ.get('HF_HOME', str(pathlib.Path(need('VLA_MODEL_ROOT'))/'.cache')),
               QWEN3VL_PATH=need('VLA_TOKENIZER'),
               VK_ICD_FILENAMES=vk_icd,VK_DRIVER_FILES=vk_icd,SAPIEN_RT_DENOISER=os.environ.get('SAPIEN_RT_DENOISER','optix'),
               __EGL_VENDOR_LIBRARY_FILENAMES=egl_icd,
               LINGBOT_INFERENCE_CONFIG=str(run/'snapshot/inference_config.yaml'),
               SETUPTOOLS_SCM_PRETEND_VERSION='0.0.0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONHASHSEED='42')
    env.pop('VK_ADD_DRIVER_FILES', None)
    env['LD_LIBRARY_PATH']=lib_prefix+str(pathlib.Path(IPY).parent.parent/'lib')+':'+env.get('LD_LIBRARY_PATH','')
    env['PATH']=str(pathlib.Path(IPY).parent)+':'+env['PATH']
    env['PYTHONPATH']=str(REPO)
    return env

def wait_server(p,port):
    for _ in range(900):
        if STOP.is_set():raise RuntimeError('Interrupted')
        if p.poll() is not None:raise RuntimeError(f'Inference server exited {p.returncode}')
        try:
            from websockets.sync.client import connect
            with connect(f'ws://127.0.0.1:{port}',open_timeout=2,close_timeout=2) as ws:
                ws.recv(timeout=2)
                return
        except OSError:time.sleep(2)
    raise TimeoutError('Inference server not ready after 30 minutes')

def worker(gpu,jobs,run,args):
    port=args.port+gpu
    with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    env=environment(gpu,run)
    server=spawn([IPY,'-u',str(run/'snapshot/baseline_server.py'),'--model_path',str(MODEL),'--port',str(port),
                  '--use_bf16','False','--use_fp32','True','--use_compile','False','--use_length','50'],env,REPO,run/f'inference_gpu{gpu}.log')
    try:
        try:
            wait_server(server,port)
        except Exception as e:
            quarantine(run,gpu,f'Inference startup failed: {e}')
            raise GPUQuarantined(f'GPU {gpu}: inference startup failed') from e
        while not STOP.is_set():
            try:setting,task=jobs.get_nowait()
            except queue.Empty:break
            try:
                if GPU_INFO and inventory()[gpu]['uncorrected_ecc_volatile']:
                    jobs.put((setting,task))
                    quarantine(run,gpu,'Uncorrectable ECC detected before task dispatch')
                    raise GPUQuarantined(f'GPU {gpu}: ECC; job returned to queue')
                previous=sorted((run/'evaluation'/setting/task).glob('attempt_*'))
                first=1+max([int(x.name.split('_')[-1]) for x in previous] or [0])
                for attempt in range(first,first+5):
                    dest=run/'evaluation'/setting/task/f'attempt_{attempt:02d}'
                    dest.mkdir(parents=True,exist_ok=False)
                    predecessors=sorted((run/'evaluation'/setting/task).glob('attempt_*'))
                    predecessors=[x for x in predecessors if x!=dest]
                    if predecessors:
                        source=predecessors[-1]/task
                        target=dest/task;target.mkdir(exist_ok=True)
                        for filename in ['episodes.jsonl','pending_episode.json']:
                            if (source/filename).exists():shutil.copy2(source/filename,target/filename)
                        atomic(dest/'resumed_from.json',{'source':str(source),'episodes':len((target/'episodes.jsonl').read_text().splitlines()) if (target/'episodes.jsonl').exists() else 0})
                    senv=env.copy();senv['PATH']=str(pathlib.Path(SPY).parent)+':'+senv['PATH']
                    senv['PYTHONPATH']=str(py_site(SPY))+':'+str(SIM)
                    senv['ROBOTWIN_TEST_EPISODES']=str(args.episodes)
                    senv['LD_LIBRARY_PATH']=str(pathlib.Path(SPY).parent.parent/'lib')+':'+senv['LD_LIBRARY_PATH']
                    cmd=[SPY,'-u',str(run/'recovery_20260917/baseline_client.py'),'--config','policy/ACT/deploy_policy.yml','--overrides',
                         '--task_name',task,'--task_config',setting,'--train_config_name','0','--seed','0','--policy_name','ACT',
                         '--port',str(port),'--robo_name','robotwin','--video_fps','10','--eval_video_log','True','--output_dir',str(dest)]
                    atomic(dest/'command.json',{'argv':cmd,'gpu':gpu,'gpu_uuid':GPU_INFO.get(gpu,{}).get('uuid'),'started_at':time.time()})
                    p=spawn(cmd,senv,SIM,dest/'eval.log')
                    started=time.monotonic()
                    start_wall=time.time();dumped=False;stall_reason=None
                    while p.poll() is None and not STOP.is_set() and server.poll() is None and time.monotonic()-started<args.task_timeout:
                        heartbeat=dest/task/'heartbeat.json'
                        last=heartbeat.stat().st_mtime if heartbeat.exists() else start_wall
                        idle=time.time()-last
                        if idle>float(os.environ.get('EVAL_DUMP_AFTER_SECONDS','180')) and not dumped:
                            os.kill(p.pid,signal.SIGUSR1);dumped=True
                            atomic(dest/'stall_diagnostic.json',{'time':time.time(),'idle_seconds':idle,'pid':p.pid,
                                'heartbeat':json.loads(heartbeat.read_text()) if heartbeat.exists() else None})
                        if idle>float(os.environ.get('EVAL_STALL_AFTER_SECONDS','300')):
                            stall_reason=f'No simulator progress for {idle:.0f} seconds'
                            print(stall_reason,setting,task,flush=True)
                            break
                        if idle<60:dumped=False
                        time.sleep(2)
                    timed_out=p.poll() is None
                    if timed_out:terminate(p)
                    else:
                        with LOCK:CHILDREN.discard(p)
                    ok=p.returncode==0 and (dest/task/'result.json').exists()
                    atomic(dest/'exit.json',{'returncode':p.returncode,'stall_reason':stall_reason,'interrupted_or_timeout':timed_out,'result_present':ok,'seconds':time.monotonic()-started})
                    print(f'{setting}/{task} attempt={attempt} exit={p.returncode} complete={ok}',flush=True)
                    if ok or STOP.is_set():break
                    failure_text=(dest/'eval.log').read_text(errors='replace')
                    kind=failure_kind(failure_text)
                    if GPU_INFO and inventory()[gpu]['uncorrected_ecc_volatile']:
                        kind='gpu_fault'
                    if kind=='gpu_fault' or server.poll() is not None:
                        jobs.put((setting,task))
                        quarantine(run,gpu,f'{setting}/{task}: {kind or "server exited"}; inspect {dest}/eval.log')
                        raise GPUQuarantined(f'GPU {gpu} removed; incomplete task requeued')
                    if kind=='dependency_error':
                        jobs.put((setting,task));FATAL.set();STOP.set()
                        atomic(run/'infrastructure_error.json',{'task':task,'setting':setting,'gpu':gpu,'log':str(dest/'eval.log')})
                        raise RuntimeError('Dependency failure; stopped dispatch rather than consuming all task retries')
            finally:jobs.task_done()
    finally:terminate(server)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',type=pathlib.Path,required=True)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--num-tasks',type=int,default=50)
    p.add_argument('--gpus',type=int,default=5)
    p.add_argument('--gpu-ids',default=None,help='Physical nvidia-smi indices; ECC-faulted devices are always excluded')
    p.add_argument('--port',type=int,default=20330)
    p.add_argument('--task-timeout',type=int,default=43200)
    p.add_argument('--status',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();run=args.run_dir
    requested_gpu_ids=[int(x) for x in args.gpu_ids.split(',')] if args.gpu_ids else list(range(args.gpus))
    assert requested_gpu_ids and len(requested_gpu_ids)==len(set(requested_gpu_ids))
    if args.status:
        # Read-only when scheduler owns the atomic writer.
        print((run/'results.json').read_text());return
    assert 1<=args.num_tasks<=50 and args.episodes>=1 and 1<=args.gpus<=5
    lock=open(HERE/('evaluation.gpus-' + '-'.join(map(str,requested_gpu_ids)) + '.port-' + str(args.port) + '.lock'),'w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(parents=True,exist_ok=True)
    if args.resume:
        manifest=json.loads((run/'manifest.json').read_text())
        args.episodes=manifest['episodes'];args.num_tasks=len(manifest['tasks'])
        args.gpus=manifest['gpus']
    else:
        if (run/'manifest.json').exists():raise RuntimeError('Run already exists; use --resume or a new directory')
        required=[HERE/'clean_norm_stats.json',HERE/'clean_stats_provenance.json']
        assert all(x.exists() for x in required),'clean normalization files are missing from the eval directory'
        provenance=json.loads(required[1].read_text())
        assert provenance['episodes']==2500 and {x['task'] for x in provenance['tasks']}==set(ALL_TASKS)
        snap=run/'snapshot';snap.mkdir()
        for name in ['baseline_server.py','baseline_client.py','inference_config.yaml','clean_norm_stats.json','clean_stats_provenance.json','robotwin.yaml','evaluate.py','compare.py']:
            shutil.copy2(HERE/name,snap/name)
        # Make the per-run stats immutable by pathname (never reads subsequent recomputation).
        cfg=(snap/'inference_config.yaml').read_text().replace('__NORM_STATS__',str(snap/'clean_norm_stats.json')).replace('__TOKENIZER__',need('VLA_TOKENIZER')).replace('__CHECKPOINT__',str(MODEL))
        (snap/'inference_config.yaml').write_text(cfg)
        for name in ['demo_clean.yml','demo_randomized.yml','_eval_step_limit.yml','_camera_config.yml']:
            shutil.copy2(SIM/'task_config'/name,snap/name)
        manifest={'model':str(MODEL),'model_revision':'local-clean-training-checkpoint',
                  'model_index_sha256':digest(MODEL/'model.safetensors.index.json'),
                  'model_weights_updated':True,'training_provenance':'Clean-only training from the raw pretrained checkpoint', 'comparison_baseline':os.environ.get('VLA_COMPARE_RUN',''),'normalization':'Aloha clean 50x50 only; frame-weighted absolute joint quantiles',
                  'architecture_config_source':'LingBot-VLA architecture; weights loaded from the checkpoint passed at launch',
                  'episodes':args.episodes,'tasks':TASKS[:args.num_tasks],'settings':os.environ.get('ROUND1_EVAL_SETTINGS','demo_clean').split(','),
                  'gpus':args.gpus,'pid':os.getpid(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'seed':0,'episode_seed_start':100000,'instruction_rng':'Python and NumPy reset to scene seed before generating/selecting instruction','inference_rng':'Reset per episode to environment seed',
                  'precision':'fp32','compile':False,'action_chunk':50,'execute_length':int(os.environ.get('ROUND1_EXECUTE_LENGTH','50')),'instruction_type':'unseen','denoiser':'optix',
                  'video':'head camera, per action frame, 10 fps, H264; ALL successful and failed completed episodes',
                  'robotwin_revision':'13c3c47ff4312dd62484bcd51be034af55c062d1',
                  'official_compliance':'Pending official template/rendering/seed confirmation; local reproducible baseline',
                  'snapshot_sha256':{x.name:digest(x) for x in snap.iterdir() if x.is_file()}}
        atomic(run/'manifest.json',manifest)
    recovery=run/'recovery_20260917'
    recovery.mkdir(exist_ok=True)
    if not (recovery/'baseline_client.py').exists():
        shutil.copy2(HERE/'client.py',recovery/'baseline_client.py')
    (run/'scheduler.pid').write_text(str(os.getpid())+'\n')
    def stop(sig,frame):STOP.set()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    jobs=queue.Queue()
    # Interleave settings so live summaries cover both as early as possible.
    existing=summarize(run)
    completed={(x['setting'],x['task']) for x in existing['tasks'] if x['complete']}
    for task in manifest['tasks']:
        for setting in manifest['settings']:
            if (setting,task) not in completed:jobs.put((setting,task))
    errors=[]
    global GPU_INFO
    GPU_INFO=inventory()
    admission=[];gpu_ids=[]
    atomic(run/'state.json',{'status':'preflight','pid':os.getpid()})
    for gpu in requested_gpu_ids:
        if gpu not in GPU_INFO:raise ValueError(f'GPU {gpu} does not exist')
        if GPU_INFO[gpu]['uncorrected_ecc_volatile']:
            quarantine(run,gpu,'Excluded at preflight: uncorrectable volatile ECC')
            admission.append({'gpu':gpu,'ok':False,'reason':'uncorrectable volatile ECC','identity':GPU_INFO[gpu]})
            continue
        env=environment(gpu,run)
        env['PATH']=str(pathlib.Path(SPY).parent)+':'+env['PATH']
        env['PYTHONPATH']=str(py_site(SPY))+':'+str(SIM)
        env['LD_LIBRARY_PATH']=str(pathlib.Path(SPY).parent.parent/'lib')+':'+env['LD_LIBRARY_PATH']
        result=probe(gpu,SPY,env,SIM);admission.append(result)
        if result['ok']:gpu_ids.append(gpu)
        else:quarantine(run,gpu,'CUDA/cuRobo preflight failed: '+str(result))
    receipt={'time':time.time(),'requested_gpu_ids':requested_gpu_ids,'active_gpu_ids':gpu_ids,
             'checks':admission,'remaining_task_settings':jobs.qsize(),'preserved_episodes':sum(x['attempts'] for x in existing['tasks']),
             'preserved_complete_task_settings':len(completed)}
    atomic(run/'gpu_admission.json',receipt)
    if not gpu_ids:
        atomic(run/'state.json',{'status':'blocked','pid':os.getpid(),'errors':['No healthy CUDA/cuRobo devices'],'gpu_admission':receipt})
        return 1
    args.gpus=len(gpu_ids)
    atomic(run/'state.json',{'status':'running','pid':os.getpid(),'gpu_ids':gpu_ids})
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.gpus) as pool:
        futures=[pool.submit(worker,gpu,jobs,run,args) for gpu in gpu_ids[:min(args.gpus,jobs.qsize())]]
        while not all(f.done() for f in futures):
            result=summarize(run)
            print(json.dumps(result['settings']),flush=True)
            time.sleep(15)
        for f in futures:
            try:f.result()
            except Exception as e:errors.append(repr(e))
    result=summarize(run)
    state={'status':'complete' if result['complete'] else 'blocked' if FATAL.is_set() else 'interrupted' if STOP.is_set() else 'incomplete',
           'errors':errors,'finished_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic(run/'state.json',state);print(json.dumps(state),flush=True)
    return 0 if state['status']=='complete' else 1

if __name__=='__main__':sys.exit(main())
