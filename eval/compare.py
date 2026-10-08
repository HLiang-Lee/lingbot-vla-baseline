"""Compare completed episodes on matching tasks, seeds and instructions only."""
import hashlib,json,pathlib

def write(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False));tmp.replace(path)

def episodes(run,setting,task):
    attempts=sorted((run/'evaluation'/setting/task).glob('attempt_*'))
    if not attempts:return {}
    p=attempts[-1]/task/'episodes.jsonl'
    if not p.exists():return {}
    result={}
    for line in p.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:continue
        seed=r['seed']
        if seed in result:raise ValueError(f'Duplicate seed {seed} in {p}')
        result[seed]=r
    return result

def compare_runs(baseline,official):
    baseline=pathlib.Path(baseline);official=pathlib.Path(official)
    a=json.loads((baseline/'manifest.json').read_text());b=json.loads((official/'manifest.json').read_text())
    fields=['seed','episode_seed_start','inference_rng','precision','compile','action_chunk','instruction_type','denoiser','robotwin_revision']
    mismatches=[k for k in fields if a.get(k)!=b.get(k)]
    for name in ['clean_norm_stats.json','robotwin.yaml','baseline_server.py','baseline_client.py','demo_clean.yml','demo_randomized.yml','_eval_step_limit.yml','_camera_config.yml']:
        x=baseline/'snapshot'/name;y=official/'snapshot'/name
        if hashlib.sha256(x.read_bytes()).digest()!=hashlib.sha256(y.read_bytes()).digest():mismatches.append(name)
    rows=[];totals={}
    for setting in b['settings']:
        total={'paired_episodes':0,'baseline_successes':0,'official50k_successes':0,'instruction_mismatches':0}
        for task in b['tasks']:
            ar=episodes(baseline,setting,task);br=episodes(official,setting,task)
            common=sorted(ar.keys() & br.keys())
            valid=[s for s in common if ar[s]['instruction']==br[s]['instruction']]
            n=len(valid);sa=sum(bool(ar[s]['success']) for s in valid);sb=sum(bool(br[s]['success']) for s in valid)
            row={'setting':setting,'task':task,'paired_episodes':n,'baseline_successes':sa,'official50k_successes':sb,
                 'baseline_rate':sa/n if n else None,'official50k_rate':sb/n if n else None,
                 'delta_percentage_points':100*(sb-sa)/n if n else None,
                 'instruction_mismatches':len(common)-n,'paired_seeds':valid}
            rows.append(row)
            for key in total:total[key]+=row[key]
        n=total['paired_episodes']
        total.update(baseline_rate=total['baseline_successes']/n if n else None,official50k_rate=total['official50k_successes']/n if n else None,
                     delta_percentage_points=100*(total['official50k_successes']-total['baseline_successes'])/n if n else None)
        totals[setting]=total
    ar=json.loads((baseline/'results.json').read_text());br=json.loads((official/'results.json').read_text())
    complete=ar['complete'] and br['complete'] and a['tasks']==b['tasks'] and a['episodes']==b['episodes'] and not mismatches and all(v['paired_episodes']==len(b['tasks'])*b['episodes'] for v in totals.values())
    report={'baseline':str(baseline),'official50k':str(official),'protocol_matches':not mismatches,'protocol_mismatches':mismatches,
            'complete':complete,'settings':totals,'tasks':rows,
            'training_target':'Once both full runs complete, match or exceed official50k success rate in each setting; also inspect per-task gaps. Before completion all rates are provisional.',
            'caveat':'Official50k is an upstream mixed-training reference evaluated with the SAME clean-only norm as baseline, not a clean-only eligible checkpoint or reproduction using its original norm.'}
    write(official/'comparison.json',report)
    lines=['# Baseline 与官方 50k 配对对比','', '状态：'+('完整对比' if complete else '阶段性对比，不能当作最终训练门槛'),
           '协议核对：'+('一致' if not mismatches else '不一致：'+', '.join(mismatches)), '',
           '| Setting | 配对回合 | Baseline 成功率 | 官方 50k 成功率 | 差值（百分点） |', '|---|---:|---:|---:|---:|']
    fmt=lambda v:'—' if v is None else f'{v:.2%}'
    for setting,v in totals.items():
        delta='—' if v['delta_percentage_points'] is None else f"{v['delta_percentage_points']:+.2f}"
        lines.append(f"| {setting} | {v['paired_episodes']} | {fmt(v['baseline_rate'])} | {fmt(v['official50k_rate'])} | {delta} |")
    lines+=['','训练目标：完整评测后，clean 与 randomized 分别达到或超过本次官方 50k 成功率，并检查逐任务差距。','仅匹配相同任务、种子、指令的已完成回合；不拿两边不同进度的总体成功率直接相减。','官方 50k 使用与 baseline 相同的 clean-only 归一化；它是混合数据训练的参考权重，不是 clean-only 合规参赛权重。', '',
            '| Setting / Task | 配对回合 | Baseline 成功数 | 官方 50k 成功数 |','|---|---:|---:|---:|']
    for x in rows:lines.append(f"| {x['setting']} / {x['task']} | {x['paired_episodes']} | {x['baseline_successes']} | {x['official50k_successes']} |")
    tmp=official/'comparison.md.tmp';tmp.write_text('\n'.join(lines)+'\n');tmp.replace(official/'comparison.md')
    return report


def compare_by_seed(baseline,official):
    """Same protocol with upstream stochastic paraphrases; do not imply identical inputs."""
    baseline=pathlib.Path(baseline);official=pathlib.Path(official)
    a=json.loads((baseline/'manifest.json').read_text());b=json.loads((official/'manifest.json').read_text())
    ar=json.loads((baseline/'results.json').read_text());br=json.loads((official/'results.json').read_text())
    mismatches=[]
    for key in ['seed','episode_seed_start','inference_rng','precision','compile','action_chunk','instruction_type','denoiser','robotwin_revision']:
        if a.get(key)!=b.get(key):mismatches.append(key)
    for name in ['clean_norm_stats.json','robotwin.yaml','baseline_server.py','baseline_client.py','demo_clean.yml','demo_randomized.yml','_eval_step_limit.yml','_camera_config.yml']:
        if (baseline/'snapshot'/name).read_bytes()!=(official/'snapshot'/name).read_bytes():mismatches.append(name)
    rows=[];totals={}
    for setting in b['settings']:
        total=dict(paired_episodes=0,baseline_successes=0,official50k_successes=0,identical_instruction_episodes=0,
                   identical_instruction_baseline_successes=0,identical_instruction_official50k_successes=0)
        for task in b['tasks']:
            x=episodes(baseline,setting,task);y=episodes(official,setting,task);seeds=sorted(x.keys() & y.keys())
            identical=[s for s in seeds if x[s]['instruction']==y[s]['instruction']]
            row={'task':task,'setting':setting,'paired_episodes':len(seeds),
                 'baseline_successes':sum(bool(x[s]['success']) for s in seeds),'official50k_successes':sum(bool(y[s]['success']) for s in seeds),
                 'identical_instruction_episodes':len(identical),'identical_instruction_baseline_successes':sum(bool(x[s]['success']) for s in identical),
                 'identical_instruction_official50k_successes':sum(bool(y[s]['success']) for s in identical),'paired_seeds':seeds}
            for key in total:total[key]+=row[key]
            rows.append(row)
        n=total['paired_episodes'];total.update(baseline_rate=total['baseline_successes']/n if n else None,
            official50k_rate=total['official50k_successes']/n if n else None,
            delta_percentage_points=100*(total['official50k_successes']-total['baseline_successes'])/n if n else None)
        totals[setting]=total
    complete=bool(ar['complete'] and br['complete'] and not mismatches and a['tasks']==b['tasks'] and a['episodes']==b['episodes'])
    result={'complete':complete,'protocol_matches':not mismatches,'protocol_mismatches':mismatches,
        'pairing':'Same task and environment seed; language paraphrases can differ because upstream Python random is unseeded. Same seed does not prove bitwise-identical scenes. Identical-instruction subset is reported separately.',
        'baseline':str(baseline),'official50k':str(official),'settings':totals,'tasks':rows}
    write(official/'comparison_by_seed.json',result)
    targets={'reference':str(official),'complete':br['complete'],'settings':{},'tasks':[],
        'rule':'Meet or exceed official50k successes out of 5000 in clean and randomized separately; task values are per-task references, not mandatory competition thresholds. Null means not yet established.',
        'caveat':'Same clean-only normalization; official weights were trained on mixed data. Unseeded upstream language variation remains.'}
    for setting,v in br['settings'].items():
        ready=v['completed_tasks']==v['expected_tasks'] and v['attempts']==v['expected_episodes']
        targets['settings'][setting]={'complete':ready,'evaluation_episodes':v['expected_episodes'],
            'minimum_successes':v['successes'] if ready else None,'minimum_success_rate':v['success_rate'] if ready else None}
    for t in br['tasks']:
        targets['tasks'].append({'task':t['task'],'setting':t['setting'],'complete':t['complete'],
            'reference_successes':t['successes'] if t['complete'] else None,'evaluation_episodes':b['episodes']})
    write(official/'training_targets.json',targets)
    lines=['# Baseline 与官方 50k：同任务、同种子对比','', '状态：'+('全量已完成' if complete else '阶段性结果，不是最终训练门槛'),
        '协议核对：'+('一致' if not mismatches else '不一致：'+str(mismatches)),
        '上游 Python random 未固定，指令表述可能不同；同种子不承诺场景逐位一致。这是同评测协议的统计对比，不是完全相同输入的模型消融。','',
        '| Setting | 配对回合 | Baseline 成功数 | 官方 50k 成功数 | 成功率差（百分点） | 同指令子集：Baseline / 官方 |',
        '|---|---:|---:|---:|---:|---|']
    for setting,t in totals.items():
        d='—' if t['delta_percentage_points'] is None else f"{t['delta_percentage_points']:+.2f}"
        n=t['identical_instruction_episodes']
        lines.append(f"| {setting} | {t['paired_episodes']} | {t['baseline_successes']} | {t['official50k_successes']} | {d} | {t['identical_instruction_baseline_successes']}/{n} / {t['identical_instruction_official50k_successes']}/{n} |")
    lines+=['','训练目标：最终 clean、randomized 分别达到或超过官方 50k 的成功数 / 5000。完整 setting 完成后自动写入 training_targets.json；当前未完成项为 null。',
        '每个任务参考成功数 / 100 见 training_targets.json；逐任务配对差异见 comparison_by_seed.json。',
        '官方 50k 与 baseline 使用同一 clean-only 归一化。本次目标是同协议实测表现，不是论文分数复现，也不是 clean-only 合规权重。']
    p=official/'comparison_by_seed.md';tmp=p.with_suffix('.md.tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.replace(p)
    return result
