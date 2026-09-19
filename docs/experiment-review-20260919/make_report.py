"""Deterministic CPU-only report builder; emits path/content JSON for apply_patch."""
import collections
import base64
import csv
import gzip
import io
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ARMS=('clean','uniform_matched_q','selected_recipe')
GROUPS=('Clean','FMT','ANS','EVD','PARA')
MODELS=['Qwen/Qwen3-0.6B','Qwen/Qwen3-1.7B','Qwen/Qwen3-4B','Qwen/Qwen3-8B','Qwen/Qwen2.5-7B-Instruct','Qwen/Qwen3.5-2B','Qwen/Qwen3.5-9B','meta-llama/Llama-3.1-8B-Instruct','mistralai/Mistral-7B-Instruct-v0.3','allenai/Olmo-3-7B-Instruct','google/gemma-4-12B-it']
GEN={'Math':{'Clean':('original','acc_loose'),'FMT':('format','main'),'ANS':('insufficient','insufficient_stop'),'EVD':('distractor','acc_loose'),'PARA':('paraphrase','acc_loose')},
     'Knowledge':{'Clean':('original','acc_strict'),'FMT':('format','main'),'ANS':('insufficient','insufficient_stop'),'EVD':('distractor','acc_strict')}}

def short(m):return m.split('/')[-1]
def clean_share(r):return r['shares'].get('Clean',1-sum(r['shares'].get(g,0) for g in GROUPS[1:]))
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in r)+' |' for r in rows])+'\n'
def fmt(values,scale=1,places=3):
    values=list(values)
    if not values:return '—'
    mean=st.mean(values)*scale
    return f'{mean:.{places}f}'+(f' ± {st.stdev(values)*scale:.{places}f}' if len(values)>1 else '')+f' [{len(values)}]'
def sign(v,places=3):return f'{v:+.{places}f}'
def get(r,metric,group='Clean',domain='Math'):
    if metric in ('repair_rel_nll','clean_rel_nll'):return r.get(metric)
    if metric in ('nll','bpb'):
        return r.get('nll',{}).get(group,{}).get('mean_per_item_nll' if metric=='nll' else 'bits_per_byte')
    if metric=='macro_nll':
        gs=GROUPS if domain=='Math' else GROUPS[:4]
        vals=[get(r,'nll',g,domain) for g in gs]
        return st.mean(vals) if all(v is not None for v in vals) else None
    if metric=='strict_clean':return r.get('generation',{}).get('original',{}).get('acc_exact' if domain=='Math' else 'acc_strict')
    if metric=='false_abstain':return r.get('generation',{}).get('suff_ctr',{}).get('false_abstain')
    cond,key=GEN[domain][group]
    return r.get('generation',{}).get(cond,{}).get(key)

def by_seed(rows,model,arm,metric,group='Clean',domain='Math'):
    out={}
    for r in rows:
        if r['model']!=model or r['arm']!=arm:continue
        v=get(r,metric,group,domain)
        if v is None:continue
        if r['seed'] in out:raise ValueError('duplicate model/arm/seed')
        assert math.isfinite(v)
        out[r['seed']]=v
    return out

def comparison(rows,models,metric,group='Clean',domain='Math'):
    result=[];verdicts=[];percentage=metric in ('gen','strict_clean','false_abstain','repair_rel_nll','clean_rel_nll');scale=100 if percentage else 1
    for m in models:
        vals=[by_seed(rows,m,a,metric,group,domain) for a in ARMS]
        paired=sorted(vals[1].keys()&vals[2].keys());diffs=[(vals[2][s]-vals[1][s])*scale for s in paired]
        delta=sign(st.mean(diffs),2 if percentage else 4) if diffs else '—'
        status='完整配对' if len(paired)==3 else f'部分配对 {len(paired)}/3' if paired else '缺匹配对照/读数'
        result.append([short(m),*[fmt(v.values(),scale,2 if percentage else 3) for v in vals],delta,str(len(paired)),status])
        if len(paired)==3:verdicts.append((m,st.mean(diffs)))
    return table(['模型','Clean-SFT','Uniform','Ours','Δ Ours−Uniform'+(' (pp)' if percentage else ''),'O/U 配对 seeds','O/U 状态'],result),verdicts

def add_objective(rows):
    clean={(r['model'],r['seed']):r for r in rows if r['arm']=='clean'}
    for r in rows:
        baseline=clean.get((r['model'],r['seed']))
        if not baseline:continue
        r['repair_rel_nll']=st.mean(get(r,'nll',g)/get(baseline,'nll',g)-1 for g in GROUPS[1:])
        r['clean_rel_nll']=get(r,'nll','Clean')/get(baseline,'nll','Clean')-1

def paired_delta(rows,model,metric,domain='Math'):
    a,b=[by_seed(rows,model,arm,metric,domain=domain) for arm in ARMS[1:]]
    seeds=sorted(a.keys()&b.keys())
    return (st.mean(b[z]-a[z] for z in seeds) if seeds else None),len(seeds)

def csv_text(rows):
    if not rows:return ''
    out=io.StringIO();keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(out,keys,lineterminator='\n');w.writeheader();w.writerows(rows);return out.getvalue()

def read_input(name):
    path=HERE/'data'/name
    if path.exists():return path.read_text()
    return gzip.decompress(base64.b64decode(path.with_name(path.name+'.gz.b64').read_text())).decode()

def build():
    data=json.loads(read_input('readouts.json'));history=json.loads(read_input('historical.json'))
    forecasts=list(csv.DictReader(io.StringIO(read_input('forecast_errors.csv'))))
    budgets=list(csv.DictReader(io.StringIO(read_input('calibration_budget_errors.csv'))))
    files={};volume=data['volume'];add_objective(volume);models=[m for m in MODELS if any(r['model']==m for r in volume)]
    intro='数值为训练 seed 均值 ± 样本标准差；方括号为可用 seed 数。Δ 只使用 Ours 与 Uniform 都存在的相同 seed，不能直接减两列的非配对均值。SD 不是置信区间；不作显著性声明。所有结果均为已知 DEV，非盲测或 sealed final。\n\n'
    main=['# 固定 N=3,968：实际 BMK 与损失三方对照','',intro,'固定 31 updates；这批 Ours 来自 9 月 14–15 日的旧选择器，**不是** 19 日剂量曲线重新选择的配方。Uniform 与 Ours 的总 repair token 比例相同。\n',
          '## 1. Clean BMK 答案准确率 ↑（%，宽松答案提取）','']
    clean_table,clean_v=comparison(volume,models,'gen');main.append(clean_table)
    main+=['','## 2. 明确答案格式下的准确率 ↑（%，original.acc_exact）','',comparison(volume,models,'strict_clean')[0],
           '\n## 3. 五组 reference NLL 等权均值 ↓（nats/token，描述性汇总）','',comparison(volume,models,'macro_nll')[0],
           '\n此等权均值不是原选择器的优化目标；原目标是约束 Clean 不退化后，最小化四项 repair 相对 Clean-SFT 的平均 NLL。\n',
           '### 原目标口径：四项 repair 相对 Clean-SFT 的 NLL 变化 ↓（%）','',comparison(volume,models,'repair_rel_nll')[0],
           '每个 seed 先计算 mean(NLL_arm / NLL_Clean-SFT − 1)，再跨 seed 汇总；必须具有相同 seed 的 Clean-SFT 读数。Qwen2.5 缺 seed 71 的 Clean，因此此表只比较两个 seed。\n',
           '### Clean 约束的实际读数：相对 Clean-SFT 的 Clean NLL 变化 ↓（%）','',comparison(volume,models,'clean_rel_nll')[0],
           '## 4. 各评测组：准确率、NLL、BPB','']
    for m in models:
        main += [f'<details><summary><b>{short(m)}</b>：全部端点与误拒答</summary>','',table(['评测组 / 指标','Clean-SFT','Uniform','Ours','Δ Ours−Uniform','配对 seeds'],[
            [g+' / '+label,*[fmt(by_seed(volume,m,a,k,g).values(),scale,2 if scale==100 else 3) for a in ARMS],
             (sign(st.mean((by_seed(volume,m,'selected_recipe',k,g)[s]-by_seed(volume,m,'uniform_matched_q',k,g)[s])*scale for s in sorted(by_seed(volume,m,'selected_recipe',k,g).keys()&by_seed(volume,m,'uniform_matched_q',k,g).keys())),2 if scale==100 else 4) if by_seed(volume,m,'selected_recipe',k,g).keys()&by_seed(volume,m,'uniform_matched_q',k,g).keys() else '—'),
             len(by_seed(volume,m,'selected_recipe',k,g).keys()&by_seed(volume,m,'uniform_matched_q',k,g).keys())]
            for g in GROUPS for k,label,scale in [('gen','BMK % ↑',100),('nll','NLL ↓',1),('bpb','BPB ↓',1)]]),
            '**误拒答率 ↓（suff_ctr.false_abstain）**：'+ ' / '.join(fmt(by_seed(volume,m,a,'false_abstain').values(),100,2) for a in ARMS),
            '\n顺序为 Clean-SFT / Uniform / Ours。','</details>','']
    main += ['## 5. 配方及未完整项','',table(['模型','Clean / FMT / ANS / EVD / PARA（token %）'],[[short(m),' / '.join(f'{next(r for r in volume if r["model"]==m and r["arm"]=="selected_recipe")["shares"].get(g,0)*100:g}' for g in GROUPS)] for m in models]),'',table(['模型','arm','seed','原因'],[[short(r['model']),r['arm'],r['seed'],r['reason']] for r in data['failures']])]
    files['VOLUME_3968.md']='\n'.join(main)+'\n'

    # Independent fixed-18-update panel. Do not substitute a different repair budget.
    selections={s['model']:s for s in data['selections']};fixed=[];unmatched=[]
    for r in data['fixed']:
        r=dict(r)
        if r['arm']=='uniform_reference':
            recipe=selections.get(r['model'],{}).get('recipe')
            if recipe is None or abs(clean_share(r)-recipe['Clean'])>1e-8:
                unmatched.append(r['model']);continue
            r['arm']='uniform_matched_q'
        fixed.append(r)
    add_objective(fixed)
    knowledge=[]
    for r in fixed:
        k=data['knowledge'].get(r['checkpoint_sha256'])
        if k:
            assert k['model']==r['model'] and k['seed']==r['seed']
            knowledge.append({**k,'arm':r['arm']})
    f=['# 固定原校准条件：11 模型 × Math / grounded Knowledge','',intro,
       '这组保持原 18 updates / 2,304 draws，与 N=3,968 面板分开，不能合并计算胜率。Knowledge 是**同一 Math-trained checkpoint 的跨域读数**，不是 Knowledge-domain 训练。按 checkpoint tree hash 精确绑定。',
       '\n原校准 uniform 总 repair 预算不匹配的模型：'+', '.join(short(m) for m in sorted(set(unmatched)))+'；不把它冒充 matched-uniform。',
       '\nQwen2.5 的 selected 是原校准 checkpoint 精确复用，属于开发内比较，不是独立确认。Knowledge 无 PARA；四组宏均值与 Math 五组宏均值不直接横比。\n']
    for domain,records in [('Math',fixed),('Knowledge',knowledge)]:
        f += [f'## {domain}：Clean BMK ↑（%）','',comparison(records,MODELS,'gen',domain=domain)[0],
              f'## {domain}：主评测组 NLL 等权均值 ↓（描述性）','',comparison(records,MODELS,'macro_nll',domain=domain)[0]]
        if domain=='Math':f += ['## Math：原四项 repair 相对 NLL 目标 ↓（%）','',comparison(records,MODELS,'repair_rel_nll')[0],
                                '## Math：Clean NLL 相对 Clean-SFT 变化 ↓（%）','',comparison(records,MODELS,'clean_rel_nll')[0]]
        for g in GEN[domain]:
            f += [f'<details><summary>{domain} / {g}：BMK、NLL、BPB 三方明细</summary>','']
            for metric,label in [('gen','BMK % ↑'),('nll','NLL ↓'),('bpb','BPB ↓')]:f += [f'### {label}','',comparison(records,MODELS,metric,g,domain)[0]]
            f += ['</details>','']
    files['FIXED_CHECKPOINTS.md']='\n'.join(f)+'\n'

    p=['# 对 benchmark loss 的预测：与实际答题准确率分开','',
       '所有误差均使用完整三个 seed 的实际均值；一个 cell = 模型 × repair 训练轴 × 评测组 × 目标剂量。每次比较严格使用预测都有效的相同 cells。更小更好。目标是已知 DEV 模型的 held-out dose，不是新模型盲测。\n',
       '## 1. 冻结饱和响应模型 vs 最近校准点','']
    indexed=collections.defaultdict(dict)
    for r in forecasts:
        if r['primary_comparison_eligible']=='True' and r['absolute_error']:
            key=tuple(r[k] for k in ('model','repair_axis','endpoint','metric','target_q'));indexed[key][r['predictor']]=float(r['absolute_error'])
    primary=[]
    for metric in ('mean_per_item_nll','bits_per_byte'):
        for model in ['ALL']+MODELS:
            pairs=[v for k,v in indexed.items() if k[3]==metric and (model=='ALL' or k[0]==model) and {'fixed_tau_saturating','nearest_q10'}<=v.keys()]
            if not pairs:continue
            a=st.mean(v['fixed_tau_saturating'] for v in pairs);b=st.mean(v['nearest_q10'] for v in pairs)
            primary.append([metric,'全部 cells 加权' if model=='ALL' else short(model),len(pairs),f'{a:.5f}',f'{b:.5f}',sign(a-b,5),'饱和模型' if a<b else '最近校准点' if a>b else '相同'])
    p.append(table(['指标','模型','cells','饱和 MAE','最近点 MAE','误差差值','均值较小者'],primary))
    dose_rows=[]
    for metric in ('mean_per_item_nll','bits_per_byte'):
        for dose in ('0.2','0.4','0.8'):
            pairs=[v for k,v in indexed.items() if k[3]==metric and k[4]==dose and {'fixed_tau_saturating','nearest_q10'}<=v.keys()]
            a=st.mean(v['fixed_tau_saturating'] for v in pairs);b=st.mean(v['nearest_q10'] for v in pairs)
            dose_rows.append([metric,f'{float(dose)*100:g}%',len(pairs),f'{a:.5f}',f'{b:.5f}',sign(a-b,5)])
    p += ['\n### 按目标 repair 剂量拆分','',table(['指标','repair token 比例','cells','饱和 MAE','最近点 MAE','误差差值'],dose_rows),
          '20% 覆盖 11 个模型；40% / 80% 只覆盖两个 anchor。因此上方总体是 cell 加权，不是 11 模型等权。\n']
    p += ['\n## 2. 其他预测基线：共同有效 cell 交集','']
    allrows=[]
    methods=('fixed_tau_saturating','nearest_q10','log_linear','raw_linear')
    for metric in ('mean_per_item_nll','bits_per_byte'):
        common=[v for k,v in indexed.items() if k[3]==metric and set(methods)<=v.keys()]
        for method in methods:allrows.append([metric,method,len(common),f'{st.mean(v[method] for v in common):.5f}'])
    p.append(table(['指标','预测器','共同 cells','MAE ↓'],allrows))
    p.append('raw-linear 每项指标各有 3 个 cell 没有有效预测；本表统一使用 297 个共同有效 cells，不能把它和上方 300-cell MAE 直接相减。\n')
    p += ['\n## 3. 额外九模型：校准预算消融','',
          '同一目标 cell 内比较；Clean baseline 始终需要。一个 repair pilot 档位 = 四个 repair 各跑该档 × 三个 seeds，不是一个训练 job。复用已有校准数据，属于预先冻结的离线消融。\n']
    rules=('transfer_no_repair_pilot','calibrate_q025_only','calibrate_q10_only','calibrate_q025_q10');brows=[]
    for metric in ('mean_per_item_nll','bits_per_byte'):
        for rule in rules:
            z=[float(r['relative_absolute_error'])*100 for r in budgets if r['model'] not in ('Qwen/Qwen3-8B','meta-llama/Llama-3.1-8B-Instruct') and r['metric']==metric and r['rule']==rule and r['relative_absolute_error']]
            brows.append([metric,rule,len(z),f'{st.mean(z):.3f}%'])
    p.append(table(['指标','校准方案','cells','相对绝对误差均值 ↓'],brows))
    p += ['\n### 校准成本（每个新模型）','',table(['校准方案','Clean baseline 训练','repair pilot 训练','本次新增训练'],[
        ['无需 repair pilot','3 seeds','0','0（离线复用）'],
        ['仅低剂量档','3 seeds','4 repair × 3 seeds = 12','0（离线复用）'],
        ['仅较高剂量档','3 seeds','4 repair × 3 seeds = 12','0（离线复用）'],
        ['两档一起','3 seeds','4 repair × 2 档 × 3 seeds = 24','0（离线复用）']]),
        '不含用于迁移拟合的 anchor 成本，也不含目标剂量的检验训练；一次训练身份不代表跨模型 GPU 小时相等。\n']
    p += ['\n方法追溯：nearest_q10 指直接沿用最近校准剂量观测；固定饱和曲线用同一套已冻结校准数据，τ=0.10。具体剂量与预测器标识保留在原始数据中。q 是监督 token 中 repair 的比例，不是数据量 N。']
    p += ['\n[逐 cell 的四预测器绝对误差 CSV](data/forecast_errors_by_cell.csv)：保留全部 600 个指标 cell，空白代表该预测器无有效预测。']
    files['PREDICTION.md']='\n'.join(p)+'\n'

    h=['# 数据量扩展：两个 anchor，三个规模','',
       '仅报告 seeds 72/73 的复验。Selected 与 matched-uniform **配方完全相同，复用同一 checkpoint**，因此不是独立的两组训练证据。N 增长时更新数也增长，不是固定计算预算实验。\n',
       'Qwen3-8B：Clean/FMT/ANS/EVD/PARA = 0/25/25/25/25%；Llama-3.1-8B：20/20/20/20/20%。\n']
    paired=[r for r in history['paired'] if 'MODEL_SELECTED' in r['recipe_role'] and int(r['seed']) in (72,73)]
    assert len(paired)==12
    macro=[];detail=[]
    for m in ('Qwen/Qwen3-8B','meta-llama/Llama-3.1-8B-Instruct'):
        for n in (1920,3968,9984):
            rs=[r for r in paired if r['model']==m and int(r['N'])==n];assert len(rs)==2
            c=[st.mean(float(r[g+'_reference_NLL']) for g in GROUPS) for r in rs];o=[st.mean(float(r[g+'_NLL']) for g in GROUPS) for r in rs]
            macro.append([short(m),n,fmt(c),fmt(o),fmt(o),'0（同一 checkpoint）'])
            for g in GROUPS:
                detail.append([short(m),n,g,fmt([float(r[g+'_reference_NLL']) for r in rs]),fmt([float(r[g+'_NLL']) for r in rs]),fmt([float(r[g+'_relative_NLL_pct']) for r in rs],places=2)])
    h += ['## 五组 NLL 等权均值 ↓（描述性）','',table(['模型','N','Clean-SFT','Uniform','Ours','Ours−Uniform'],macro),
          '\n## 每个评测组','',table(['模型','N','组','Clean-SFT NLL','Ours = Uniform NLL','配对相对 Clean 变化 (%)'],detail),
          '\nQwen 均值先降后升，Llama 均值上升；当前支持跨规模保持相对 Clean-SFT 的平均损失优势，不支持已经验证单调或幂律 data scaling。']
    files['DATA_SCALING.md']='\n'.join(h)+'\n'

    rows=[]
    for stage,domain,records in [('volume3968','Math',volume),('fixed18','Math',fixed),('fixed18','Knowledge',knowledge)]:
        for r in records:
            for g in GEN[domain]:
                for metric in ('gen','nll','bpb'):
                    value=get(r,metric,g,domain)
                    rows.append(dict(stage=stage,domain=domain,model=r['model'],arm=r['arm'],seed=r['seed'],group=g,metric=metric,value=value,checkpoint_sha256=r['checkpoint_sha256'],source=r['source']))
    files['data/measurements_long.csv']=csv_text(rows)
    forecast_cells=[]
    for key,values in sorted(indexed.items()):
        forecast_cells.append(dict(zip(('model','repair_axis','endpoint','metric','target_repair_fraction'),key),**{method+'_absolute_error':values.get(method,'') for method in methods}))
    files['data/forecast_errors_by_cell.csv']=csv_text(forecast_cells)
    completed=[x for x in clean_v if x[1]>0];tied=[x for x in clean_v if x[1]==0]
    _,nll_v=comparison(volume,models,'macro_nll');nll_wins=[x for x in nll_v if x[1]<0]
    domain_summary={}
    for domain,records in [('Math',fixed),('Knowledge',knowledge)]:
        z={}
        for metric in ('gen','macro_nll'):
            _,v=comparison(records,MODELS,metric,domain=domain)
            z[metric]={'models':len(v),'wins':sum(d>0 if metric=='gen' else d<0 for _,d in v)}
        domain_summary[domain]=z
    summary=dict(volume_complete=len(volume),volume_expected=81,fixed_records=len(fixed),knowledge_bound_records=len(knowledge),clean_bmk_full_pair_models=len(clean_v),clean_bmk_wins=len(completed),clean_bmk_ties=len(tied),clean_bmk_winner_models=[m for m,v in completed],nll_full_pair_models=len(nll_v),nll_wins=len(nll_wins),fixed_domain_summary=domain_summary,export_errors=data['errors'])
    tradeoffs=[]
    for model in models:
        row=[short(model)];count=0
        for metric in ('gen','strict_clean','macro_nll','false_abstain'):
            delta,count=paired_delta(volume,model,metric)
            row.append(sign(delta*(1 if metric=='macro_nll' else 100),4 if metric=='macro_nll' else 2) if delta is not None else '—')
        tradeoffs.append(row+[count])
    files['data/report_checks.json']=json.dumps(summary,ensure_ascii=False,indent=2)+'\n'
    files['README.md']='\n'.join(['# 实验对照总览 · 2026-09-19','',f'远端快照：`{data["snapshot_utc"]}`。只读汇总，不改变任何训练、冻结预测或旧结果。','',
        '> **BMK 答题更准**与**数学响应模型预测 loss 更准**是两个问题；本报告分开作答，不用 NLL 代替准确率。','',
        '## 先看结论','',
        f'- **N=3,968 的 Clean BMK（宽松提取）**：三 seed 完整配对的 {len(clean_v)} 个模型中，Ours 均值高于 uniform 的有 **{len(completed)} 个**，持平 {len(tied)} 个；胜出模型：'+(', '.join(short(m) for m,v in completed) or '无')+'。只描述均值，不声称显著。',
        f'- **同批五组平均 NLL**：{len(nll_v)} 个完整配对模型中 {len(nll_wins)} 个更低。准确率胜负不能替代 loss 胜负。',
        f'- **原 18-step 条件**：有匹配 uniform 的 9 个模型，Math 宏 NLL {domain_summary["Math"]["macro_nll"]["wins"]}/9 更低，Knowledge 宏 NLL {domain_summary["Knowledge"]["macro_nll"]["wins"]}/9 更低；仍不是所有模型、所有域胜出。',
        '- **冻结剂量预测**：饱和模型总体 MAE 没有优于最近校准点基线；逐模型和共同 cell 比较见预测页。',
        '- **数据量扩展**：两 anchor 的 selected 与 uniform 相同；可以比较 Clean-SFT，不能据此证明选择算法胜过 uniform。','',
        '## 一眼看 trade-off：N=3,968，Ours − Uniform','',
        'BMK 两列越正越好；NLL 与误拒答两列越负越好。准确率差的单位为百分点（pp）；部分 seed 单独标识，不纳入上方完整配对胜出计数。列出全部九模型，不筛选方向。\n',
        table(['模型','宽松 Clean BMK ↑ (pp)','明确格式 Clean BMK ↑ (pp)','五组 NLL ↓','误拒答率 ↓ (pp)','配对 seeds / 3'],tradeoffs),
        '绝对分数与 seed 波动请看第一张详细表。不要把宽松提取的提升当成格式遵循也提升，也不要把低 reference NLL 当成实际答题更准。\n',
        '## 表格导航','',table(['页面','你能看到什么'],[
            ['[固定 N=3,968：BMK / NLL / BPB](VOLUME_3968.md)','九模型三方表、严格/宽松准确率、五组分项、配方、失败项'],
            ['[固定原训练条件：全部 checkpoints / 两域](FIXED_CHECKPOINTS.md)','11 模型、Math 与 grounded Knowledge、精确 checkpoint 绑定、不可比和缺失项'],
            ['[Loss 预测误差](PREDICTION.md)','逐模型饱和/最近点基线、线性基线共同交集、校准成本消融'],
            ['[Data-size scaling](DATA_SCALING.md)','两个模型、三个 N、三方绝对 NLL 与逐项相对增益'],
            ['[指标定义与审计](AUDIT.md)','训练条件变化、原目标、样本数、来源与可复现性'],
            ['[逐 seed 长表 CSV](data/measurements_long.csv)','完整数值及 checkpoint/source 绑定，便于自己做 trade-off']]),
        '## 当前执行状态','',f'- N=3,968：{len(volume)}/81 完整结果，{len(data["failures"])} 个缺完整结果；旧 GPU 队列已收尾。',
        '- 本次运行的是 CPU 结果导出、条件复核和报告生成；**本次没有启动新 GPU 训练**。新配方仍需版本化冻结、资源与预算检查。',
        '- 保守旧账本尚余 55 个未分配训练身份；没有把表格汇总计为新训练。',''])
    files['AUDIT.md']='\n'.join(['# 指标与来源审计','',intro,
        '## 三批实验不能混用','',table(['实验','选择方法 / 条件','能回答的问题'],[
            ['九模型 N=3968','旧分段响应 + balanced residual；配方选于18步，验证31步并扩展源池','旧配方迁移到新训练条件后的实测对照'],
            ['固定原条件','18步/2304draws；同一 Math-trained checkpoint 另读 Knowledge','原训练条件下的配方表现及跨域读数'],
            ['19日纯剂量预测','固定饱和曲线；冻结后测新剂量；不重选上述配方','未用于校准的剂量预测误差'],
            ['两个anchor数据量','有限候选响应模型选择；N1920/3968/9984，seed72/73','已选配方跨数据量验证；此处与uniform重合']]),
        '## BMK 定义','',
        '- Math Clean/EVD/PARA：acc_loose（宽松提取）；Clean 严格 acc_exact 另表。FMT：format.main（格式与内容共同正确）；ANS：insufficient_stop（应停止时停止），不能称通用答案准确率。',
        '- Knowledge Clean/EVD：acc_strict；FMT：format.main；ANS：insufficient_stop。Knowledge 无 PARA，不做填补。ANS 两项控制指标保留在快照，不混进四/五组均值。',
        '- Math 主生成面板分组 n：Clean 529、FMT 529、ANS 249、EVD 529、PARA 50；完整4700项面板还包含控制条件。逐 seed 原始 summary 的 n 随快照保留。',
        '- NLL：每项 canonical full-response token-average NLL 再按 item 平均，单位 nats/token；BPB：bits/UTF-8 response byte。不是通用语言建模 loss。',
        '- 宏 NLL 为组间等权描述性汇总；不是每项 pooled loss，也不是原选择器目标。',
        '- 所有数值均保留不利方向；不按结果删模型。完整三 seed 的 pair 才计入摘要胜出计数；不是成功概率或统计显著性。','',
        '## 来源与复现','',
        '- `data/readouts.json.gz.b64`：gzip + Base64 编码的 JSON 快照，只含汇总指标、配方、hash 和来源路径，不含训练文本、原始问题或权重。导出逐条核对 spec 与 checkpoint 绑定。',
        '- `data/historical.json`：历史数据量面板的 source-backed 配对数据。',
        '- `data/forecast_errors.csv.gz.b64` 与 `data/calibration_budget_errors.csv.gz.b64`：压缩保留已冻结预测与实测，不做重拟合。`make_report.read_input()` 自动解码；`data/measurements_long.csv` 可直接查看。',
        '- `export_readouts.py`：远端只读导出；`make_report.py`：离线构建，stdout 输出 path/content JSON。',
        '- 复算：在本目录执行 `python3 make_report.py`；输出是所有生成文档的内容映射。`python3 make_report.py --file VOLUME_3968.md` 输出单个文档。`python3 -m unittest -v test_report.py` 执行核验。',
        '- `.gz.b64` 为 UTF-8 文本的 gzip 压缩再 Base64，不是加密；压缩前后的 SHA-256 见 `data/input_manifest.json`。Knowledge 的生成与 NLL 均检查同一 checkpoint，生成面板的 scorer/fixture hash 也保留。','',
        '## 导出审计警告','',json.dumps(data['errors'],ensure_ascii=False,indent=2),''])
    for name in files:
        if name.endswith('.md') and name!='README.md':
            title,body=files[name].split('\n',1)
            files[name]=title+'\n\n[← 返回总览](README.md)\n'+body
    return {name:content.rstrip()+'\n' for name,content in files.items()}

if __name__=='__main__':
    files=build()
    if len(sys.argv)==2 and sys.argv[1]=='--list':print(json.dumps(list(files)))
    elif len(sys.argv)==3 and sys.argv[1]=='--file':print(files[sys.argv[2]],end='')
    else:print(json.dumps(files,ensure_ascii=False))
