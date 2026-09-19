"""Read-only HDFS export. Emits aggregate measurements, never prompt text or weights."""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/hdfs/xwqu')
GROUPS = ('Clean', 'FMT', 'ANS', 'EVD', 'PARA')
ARMS = ('clean', 'uniform_matched_q', 'selected_recipe')

def load(path):
    return json.loads(Path(path).read_text())

def compact(bundle, path, spec=None):
    if bundle.get('status') != 'complete' or bundle.get('final_accessed') is not False:
        raise ValueError('not a completed DEV bundle: ' + str(path))
    if spec and bundle['spec_sha256'] != hashlib.sha256(Path(spec).read_bytes()).hexdigest():
        raise ValueError('spec mismatch: ' + str(path))
    gen = bundle['generation']
    if gen.get('status') != 'complete' or gen.get('checkpoint_tree_sha256') != bundle['checkpoint_tree_sha256']:
        raise ValueError('generation/checkpoint binding mismatch: ' + str(path))
    if gen.get('scientific_result') is not True or gen.get('qualification_only') is not False:
        raise ValueError('not a scientific readout: ' + str(path))
    return dict(model=bundle['model_id'], seed=bundle['seed'], job_id=bundle['job_id'],
                step=bundle['step'], N=bundle.get('N_actual_distinct_rows',2304),
                setting=bundle['setting'], shares=bundle['requested_token_shares'],
                checkpoint_sha256=bundle['checkpoint_tree_sha256'],
                spec_sha256=bundle['spec_sha256'], source=str(path),
                source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                nll={g: {k:v for k,v in z.items() if k in ('mean_per_item_nll','bits_per_byte','items')} for g,z in bundle['benchmark_nll_groups'].items()},
                generation=gen['summary'], main_metrics=gen.get('main_metrics',{}),
                generation_fixture=gen.get('input_sha256'), scorer=gen.get('scorer_sha256'))

def collect():
    out=dict(snapshot_utc=datetime.now(timezone.utc).isoformat(), volume=[], fixed=[], failures=[], selections=[], knowledge={}, errors=[])
    volume=ROOT/'five_component_multimodel_volume_validation_v1'
    for batch in ('batch01_qwen06_mistral_n3968','batch02_qwen17_qwen4_qwen35_2_n3968','batch03_qwen35_9_qwen25_olmo_gemma_n3968'):
        root=volume/batch
        for task in load(root/'snapshot/TASK_QUEUE.json'):
            spec=load(task['spec']); p=root/'results'/task['node']/task['job_id']/'CALIBRATION_RESULT_BUNDLE.json'
            if not p.exists():
                out['failures'].append(dict(model=spec['canonical_model_id'], arm=spec['repair'], seed=spec['seed'],job_id=task['job_id'],stage='volume',reason='NO_COMPLETE_BUNDLE'))
                continue
            record=compact(load(p),p,task['spec']);record['arm']=spec['repair'];out['volume'].append(record)
    for root in (ROOT/'five_component_calibrated_repair_dev_v1',ROOT/'five_component_multimodel_multidomain_dev_v2/math'):
        originals={}
        for task in load(root/'snapshot/TASK_QUEUE.json'):
            spec=load(task['spec']); originals[spec['canonical_model_id'],spec['repair'],spec['seed']]=(task,spec)
            if spec['repair'] not in ('clean','balanced-q025x4'):continue
            p=root/'results'/task['node']/task['job_id']/'CALIBRATION_RESULT_BUNDLE.json'
            record=compact(load(p),p,task['spec']);record['arm']='clean' if spec['repair']=='clean' else 'uniform_reference';out['fixed'].append(record)
        for event in sorted((root/'selection_events').glob('*.json')):
            e=load(event)
            if not e.get('selected_recipe'):continue
            selected=load(e['selected_recipe']);model=selected['model_id'];shares=selected['selected']['requested_shares'];total=sum(shares.values())
            out['selections'].append(dict(model=model,recipe={'Clean':1-total,**shares},source=e['selected_recipe'],event_time=e.get('timestamp'),
                predicted_repair_relative=selected['selected']['four_task_mean_normalized_delta'],predicted_clean_relative=selected['selected']['mean_normalized_delta']['Clean']))
            confirmation=root/'confirmation_snapshots'/model.replace('/','--')/'TASK_QUEUE.json'
            if confirmation.exists():
                tasks=load(confirmation)
                for task in tasks:
                    p=root/'results'/task['node']/task['job_id']/'CALIBRATION_RESULT_BUNDLE.json'
                    if not p.exists():
                        out['errors'].append(dict(source=str(p),error='missing fixed-condition confirmation'));continue
                    record=compact(load(p),p,task['spec']);record['arm']='selected_recipe';out['fixed'].append(record)
            else:
                for seed in (71,72,73):
                    matches=[(t,s) for (m,setting,z),(t,s) in originals.items() if m==model and z==seed and all(abs(s['requested_supervised_token_shares'].get(k,0)-v)<1e-9 for k,v in shares.items())]
                    if len(matches)!=1:
                        out['errors'].append(dict(model=model,seed=seed,error='ambiguous/missing exact calibration reuse'));continue
                    task,spec=matches[0];p=root/'results'/task['node']/task['job_id']/'CALIBRATION_RESULT_BUNDLE.json'
                    record=compact(load(p),p,task['spec']);record['arm']='selected_recipe';record['exact_calibration_reuse']=True;out['fixed'].append(record)
    wanted={r['checkpoint_sha256'] for r in out['fixed']+out['volume']}
    kr=ROOT/'knowledge_checkpoint_readout_v1'
    for p in sorted((kr/'results').glob('*/KNOWLEDGE_ABSOLUTE_RESULT.json')):
        if p.parent.name.startswith('.'):continue
        terminal=kr/'terminals'/(p.parent.name+'.json')
        if not terminal.exists():continue
        try:
            if load(terminal).get('status')!='complete':continue
            b=load(p);key=b['checkpoint_tree_sha256']
            if key not in wanted:continue
            gen=b['generation']
            if (b.get('status')!='complete' or b.get('final_accessed') is not False or
                gen.get('status')!='complete' or gen.get('checkpoint_tree_sha256')!=key or
                gen.get('scientific_result') is not True or gen.get('qualification_only') is not False):
                raise ValueError('knowledge DEV/generation/checkpoint binding mismatch')
            rec=dict(model=b['model_id'],seed=b['seed'],checkpoint_sha256=key,source=str(p),
                source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),nll={g:{k:v for k,v in z.items() if k in ('mean_per_item_nll','bits_per_byte','items')} for g,z in b['nll_groups'].items()},generation=gen['summary'],generation_fixture=gen.get('input_sha256'),scorer=gen.get('scorer_sha256'))
            if key in out['knowledge'] and out['knowledge'][key]['nll']!=rec['nll']:
                raise ValueError('conflicting knowledge evaluations for checkpoint '+key)
            out['knowledge'][key]=rec
        except (ValueError,KeyError,OSError) as err:out['errors'].append(dict(source=str(p),error=str(err)))
    json.dump(out,sys.stdout,ensure_ascii=False,allow_nan=False,separators=(',',':'))

if __name__=='__main__':collect()
