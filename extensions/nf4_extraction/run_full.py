"""Authorized full SRO extraction: NF4 extension, immutable inputs, session checkpoints."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
sys.dont_write_bytecode = True
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from choice_extractor_snapshot import FrozenChoiceExtractor, ChoiceExtraction

ROOT = Path(__file__).resolve().parent
PROJECT = Path(os.environ.get('CM_NF4_PROJECT_ROOT', ROOT.parents[1])).resolve()
ORIGINAL = Path(os.environ.get('CM_NF4_ORIGINAL_ROOT', PROJECT / 'RESULTS_AND_REPRODUCIBILITY')).resolve()
INPUTS = Path(os.environ.get('CM_NF4_INPUTS_ROOT', PROJECT / 'DATA_INPUTS/data/derived/original/sro_sessions')).resolve()
PILOT = Path(os.environ.get('CM_NF4_PILOT_ROOT', PROJECT / 'EXTENSIONS/pilot_20260914')).resolve()
BASE = Path(os.environ.get('CM_NF4_BASE_MODEL', PROJECT / 'models/Llama-3.1-70B')).resolve()
ADAPTER = Path(os.environ.get('CM_NF4_ADAPTER_MODEL', PILOT / 'models/centaur_adapter')).resolve()
LAYERS = [20, 40, 60, 80]
MODELS = ['centaur_70b_nf4', 'llama_base_70b_nf4']
SPLITS = ['development', 'confirmation_t1', 'confirmation_t2']
STATE = {'profile': 'extension_full_nf4', 'pid': os.getpid()}


class ChunkedChoiceExtractor(FrozenChoiceExtractor):
    """Bound transient activations without truncating the full-history KV cache."""
    @torch.inference_mode()
    def forward_chunks(self, ids, cache):
        for start in range(0, ids.shape[1], 2048):
            end = min(start + 2048, ids.shape[1])
            capture = self._capture() if end == ids.shape[1] else nullcontext({})
            with capture as states:
                output = self.model.model(input_ids=ids[:, start:end], past_key_values=cache,
                                          use_cache=True, return_dict=True)
            cache = output.past_key_values
            if end == ids.shape[1]:
                logits = self.model.lm_head(output.last_hidden_state[:, -1, :])
                captured = {l:t.squeeze(0) for l,t in states.items()}
            del output
        return logits, captured, cache

    @torch.inference_mode()
    def extract_session(self, prefixes, candidates, responses):
        results = []
        previous = cache = None
        evidence = dict(cache_reused=0, cache_crop_events=0, cache_prefix_tokens_reused=0, full_recomputations=0)
        for prefix, values, response in zip(prefixes, candidates, responses, strict=True):
            labels = list(values)
            assert response in labels
            current = self._tokenize_prefix(prefix)
            common = 0
            if cache is not None and previous is not None:
                length = min(current.shape[1], previous.shape[1])
                differences = torch.nonzero(current[:, :length] != previous[:, :length], as_tuple=False)
                common = length if not len(differences) else int(differences[0, 1])
            if cache is not None and 0 < common < current.shape[1]:
                if common < previous.shape[1]:
                    cache.crop(common)
                    evidence['cache_crop_events'] += 1
                ids = current[:, common:]
                evidence['cache_reused'] += 1
                evidence['cache_prefix_tokens_reused'] += common
            else:
                cache = None
                ids = current
                evidence['full_recomputations'] += 1
            logits, states, cache = self.forward_chunks(ids, cache)
            scores, logprob = self._single_token_choice(logits, labels, response)
            results.append(ChoiceExtraction(hidden_states=states, candidate_logprobs=scores,
                nll=-logprob, response_logprob=logprob, token_count=int(current.shape[1])))
            previous = current
            del logits
        return results, evidence

    @torch.inference_mode()
    def extract(self, prefix, candidates, response):
        results, _ = self.extract_session([prefix], [candidates], [response])
        return results[0]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 * 1024**2):
            h.update(b)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def status(state, **kwargs):
    STATE.update(state=state, updated_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    atomic(ROOT / 'status.json', STATE)
    print(json.dumps(STATE), flush=True)


def inventory():
    split_path = ORIGINAL / 'configs/original/splits.json'
    splits = json.loads(split_path.read_text())
    dev = {r['participant']: int(r['fold']) for r in splits['development']}
    confirmation = set(splits['confirmation'])
    assert not set(dev) & confirmation
    items = []
    for split in SPLITS:
        paths = sorted((INPUTS / split).glob('*.parquet'))
        assert len(paths) == 16, (split, len(paths))
        for p in paths:
            f = pd.read_parquet(p, columns=['validity', 'participant', 'trial'])
            f = f.loc[f.validity == 'primary']
            allowed = set(dev) if split == 'development' else confirmation
            assert set(f.participant) <= allowed
            assert not f.duplicated(['participant', 'trial']).any()
            items.append(dict(split=split, task=p.stem, path=str(p), sha256=sha(p), rows=len(f), sessions=f.participant.nunique()))
    totals = {s: sum(i['rows'] for i in items if i['split'] == s) for s in SPLITS}
    assert totals['development'] == 1732227, totals
    assert totals['confirmation_t1'] + totals['confirmation_t2'] == 1402036, totals
    hashes = {n: sha(ROOT / n) for n in ['run_full.py', 'choice_extractor_snapshot.py']}
    source_configs = {n: sha(ORIGINAL / n) for n in ['configs/original/splits.json', 'configs/original/main.yaml', 'configs/original/centaur70b_frozen.yaml']}
    plan = dict(schema_version=1, profile='extension_full_nf4', models=MODELS, splits=SPLITS, layers=LAYERS,
                layer_semantics='one-based post-block residual, including layer 80 BEFORE final RMSNorm',
                primary_layer=60, compute_dtype='bfloat16', storage_dtype='float16',
                quantization='NF4 double quantization', prefix_policy='full frozen prefix, independently tokenized, no truncation',
                maximum_prefill_chunk_tokens=2048, kv_cache_policy='full history retained on GPU, no sliding window or quantized KV',
                checkpoint_unit='model/split/task/participant', input_files=items,
                source_config_sha256=source_configs, source_code_sha256=hashes, trials_per_model=sum(totals.values()),
                trials_both_models=2*sum(totals.values()), trials_by_split=totals,
                archived_resource_extrapolation=dict(development=1732627, confirmation_T1_and_T2=1397280),
                count_discrepancy_policy='Use all primary rows in the immutable prepared tables; archived resource estimates differ',
                analysis_policy='extraction only; no fitting or selection on confirmation data',
                authorization='User explicitly requested full-scale 70B 4-bit on 2026-09-14',
                checkpoint_and_consolidated_state_bytes=4*sum(totals.values())*4*8192*2)
    p = ROOT / 'frozen_plan.json'
    if p.exists():
        assert json.loads(p.read_text()) == plan, 'Frozen run inputs/config changed; use a new run directory'
    else:
        atomic(p, plan)
    return plan, dev


def verify_assets():
    source = json.loads((ORIGINAL / 'manifests/private/original/llama70b_download.json').read_text())
    assert str(BASE) == source['source']
    for item in source['files']:
        p = BASE / item['path']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
    adapter = json.loads((PILOT / 'adapter_manifest.json').read_text())
    for item in adapter['files']:
        p = ADAPTER / item['path']
        assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256'], str(p)
    atomic(ROOT / 'asset_verification.json', dict(base_manifest_sha256=sha(ORIGINAL / 'manifests/private/original/llama70b_download.json'),
        adapter_manifest_sha256=sha(PILOT / 'adapter_manifest.json'), verified_utc=datetime.now(timezone.utc).isoformat()))


def construct_extractor(model):
    ext = ChunkedChoiceExtractor.__new__(ChunkedChoiceExtractor)
    ext.model_path = BASE
    ext.layers = tuple(LAYERS)
    ext.device = torch.device('cuda:0')
    ext.tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    ext.model = model
    ext.sdpa_backend = {'cudnn_enabled': False, 'implementation': 'sdpa'}
    return ext


def validate(ext, model_name):
    """Checks that can detect cache/state mistakes and longest-prefix OOM before production."""
    status('validating_extractor', model=model_name)
    frame = pd.read_parquet(INPUTS / 'development/simon.parquet')
    primary = frame.loc[frame.validity == 'primary'].sort_values(['participant', 'trial'], kind='stable')
    session = next(iter(primary.groupby('participant', sort=True)))[1]
    # Early and final adjacent trials exercise both cache reuse and a long common prefix.
    sample = pd.concat([session.head(3), session.tail(2)]).drop_duplicates(['participant', 'trial'])
    cached, evidence = ext.extract_session(sample.prompt_prefix, sample.candidate_responses, sample.response)
    checks = []
    for row, got in zip(sample.itertuples(index=False), cached, strict=True):
        direct = FrozenChoiceExtractor.extract(ext, row.prompt_prefix, row.candidate_responses, row.response)
        cosines = {str(l): float(torch.nn.functional.cosine_similarity(got.hidden_states[l], direct.hidden_states[l], dim=0)) for l in LAYERS}
        delta = abs(got.nll - direct.nll)
        assert got.token_count == direct.token_count
        assert delta < 0.05 and min(cosines.values()) > 0.999, (delta, cosines)
        checks.append(dict(trial=int(row.trial), absolute_nll_delta=delta, state_cosines=cosines))
    assert evidence['cache_reused'] > 0
    # Force multi-chunk prefill independently, and compare with the original full forward.
    tail = session.iloc[-1]
    chunked = ext.extract(tail.prompt_prefix, tail.candidate_responses, tail.response)
    direct = FrozenChoiceExtractor.extract(ext, tail.prompt_prefix, tail.candidate_responses, tail.response)
    assert chunked.token_count > 2048
    chunk_cosines = {str(l):float(torch.nn.functional.cosine_similarity(chunked.hidden_states[l],direct.hidden_states[l],dim=0)) for l in LAYERS}
    assert abs(chunked.nll-direct.nll) < 0.05 and min(chunk_cosines.values()) > 0.999
    del frame, primary, session, sample, cached
    gc.collect()
    # This decision is based only on development prompt lengths, never response performance.
    frame = pd.read_parquet(INPUTS / 'development/stop_signal.parquet')
    last = frame.loc[frame.validity == 'primary'].sort_values(['participant', 'trial'], kind='stable').groupby('participant', sort=True).tail(1)
    counts = [len(ext.tokenizer(p, add_special_tokens=True).input_ids) for p in last.prompt_prefix]
    row = last.iloc[int(np.argmax(counts))]
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    result = ext.extract(row.prompt_prefix, row.candidate_responses, row.response)
    assert result.token_count == max(counts) and result.token_count >= 55878
    assert np.isfinite(result.nll) and all(torch.isfinite(v).all().item() for v in result.hidden_states.values())
    report = dict(cache_comparisons=checks, cache_evidence=evidence, chunked_prefill_cosines=chunk_cosines,
        chunked_prefill_absolute_nll_delta=abs(chunked.nll-direct.nll), longest_development_tokens=result.token_count,
        longest_prefix_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        layer_80_semantics='post-block residual before final RMSNorm; unlike pilot hidden_states[80]')
    atomic(ROOT / ('validation_' + model_name + '.json'), report)
    del frame, last, result
    gc.collect()
    torch.cuda.empty_cache()


def verify_checkpoint(dest, item, participant, rows):
    marker = dest / 'complete.json'
    if not marker.exists():
        return None
    d = json.loads(marker.read_text())
    assert d['source_sha256'] == item['sha256'] and d['participant'] == participant and d['rows'] == rows
    for name, digest in d['files'].items():
        assert sha(dest / name) == digest, 'Checkpoint integrity mismatch: ' + str(dest / name)
    return d


def write_checkpoint(dest, item, participant, session, results, model_name, dev, evidence, elapsed):
    dest.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for layer in LAYERS:
        a = np.stack([r.hidden_states[layer].numpy() for r in results]).astype(np.float16)
        assert a.shape == (len(session), 8192) and np.isfinite(a).all()
        arrays[f'layer_{layer:02d}'] = a
    tmp = dest / 'states.tmp.npz'
    np.savez(tmp, **arrays)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    tmp.replace(dest / 'states.npz')
    records = []
    for i, (row, result) in enumerate(zip(session.itertuples(index=False), results, strict=True)):
        assert np.isfinite(result.nll) and abs(sum(np.exp(list(result.candidate_logprobs.values()))) - 1) < 1e-6
        d = {k: getattr(row, k) for k in ['dataset','participant','timepoint','task','task_family','trial','block','condition','response','config_hash','dataset_revision']}
        d.update(model=model_name, model_revision='extension_nf4_adapter_159600db8be99dc183c289923148dfd96cbd8e07' if model_name.startswith('centaur') else 'verified_original_base_nf4',
            split=item['split'], fold=dev.get(participant), candidate_responses=list(row.candidate_responses),
            candidate_logprobs=ext_encode(result.candidate_logprobs), response_logprob=result.response_logprob,
            nll=result.nll, token_count=result.token_count, state_row=i)
        records.append(d)
    pd.DataFrame(records).to_parquet(dest / 'metadata.tmp.parquet', index=False, compression='zstd')
    (dest / 'metadata.tmp.parquet').replace(dest / 'metadata.parquet')
    data = dict(source_sha256=item['sha256'], participant=participant, rows=len(session),
                files={n: sha(dest/n) for n in ['states.npz','metadata.parquet']}, cache_evidence=evidence,
                seconds=elapsed, max_tokens=max(r.token_count for r in results), completed_utc=datetime.now(timezone.utc).isoformat())
    atomic(dest / 'complete.json', data)
    return data


def ext_encode(scores):
    return json.dumps(scores, sort_keys=True, separators=(',', ':'))


def consolidate(destination, shard_dirs, item):
    marker = destination / 'complete.json'
    if marker.exists():
        d = json.loads(marker.read_text())
        assert d['rows'] == item['rows'] and d['source_sha256'] == item['sha256']
        for n,h in d['files'].items():
            assert sha(destination/n) == h
        return
    destination.mkdir(parents=True, exist_ok=True)
    arrays = {l: np.lib.format.open_memmap(destination / f'layer_{l:02d}.partial.npy', mode='w+', dtype=np.float16, shape=(item['rows'],8192)) for l in LAYERS}
    offset = 0
    frames = []
    for d in shard_dirs:
        f = pd.read_parquet(d / 'metadata.parquet')
        with np.load(d / 'states.npz', allow_pickle=False) as z:
            for l in LAYERS:
                arrays[l][offset:offset+len(f)] = z[f'layer_{l:02d}']
        f['state_row'] = np.arange(offset,offset+len(f))
        frames.append(f)
        offset += len(f)
    assert offset == item['rows']
    for l,a in arrays.items():
        a.flush()
    arrays.clear()
    for l in LAYERS:
        (destination/f'layer_{l:02d}.partial.npy').replace(destination/f'layer_{l:02d}.npy')
    pd.concat(frames, ignore_index=True).to_parquet(destination/'metadata.tmp.parquet', index=False, compression='zstd')
    (destination/'metadata.tmp.parquet').replace(destination/'metadata.parquet')
    names = ['metadata.parquet'] + [f'layer_{l:02d}.npy' for l in LAYERS]
    atomic(marker, dict(rows=offset, source_sha256=item['sha256'], files={n:sha(destination/n) for n in names}))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    lock = (ROOT/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status('inventory')
    plan, dev = inventory()
    free = shutil.disk_usage(ROOT).free
    assert free > plan['checkpoint_and_consolidated_state_bytes'] + 100*1024**3, 'Insufficient output storage'
    if args.check_only:
        status('preflight_complete', trials_total=plan['trials_both_models'], output_state_bytes=plan['checkpoint_and_consolidated_state_bytes'])
        return
    status('verifying_assets')
    verify_assets()
    torch.manual_seed(20260914)
    torch.set_num_threads(4)
    torch.backends.cuda.enable_cudnn_sdp(False)
    gpu_jobs = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
    assert not [p for p in gpu_jobs.split() if p != str(os.getpid())], 'GPU occupied by another job'
    assert torch.cuda.mem_get_info()[0] > 70*1024**3
    runtime = {n: importlib.metadata.version(n) for n in ['torch','transformers','peft','bitsandbytes','numpy','pandas','pyarrow','accelerate','safetensors']}
    atomic(ROOT/'runtime.json', dict(packages=runtime, python=sys.executable, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)))
    (ROOT/'environment.freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    quant = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    status('loading_model')
    base = AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,quantization_config=quant,dtype=torch.bfloat16,
        device_map={'':0},attn_implementation='sdpa',low_cpu_mem_usage=True)
    assert base.is_loaded_in_4bit and all(p.device.type=='cuda' for p in base.parameters())
    peft = PeftModel.from_pretrained(base,ADAPTER,local_files_only=True,is_trainable=False)
    assert any('lora_' in n for n,_ in peft.named_parameters())
    peft.eval()
    peft.requires_grad_(False)
    ext = construct_extractor(peft.get_base_model())
    completed_trials = completed_sessions = 0
    start = time.monotonic()
    for model_name in MODELS:
        context = nullcontext() if model_name.startswith('centaur') else peft.disable_adapter()
        with context:
            validate(ext, model_name)
            for split in SPLITS:
                tasks = sorted([i for i in plan['input_files'] if i['split']==split], key=lambda i:(i['rows'],i['task']))
                for item in tasks:
                    status('reading_task', model=model_name, split=split, task=item['task'], completed_trials=completed_trials)
                    f = pd.read_parquet(item['path'])
                    f = f.loc[f.validity=='primary'].sort_values(['participant','trial'],kind='stable').reset_index(drop=True)
                    assert len(f)==item['rows']
                    shards=[]
                    for participant, session in f.groupby('participant',sort=False):
                        name=hashlib.sha256(participant.encode()).hexdigest()[:24]
                        dest=ROOT/'checkpoints'/model_name/split/item['task']/name
                        shards.append(dest)
                        found=verify_checkpoint(dest,item,participant,len(session))
                        if found is None:
                            assert shutil.disk_usage(ROOT).free > 100*1024**3
                            status('extracting', model=model_name, split=split, task=item['task'], session_key=name,
                                   completed_trials=completed_trials, completed_sessions=completed_sessions,
                                   trials_total=plan['trials_both_models'])
                            t=time.monotonic()
                            results,evidence=ext.extract_session(session.prompt_prefix,session.candidate_responses,session.response)
                            write_checkpoint(dest,item,participant,session,results,model_name,dev,evidence,time.monotonic()-t)
                            del results
                        completed_trials+=len(session)
                        completed_sessions+=1
                        status('checkpoint_saved',completed_trials=completed_trials,completed_sessions=completed_sessions,
                               elapsed_seconds=time.monotonic()-start, peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    del f
                    gc.collect()
                    status('consolidating_task')
                    consolidate(ROOT/'states'/model_name/split/item['task'],shards,item)
                    status('task_complete')
    assert completed_trials==plan['trials_both_models']
    status('complete', completed_trials=completed_trials, completed_sessions=completed_sessions)


if __name__=='__main__':
    try:
        run()
    except BaseException as exc:
        status('failed', error=repr(exc))
        raise
