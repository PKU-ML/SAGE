"""Train a variable-length Push-T local-subgoal action prior.

The model is conditioned on the local future latent z_{t+A} and the requested
chunk length A.  A is sampled on a 5-env-step grid, so A=10/15/... correspond
to 2/3/... action tokens under the standard PushT frameskip=5 setup.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sage.runtime.lewm import build_window_specs, denormalize_action_blocks, encode_lewm_context, image_batch_to_lewm, load_json, load_lewm, load_swm_dataset, lowdim_from_batch, normalize_action_blocks, normalize_lowdim, target_action_chunk
from sage.models.subgoal import load_subgoal_prior
from sage.sampling import sample_dense_unique_pairs
from sage.models.action_prior import (
    PushtVariablePooledGoalPrior,
    PushtVariableTransformerGoalPrior,
)
from sage.training import atomic_torch_save, compute_stats, finalize, move_stats, subset_specs, update

class VariableActionDataset(Dataset):
    def __init__(self, dataset, specs, rows, history_len:int, frameskip:int):
        self.dataset=dataset; self.specs=list(specs); self.rows=list(rows); self.history_len=int(history_len); self.frameskip=int(frameskip)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row=self.rows[int(index)]; spec=self.specs[int(row['spec_index'])]
        item=self.dataset[spec.dataset_index]
        current=int(spec.start)+(self.history_len-1)*self.frameskip
        a_env=int(row['action_offset']); goal=self.dataset._load_slice(spec.local_episode, current+a_env, current+a_env+1)
        far_env=int(row.get('goal_offset', a_env)); far_goal=self.dataset._load_slice(spec.local_episode, current+far_env, current+far_env+1)
        item['goal_pixels']=torch.as_tensor(goal['pixels'])
        item['far_goal_pixels']=torch.as_tensor(far_goal['pixels'])
        item['action_offset']=torch.tensor(a_env, dtype=torch.long)
        item['goal_offset']=torch.tensor(far_env, dtype=torch.long)
        item['action_tokens']=torch.tensor(int(row['action_tokens']), dtype=torch.long)
        item['episode_id']=torch.tensor(spec.episode_id, dtype=torch.long); item['start']=torch.tensor(spec.start, dtype=torch.long)
        return item

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', default='pusht_expert_train.lance'); p.add_argument('--cache-dir', default=None)
    p.add_argument('--split', required=True); p.add_argument('--policy', default='pusht/lewm'); p.add_argument('--out-dir', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--epochs', type=int, default=8); p.add_argument('--batch-size', type=int, default=128); p.add_argument('--num-workers', type=int, default=4); p.add_argument('--no-pin-memory', action='store_true'); p.add_argument('--no-persistent-workers', action='store_true'); p.add_argument('--prefetch-factor', type=int, default=1)
    p.add_argument('--lr', type=float, default=1e-4); p.add_argument('--weight-decay', type=float, default=1e-4); p.add_argument('--mode-l1-weight', type=float, default=0.05); p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--architecture', choices=['pooled','transformer'], default='pooled')
    p.add_argument('--hidden-dim', type=int, default=512); p.add_argument('--num-heads', type=int, default=8); p.add_argument('--depth', type=int, default=3); p.add_argument('--num-modes', type=int, default=8); p.add_argument('--pooling', choices=['attention','mean'], default='attention')
    p.add_argument('--history-len', type=int, default=3); p.add_argument('--frameskip', type=int, default=5); p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--action-offsets', nargs='+', type=int, default=[10,15,20,25,30,35])
    p.add_argument('--goal-offsets', nargs='+', type=int, default=None)
    p.add_argument('--subgoal-generator-checkpoint', default=None)
    p.add_argument(
        '--conditioning-goal-source',
        choices=['local', 'far'],
        default='local',
        help=(
            'Latent supplied as the action-prior target before optional '
            'generated-subgoal replacement. local uses z_{t+tau}; far uses '
            'the query goal z_{t+Delta}.'
        ),
    )
    p.add_argument('--generated-subgoal-ratio', type=float, default=0.0)
    p.add_argument('--eval-use-generated-subgoal', action='store_true')
    p.add_argument('--max-goal-offset', type=int, default=200)
    p.add_argument('--lowdim-keys', nargs='+', default=['state','proprio']); p.add_argument('--no-lowdim', action='store_true')
    p.add_argument('--max-train-windows', type=int, default=300000); p.add_argument('--max-val-windows', type=int, default=30000); p.add_argument('--max-train-examples', type=int, default=300000); p.add_argument('--max-val-examples', type=int, default=30000)
    p.add_argument('--balance-goal-offsets', action='store_true', help='When example limits are active, sample approximately equal counts for each far-goal offset.')
    p.add_argument('--dense-joint-sampling', action='store_true', help='Joint-balance (goal_offset, action_offset) cells with globally unique windows.')
    p.add_argument('--dense-balance-goals', action='store_true', help='With --dense-joint-sampling, allocate equal totals to each far-goal offset, then split uniformly over its valid action offsets.')
    p.add_argument('--dense-goal-quotas', nargs='*', default=None, help='Explicit dense far-goal quotas as H:COUNT entries, for example 25:60000 50:60000 150:50000.')
    p.add_argument('--dense-allow-repeats', action='store_true', help='After broad unique coverage, fill an oversized dense budget with balanced repeats.')
    p.add_argument('--max-train-batches', type=int, default=0); p.add_argument('--max-val-batches', type=int, default=0); p.add_argument('--eval-samples', type=int, default=64); p.add_argument('--coverage-threshold', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=42); p.add_argument('--bf16', action='store_true'); p.add_argument('--no-resume', action='store_true')
    p.add_argument(
        "--save-epochs",
        nargs="*",
        type=int,
        default=[],
        help="Also save the specified epoch_XXX.pt snapshots.",
    )
    args=p.parse_args();
    if args.no_lowdim: args.lowdim_keys=[]
    args.action_offsets=sorted(set(int(v) for v in args.action_offsets))
    if args.goal_offsets is not None:
        args.goal_offsets=sorted(set(int(v) for v in args.goal_offsets))
    if args.dense_goal_quotas is not None:
        quotas={}
        for token in args.dense_goal_quotas:
            try:
                goal,count=token.split(':',1)
                quotas[int(goal)]=int(count)
            except ValueError as exc:
                raise ValueError(f'Invalid --dense-goal-quotas entry {token!r}; expected H:COUNT') from exc
        args.dense_goal_quotas=quotas
        if not args.dense_joint_sampling:
            raise ValueError('--dense-goal-quotas requires --dense-joint-sampling')
    if any(v<=0 or v%int(args.frameskip)!=0 for v in args.action_offsets): raise ValueError('action offsets must be positive multiples of frameskip')
    if args.goal_offsets is not None and any(v<=0 for v in args.goal_offsets): raise ValueError('goal offsets must be positive')
    if not (0.0 <= float(args.generated_subgoal_ratio) <= 1.0): raise ValueError('--generated-subgoal-ratio must be in [0,1]')
    if args.generated_subgoal_ratio > 0 and not args.subgoal_generator_checkpoint:
        raise ValueError('--generated-subgoal-ratio requires --subgoal-generator-checkpoint')
    if args.subgoal_generator_checkpoint and not args.eval_use_generated_subgoal and args.generated_subgoal_ratio > 0:
        args.eval_use_generated_subgoal=True
    return args

def subset_row_indices(rows, limit:int, seed:int, balanced:bool):
    total=len(rows)
    if limit<=0 or limit>=total:
        return np.arange(total,dtype=np.int64)
    rng=np.random.default_rng(int(seed))
    if not balanced:
        return np.sort(rng.choice(total,size=int(limit),replace=False))
    labels=np.asarray([int(row['goal_offset']) for row in rows],dtype=np.int64)
    groups=[np.flatnonzero(labels==value) for value in sorted(np.unique(labels))]
    base,remainder=divmod(int(limit),len(groups)); selected=[]
    for group_index,group in enumerate(groups):
        requested=base+int(group_index<remainder); take=min(len(group),requested)
        if take:
            selected.append(rng.choice(group,size=take,replace=False))
    keep=np.concatenate(selected) if selected else np.empty(0,dtype=np.int64)
    if len(keep)<int(limit):
        available=np.setdiff1d(np.arange(total,dtype=np.int64),keep,assume_unique=False)
        keep=np.concatenate([keep,rng.choice(available,size=int(limit)-len(keep),replace=False)])
    return np.sort(keep.astype(np.int64,copy=False))

def build_rows(dataset, specs, args, split_name:str):
    if args.dense_joint_sampling:
        limit=int(args.max_train_examples if split_name=='train' else args.max_val_examples)
        goal_quotas=args.dense_goal_quotas
        if goal_quotas is not None and sum(goal_quotas.values()) != limit:
            total=float(sum(goal_quotas.values()))
            scaled={goal:int(np.floor(limit*count/total)) for goal,count in goal_quotas.items()}
            remainder=int(limit-sum(scaled.values()))
            order=sorted(goal_quotas, key=lambda goal: (-(limit*goal_quotas[goal]/total-scaled[goal]), goal))
            for goal in order[:remainder]:
                scaled[goal]+=1
            goal_quotas=scaled
        spec_indices, goal_offsets, action_offsets, diagnostics=sample_dense_unique_pairs(
            dataset,
            specs,
            history_len=args.history_len,
            frameskip=args.frameskip,
            goal_offsets=list(args.goal_offsets or args.action_offsets),
            action_offsets=list(args.action_offsets),
            limit=limit,
            seed=int(args.seed)+(3101 if split_name=='train' else 4101),
            allow_repeats=args.dense_allow_repeats,
            balance_by_goal=args.dense_balance_goals,
            goal_quotas=goal_quotas,
        )
        print(f'dense-{split_name} sampler {json.dumps(diagnostics, sort_keys=True)}', flush=True)
        rows=[
            {
                'spec_index':int(spec_index),
                'action_offset':int(action_offset),
                'goal_offset':int(goal_offset),
                'action_tokens':int(action_offset)//int(args.frameskip),
            }
            for spec_index,goal_offset,action_offset in zip(spec_indices,goal_offsets,action_offsets)
        ]
        return rows, len(rows)
    rows=[]; max_a=max(args.action_offsets); max_goal=max(args.goal_offsets or args.action_offsets)
    for spec_i,spec in enumerate(specs):
        current=int(spec.start)+(int(args.history_len)-1)*int(args.frameskip); final=int(dataset.lengths[spec.local_episode])-1
        if current+max_a>final: continue
        for a in args.action_offsets:
            if args.goal_offsets is None:
                if current+int(a)<=final:
                    rows.append({'spec_index':spec_i,'action_offset':int(a),'goal_offset':int(a),'action_tokens':int(a)//int(args.frameskip)})
                continue
            for goal_offset in args.goal_offsets:
                if int(goal_offset) < int(a):
                    continue
                if current+int(goal_offset)<=final:
                    rows.append({'spec_index':spec_i,'action_offset':int(a),'goal_offset':int(goal_offset),'action_tokens':int(a)//int(args.frameskip)})
    total=len(rows)
    if total<=0: raise ValueError(f'No valid {split_name} rows')
    limit=int(args.max_train_examples if split_name=='train' else args.max_val_examples)
    if limit and limit<total:
        keep=subset_row_indices(rows,limit,int(args.seed)+(3101 if split_name=='train' else 4101),bool(args.balance_goal_offsets)); rows=[rows[int(i)] for i in keep]
    return rows,total

def prepare_batch(batch, lewm, stats, args, device, subgoal_generator=None, subgoal_stats=None, generated_ratio:float=0.0):
    dtype=next(lewm.parameters()).dtype
    hist_pix=image_batch_to_lewm(batch['pixels'][:,:args.history_len].to(device), args.image_size).to(dtype)
    goal_pix=image_batch_to_lewm(batch['goal_pixels'].to(device), args.image_size).to(dtype)
    far_goal_pix=image_batch_to_lewm(batch['far_goal_pixels'].to(device), args.image_size).to(dtype)
    with torch.no_grad():
        hist_z=encode_lewm_context(lewm, hist_pix); goal_z=encode_lewm_context(lewm, goal_pix); far_goal_z=encode_lewm_context(lewm, far_goal_pix)
    if args.conditioning_goal_source == 'far':
        goal_z = far_goal_z
    lowdim=lowdim_from_batch(batch, args.history_len, args.lowdim_keys).to(device); lowdim_n=normalize_lowdim(lowdim, stats)
    goal_offsets=batch['goal_offset'].to(device).float()
    action_offsets=batch['action_offset'].to(device).float()
    if subgoal_generator is not None and float(generated_ratio) > 0:
        gen_stats = subgoal_stats if subgoal_stats is not None else stats
        lowdim_gen=normalize_lowdim(lowdim, gen_stats)
        with torch.no_grad():
            generated=subgoal_generator(hist_z, far_goal_z, lowdim_gen, goal_offsets, action_offsets)['prediction'].float()
        if float(generated_ratio) >= 1.0:
            goal_z=generated
        else:
            mask=(torch.rand(goal_z.size(0), device=device) < float(generated_ratio)).view(-1,1,1)
            goal_z=torch.where(mask, generated, goal_z)
    target_full=target_action_chunk(batch, args.history_len, max(args.action_offsets)//args.frameskip).to(device)
    target_n=normalize_action_blocks(target_full, stats)
    return hist_z, goal_z, far_goal_z, lowdim_n, target_full, target_n, batch['action_tokens'].to(device).long(), goal_offsets, action_offsets

def grouped_loss(model, hist, goal, far_goal, lowdim, target_n, action_tokens, goal_offsets, action_offsets, mode_l1_weight:float):
    total_loss=target_n.new_tensor(0.0); total_count=0; metrics={}
    for L in torch.unique(action_tokens).tolist():
        mask=action_tokens==int(L); count=int(mask.sum().item())
        if count<=0: continue
        try:
            out=model(hist[mask], goal[mask], lowdim[mask], action_horizon=int(L), far_goal_latents=far_goal[mask], goal_offset_steps=goal_offsets[mask], subgoal_offset_steps=action_offsets[mask])
        except TypeError:
            out=model(hist[mask], goal[mask], lowdim[mask], action_horizon=int(L))
        tgt=target_n[mask,:int(L)]
        nll=model.nll(out,tgt); best=model.best_mode_l1(out,tgt); loss=nll+float(mode_l1_weight)*best
        total_loss=total_loss+loss*count; total_count+=count
        update(metrics, {'nll':nll.item(), 'best_mode_l1_norm':best.item(), 'loss':loss.item()}, count)
    return total_loss/max(total_count,1), finalize(metrics)

@torch.no_grad()
def evaluate(model, lewm, loader, stats, args, device, subgoal_generator=None, subgoal_stats=None):
    model.eval(); total={}; by_len={}; rng=torch.Generator(device=device).manual_seed(args.seed+99)
    for bi,batch in enumerate(loader):
        if args.max_val_batches and bi>=args.max_val_batches: break
        eval_ratio=1.0 if (args.eval_use_generated_subgoal and subgoal_generator is not None) else 0.0
        hist,goal,far_goal,lowdim,target,target_n,tokens,goal_offsets,action_offsets=prepare_batch(batch,lewm,stats,args,device,subgoal_generator,subgoal_stats,generated_ratio=eval_ratio)
        for L in torch.unique(tokens).tolist():
            mask=tokens==int(L); count=int(mask.sum().item())
            if count<=0: continue
            try:
                out=model(hist[mask],goal[mask],lowdim[mask],action_horizon=int(L),far_goal_latents=far_goal[mask],goal_offset_steps=goal_offsets[mask],subgoal_offset_steps=action_offsets[mask])
            except TypeError:
                out=model(hist[mask],goal[mask],lowdim[mask],action_horizon=int(L))
            tgt_n=target_n[mask,:int(L)]; tgt=target[mask,:int(L)]
            nll=model.nll(out,tgt_n); top=out['logits'].argmax(dim=-1); gather=top[:,None,None,None].expand(-1,1,int(L),model.action_dim); top_n=out['means'][:,:,:int(L)].gather(1,gather).squeeze(1)
            top_raw=denormalize_action_blocks(top_n, stats); top_l1=torch.abs(top_raw-tgt).mean(dim=(-1,-2))
            try:
                samples_n=model.sample(hist[mask],goal[mask],lowdim[mask],args.eval_samples,generator=rng,action_horizon=int(L),far_goal_latents=far_goal[mask],goal_offset_steps=goal_offsets[mask],subgoal_offset_steps=action_offsets[mask])
            except TypeError:
                samples_n=model.sample(hist[mask],goal[mask],lowdim[mask],args.eval_samples,generator=rng,action_horizon=int(L))
            samples=denormalize_action_blocks(samples_n, stats); sample_l1=torch.abs(samples-tgt.unsqueeze(1)).mean(dim=(-1,-2)); best=sample_l1.min(dim=1).values
            vals={'nll':nll.item(),'top_l1':top_l1.mean().item(),f'best{args.eval_samples}_l1':best.mean().item(),'coverage':(best<=args.coverage_threshold).float().mean().item()}
            update(total, vals, count); update(by_len.setdefault(f'A{int(L)*int(args.frameskip)}',{}), vals, count)
    out={'all':finalize(total)}; out.update({k:finalize(v) for k,v in by_len.items()}); return out

def main():
    args=parse_args(); torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else 'cpu')
    split=load_json(args.split)
    max_tokens=max(args.action_offsets)//int(args.frameskip)
    dataset=load_swm_dataset(args.dataset, cache_dir=args.cache_dir, frameskip=args.frameskip, num_steps=args.history_len+max_tokens, keys_to_load=['pixels','action','proprio','state'])
    train_pool=build_window_specs(dataset, split, 'train', context_len=args.history_len, plan_horizon=max_tokens); val_pool=build_window_specs(dataset, split, 'val', context_len=args.history_len, plan_horizon=max_tokens)
    train_specs=subset_specs(train_pool,args.max_train_windows,args.seed); val_specs=subset_specs(val_pool,args.max_val_windows,args.seed+1)
    train_rows,train_total=build_rows(dataset,train_specs,args,'train'); val_rows,val_total=build_rows(dataset,val_specs,args,'val')
    train_unique_windows=len({int(row['spec_index']) for row in train_rows})
    val_unique_windows=len({int(row['spec_index']) for row in val_rows})
    train_goal_counts=Counter(int(row['goal_offset']) for row in train_rows)
    train_action_counts=Counter(int(row['action_offset']) for row in train_rows)
    train_joint_counts=Counter((int(row['goal_offset']),int(row['action_offset'])) for row in train_rows)
    print(f'variable-action rows train={len(train_rows)}/{train_total} val={len(val_rows)}/{val_total} unique_windows={train_unique_windows}/{val_unique_windows} action_offsets={args.action_offsets} goal_offsets={args.goal_offsets} max_tokens={max_tokens}', flush=True)
    print(f'train_goal_counts={dict(sorted(train_goal_counts.items()))} train_action_counts={dict(sorted(train_action_counts.items()))}', flush=True)
    print(f'train_joint_counts={{{", ".join(f"H{goal}_A{action}: {count}" for (goal, action), count in sorted(train_joint_counts.items()))}}}', flush=True)
    stats_cpu=compute_stats(dataset, split, args.lowdim_keys); stats=move_stats(stats_cpu,device)
    train_data=VariableActionDataset(dataset,train_specs,train_rows,args.history_len,args.frameskip); val_data=VariableActionDataset(dataset,val_specs,val_rows,args.history_len,args.frameskip)
    loader_kwargs={'num_workers':int(args.num_workers),'pin_memory':not args.no_pin_memory}
    if args.num_workers>0:
        loader_kwargs['persistent_workers']=not args.no_persistent_workers
        loader_kwargs['prefetch_factor']=max(1,int(args.prefetch_factor))
    train_loader=DataLoader(train_data,batch_size=args.batch_size,shuffle=True,drop_last=True,**loader_kwargs)
    val_loader=DataLoader(val_data,batch_size=args.batch_size,shuffle=False,**loader_kwargs)
    lewm=load_lewm(args.policy,device=device,bf16=args.bf16)
    subgoal_generator=None; subgoal_stats=None; subgoal_checkpoint=None
    if args.subgoal_generator_checkpoint:
        subgoal_generator, subgoal_stats, subgoal_checkpoint = load_subgoal_prior(args.subgoal_generator_checkpoint, device)
        subgoal_generator.eval()
        print(f"loaded subgoal generator={args.subgoal_generator_checkpoint} generated_ratio={args.generated_subgoal_ratio} eval_generated={args.eval_use_generated_subgoal}", flush=True)
    first=next(iter(DataLoader(train_data,batch_size=1))); hist,goal,far_goal,lowdim,target,target_n,tokens,goal_offsets,action_offsets=prepare_batch(first,lewm,stats,args,device,subgoal_generator,subgoal_stats,generated_ratio=0.0)
    model_config={'latent_dim':int(hist.size(-1)),'lowdim_dim':int(stats_cpu['lowdim_mean'].numel()),'action_dim':int(stats_cpu['action_mean'].numel()*args.frameskip),'max_plan_horizon':int(max_tokens),'hidden_dim':int(args.hidden_dim),'num_heads':int(args.num_heads),'depth':int(args.depth),'num_modes':int(args.num_modes),'pooling':str(args.pooling),'max_goal_offset':int(args.max_goal_offset)}
    checkpoint_architecture='variable_pooled'
    if args.architecture == 'transformer':
        checkpoint_architecture='variable_transformer'
        model_kwargs=dict(model_config)
        model_kwargs.pop('pooling', None)
        model=PushtVariableTransformerGoalPrior(**model_kwargs).to(device)
    else:
        model_config.pop('max_goal_offset', None)
        model=PushtVariablePooledGoalPrior(**model_config).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True); latest=out_dir/'latest.pt'; best=out_dir/'best.pt'; metrics=out_dir/'metrics.jsonl'
    start=0; best_val=float('inf')
    if latest.exists() and not args.no_resume:
        ckpt=torch.load(latest,map_location='cpu',weights_only=False); model.load_state_dict(ckpt['model']); opt.load_state_dict(ckpt['optimizer']); start=int(ckpt.get('epoch',0)); best_val=float(ckpt.get('best_val_nll',best_val)); print(f'resuming {latest} epoch={start}', flush=True)
    manifest={'script':'scripts/pusht/train_pusht_variable_action_prior.py','prior_type':'goal_gmm_v1','dataset':args.dataset,'split':args.split,'args':vars(args),'model_config':dict(model_config, architecture=checkpoint_architecture),'subgoal_generator_checkpoint':args.subgoal_generator_checkpoint,'subgoal_generator_model_config':None if subgoal_checkpoint is None else subgoal_checkpoint.get('model_config'),'window_protocol':'filter_by_max_action_then_add_each_valid_goal_pair','selected_train_windows':len(train_specs),'selected_val_windows':len(val_specs),'selected_train_unique_windows':train_unique_windows,'selected_val_unique_windows':val_unique_windows,'selected_train_rows':len(train_rows),'available_train_rows':train_total,'selected_val_rows':len(val_rows),'available_val_rows':val_total,'train_goal_counts':dict(sorted(train_goal_counts.items())),'train_action_counts':dict(sorted(train_action_counts.items())),'train_joint_counts':{f'H{goal}_A{action}':count for (goal,action),count in sorted(train_joint_counts.items())}}
    (out_dir/'run_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding='utf-8')
    for epoch in range(start,args.epochs):
        model.train(); total={}
        for bi,batch in enumerate(train_loader):
            if args.max_train_batches and bi>=args.max_train_batches: break
            hist,goal,far_goal,lowdim,target,target_n,tokens,goal_offsets,action_offsets=prepare_batch(batch,lewm,stats,args,device,subgoal_generator,subgoal_stats,generated_ratio=args.generated_subgoal_ratio)
            loss,parts=grouped_loss(model,hist,goal,far_goal,lowdim,target_n,tokens,goal_offsets,action_offsets,args.mode_l1_weight)
            opt.zero_grad(set_to_none=True); loss.backward()
            if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            opt.step(); update(total,parts,int(tokens.numel()))
            if bi%20==0:
                row=finalize(total); print(f"epoch={epoch+1} batch={bi} loss={row.get('loss',0):.4f} nll={row.get('nll',0):.4f} best_mode_l1_norm={row.get('best_mode_l1_norm',0):.4f}", flush=True)
        train_metrics=finalize(total); val_metrics=evaluate(model,lewm,val_loader,stats,args,device,subgoal_generator,subgoal_stats); val_nll=float(val_metrics['all']['nll'])
        with metrics.open('a',encoding='utf-8') as f: f.write(json.dumps({'epoch':epoch+1,'train':train_metrics,'val':val_metrics},sort_keys=True)+'\n')
        print(f"epoch={epoch+1} train_loss={train_metrics.get('loss',0):.4f} val_nll={val_nll:.4f} val_top_l1={val_metrics['all']['top_l1']:.4f} val_best{args.eval_samples}_l1={val_metrics['all'][f'best{args.eval_samples}_l1']:.4f} val_cov={val_metrics['all']['coverage']:.4f}", flush=True)
        payload={'prior_type':'goal_gmm_v1','model':model.state_dict(),'optimizer':opt.state_dict(),'stats':stats_cpu,'model_config':dict(model_config, architecture=checkpoint_architecture),'epoch':epoch+1,'best_val_nll':min(best_val,val_nll),'last_train_metrics':train_metrics,'last_val_metrics':val_metrics,'run_manifest':manifest}
        atomic_torch_save(latest,payload)
        if val_nll<best_val:
            best_val=val_nll; atomic_torch_save(best,payload); print(f'wrote new best checkpoint: {best}', flush=True)
        if int(epoch+1) in set(args.save_epochs):
            snapshot=out_dir/f'epoch_{epoch+1:03d}.pt'; atomic_torch_save(snapshot,payload); print(f'wrote epoch snapshot: {snapshot}', flush=True)
        print(f'wrote latest checkpoint: {latest}', flush=True)
    print(f'done latest_epoch={args.epochs} best_val_nll={best_val} wrote={latest}', flush=True)
if __name__=='__main__': main()
