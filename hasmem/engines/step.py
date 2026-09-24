"""Fixed-step and configurable curriculum training for HasMem."""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, time, traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from ..core import Experiment as BaseExperiment, write, digest_tensor
from ..eos import derive_answer_eos
from ..search import beam_search
from ..compatibility import inference_metadata, resolve_backbone_tag, validate_checkpoint_metadata

from .common import (
    choose_intervention,
    least_harm_shrink,
    arm_flags,
    force_unkeep_rate,
    unkeep_target_rate,
    rate_band_penalty,
    dynamic_keep_penalty
)
BACKBONE_TAGS = ('14B', '13B', '8B', '7B', '3B', '1.5B', '0.5B')
LARGE_BACKBONES = {'3B', '7B', '8B', '13B', '14B'}


def backbone_tag(model_path):
    name = str(model_path).replace('\\', '/').rstrip('/').split('/')[-1]
    for tag in BACKBONE_TAGS:
        if tag in name:
            return tag
    return 'unknown'


def is_large_backbone(model_path):
    return backbone_tag(model_path) in LARGE_BACKBONES


def resolve_flags(spec):
    """Per-job overlays on top of the named arm. Missing keys keep curriculum defaults."""
    base = arm_flags(spec.get('arm', 'curriculum'))
    writer = bool(spec['writer']) if 'writer' in spec else base['codec']
    hinge = bool(spec['hinge']) if 'hinge' in spec else base['codec']
    slot = bool(spec['slot']) if 'slot' in spec else base['slot']
    force = bool(spec['force']) if 'force' in spec else base['curriculum']
    band = bool(spec['band']) if 'band' in spec else base['curriculum']
    return {
        'quota': base['quota'],
        'codec': writer,
        'hinge': hinge,
        'slot': slot,
        'dynpen': base['dynpen'],
        'curriculum': force or band or base['curriculum'],
        'force': force,
        'band': band,
        'writer': writer,
        'teacher': spec.get('teacher', 'least_harm'),
        'force_unkeep_start': float(spec.get('force_unkeep_start', 0.75)),
        'force_unkeep_end': float(spec.get('force_unkeep_end', 0.15)),
        'unkeep_target_start': float(spec.get('unkeep_target_start', 0.50)),
        'unkeep_target_end': float(spec.get('unkeep_target_end', 0.22)),
        'unkeep_band': float(spec.get('unkeep_band', 0.12)),
        'band_weight': float(spec.get('band_weight', 0.6)),
        'shrink_frac': float(spec.get('shrink_frac', 0.10)),
        'use_global': bool(spec.get('use_global', True)),
        'use_reader': bool(spec.get('use_reader', True)),
        'use_soft': bool(spec.get('use_soft', True)),
        'deploy_mode': spec.get('deploy_mode') or None,
        'open_residual': bool(spec.get('open_residual', False)),
        'origin': spec.get('origin') or 'hard',
        'target_width_ratio': float(spec.get('target_width_ratio', 0.91)),
        'hard_anchor': bool(spec.get('hard_anchor', False)),
        'hard_anchor_weight': float(spec.get('hard_anchor_weight', 0.5)),
        'hard_anchor_margin': float(spec.get('hard_anchor_margin', 0.0)),
    }


def recency_shrink(records, max_harm=None):
    """Forced unkeep teacher: shrink the oldest realized slot, not least-harm."""
    best = None
    for rec in records or []:
        realized = list(rec.get('realized_sequence') or rec.get('sequence') or [])
        if 1 not in realized:
            continue
        savings = rec.get('cost_savings_vs_keep')
        if savings is None or float(savings) <= 0:
            continue
        harm = float(rec.get('max_question_harm') or 0.)
        if max_harm is not None and harm > float(max_harm):
            continue
        first = next(index for index, action in enumerate(realized) if action == 1)
        key = (first, harm, -float(savings))
        if best is None or key < best[0]:
            best = (key, rec, realized[first], first, harm)
    if best is None:
        return None
    return {'record': best[1], 'action': best[2], 'divergence': best[3], 'harm': best[4],
            'sequence': list(best[1].get('sequence') or best[1].get('realized_sequence'))}


class Experiment(BaseExperiment):
    def __init__(self, plan, spec, out):
        super().__init__(plan,spec,out)
        self.arm=spec['arm']; self.supervise_eos=False
        self.answer_eos=derive_answer_eos(self.tok)
        if self.answer_eos not in self.stops: raise RuntimeError('native_end_not_generation_stop')
        # Same added causal streak feature and initialization in both arms.
        self.memory.risk=nn.Sequential(nn.Linear(64*2+4,64),nn.Tanh(),nn.Linear(64,2)).cuda()
        nn.init.zeros_(self.memory.risk[-1].weight); nn.init.zeros_(self.memory.risk[-1].bias)
        self.parameters=list(self.memory.parameters())
        for a in self.adapters:
            self.parameters += [p for n,p in a.named_parameters() if not n.startswith('base.')]
        self.optimizer=torch.optim.AdamW(self.parameters,lr=1e-4,weight_decay=0.)
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        self.backbone_tag = resolve_backbone_tag(spec, self.model.config)
        self.large_model = self.backbone_tag in LARGE_BACKBONES
        self.flags = resolve_flags(spec)
        self.prompt_slots = int(spec.get('prompt_slots', 4))
        if self.prompt_slots < 1: raise ValueError('prompt_slots_must_be_positive')
        for adapter in self.adapters:
            adapter.skip_reader = not self.flags['use_reader']
            adapter.skip_global = not self.flags['use_global']
            adapter.skip_soft = not self.flags['use_soft']
            if not self.flags['use_reader']:
                adapter.ra.weight.requires_grad_(False)
                adapter.rb.weight.requires_grad_(False)
            if not self.flags['use_global']:
                adapter.ga.weight.requires_grad_(False)
                adapter.gb.weight.requires_grad_(False)
                adapter.gb.bias.requires_grad_(False)
        self.maintenance = 0 if plan.get('smoke_only') else (6 if self.large_model else 12)
        self.policy_started=None
        self.policy_span_seconds=1.
        self.unkeep_ema=0.
        self.search_counts={'owners':0,'positive':0,'no_positive':0,'evaluations':0,
                            'qualified_candidates':0,'keep_penalty_terms':0,
                            'preupdate_predicted_keep':0,'preupdate_predicted_shrink':0,
                            'preupdate_predicted_expand':0,'positive_predicted_nonkeep':0,
                            'negative_predicted_nonkeep':0,'positive_weight_sum':0.,
                            'position_buckets':{'early':0,'middle':0,'late':0},
                            'keep_streak_buckets':{'0':0,'3':0,'8plus':0},
                            'length_buckets':{'short':0,'medium':0,'long':0},
                            'maintenance_decisions':self.maintenance,
                            'keep_invariant_passed':False,
                            'weak_positive':0,'slot_upweighted':0,'codec_hinge_terms':0,
                            'keep_wrong':0,'noisy_slot':0,
                            'dynamic_keep_penalty_sum':0.,'dynamic_keep_penalty_terms':0,
                            'keep_rate_ema':1.,'forced_unkeep':0,'force_rate':0.75,
                            'unkeep_target':0.5,'unkeep_ema':0.,'prompt_slots':self.prompt_slots}
        self.keep_rate_ema=1.
        self.warmup_fingerprint=None
        self.shared_initializer=None

    def rollout(self, encoded, mode, train=False, forced=None, record_audit=False,
                prefix_seed=0, last_decision_step=None, forced_sequence=None, suffix_start=None,
                keep_prefix_from=None, maintenance=None):
        extra=self.maintenance if maintenance is None else maintenance
        bank=[]; state=torch.zeros(64,device='cuda'); costs=0; decisions=[]; chain=[]; streak=0
        last_feature=None; n_events=len(encoded)
        for step in range(n_events+extra):
            if step<n_events:
                event=encoded[step]; incoming=event['values']
                feature=self.memory.enc(incoming.float()).mean(0)
                gate=torch.sigmoid(self.memory.route(feature)); proposed=self.memory.gru(feature[None],state[None])[0]
                state=state+gate*(proposed-state); last_feature=feature
                # Exact hard embedding on arrival. KEEP later is a no-op on this tensor.
                bank.append({'values':incoming,'prefix':event['prefix'],'suffix':event['suffix'],
                             'label':event.get('label',''),'arrival':step,'initial':len(incoming),
                             'origin_sha':digest_tensor(incoming)})
            else:
                feature=last_feature if last_feature is not None else self.memory.enc(bank[0]['values'].float()).mean(0)
            if step:
                idx=(step-1)//2 % len(bank); item=bank[idx]; old=item['values']; n=len(old)
                metadata=torch.tensor([math.log1p(step-item['arrival']),n/max(1,item['initial']),math.log1p(step),math.log1p(min(streak,8))],device='cuda')
                risk=self.memory.risk(torch.cat((self.memory.enc(old.float()).mean(0),feature,metadata)))
                action=0
                if mode=='adaptive' and self.flags.get('deploy_mode') in ('keep_open','fixed_ratio','length_rule'):
                    mode=self.flags['deploy_mode']
                if mode=='adaptive': action=int(torch.cat((risk.new_zeros(1),risk)).argmin().item())
                elif mode=='fixed': action=1
                elif mode=='random': action=random.Random(self.seed+step*7919).randrange(3)
                elif mode=='forced': action=int(forced)
                elif mode=='keep_open': action=0
                elif mode=='fixed_ratio':
                    target=float(self.flags.get('target_width_ratio',0.91))
                    action=1 if n>4 and (n/max(1,item['initial']))>target+1e-6 else 0
                elif mode=='length_rule':
                    target=float(self.flags.get('target_width_ratio',0.91))
                    ratios=[len(b['values'])/max(1,b['initial']) for b in bank]
                    mean_r=sum(ratios)/max(1,len(ratios))
                    this_r=n/max(1,item['initial'])
                    action=1 if n>4 and mean_r>target+1e-6 and this_r>=max(ratios)-1e-12 else 0
                elif mode=='counterfactual':
                    if not train or last_decision_step is None: raise ValueError('counterfactual_fit_only')
                    action=int(forced) if step==last_decision_step else random.Random(prefix_seed+step*7919).choice([0,1,1])
                elif mode=='beamplan':
                    if not train or suffix_start is None or forced_sequence is None: raise ValueError('beam_fit_only')
                    if suffix_start<=step<suffix_start+len(forced_sequence):
                        action=int(forced_sequence[step-suffix_start])
                    elif keep_prefix_from is not None and keep_prefix_from<=step<suffix_start:
                        action=0
                    else:
                        action=int(torch.cat((risk.new_zeros(1),risk)).argmin().item())
                elif mode not in ('hard','noshrink'): raise ValueError(mode)
                requested=action; width=n
                frac=float(getattr(self,'flags',{}).get('shrink_frac',0.10))
                if action==1 and n>4: width=max(4,n-max(1,math.ceil(n*frac)))
                if action==2: width=min(item['initial'],n+max(1,math.ceil(n*frac)))
                if width==n: action=0
                if record_audit: before=digest_tensor(old)
                if width!=n: item['values']=self.memory.reencode(old,width,state)
                decisions.append({'prediction':risk,'action':action,'requested_action':requested,'step':step,
                                  'keep_streak_before':streak,'width_before':n,'width_after':width})
                if record_audit:
                    chain.append({'step':step,'entry':idx,'action':action,'requested_action':requested,
                                  'keep_streak_before':streak,'before_sha':before,'after_sha':digest_tensor(item['values']),
                                  'before_length':n,'after_length':len(item['values'])})
                streak=min(8,streak+1) if action==0 else 0
                if train: self.actions[('keep','shrink','expand')[action]]+=1
            costs+=sum(len(b['values'])+len(b['prefix'])+len(b['suffix']) for b in bank)
        residual_gate=1.0 if any(d['action']!=0 for d in decisions) else 0.0
        if self.flags.get('open_residual') and mode not in ('hard','noshrink'):
            residual_gate=1.0
        return {'bank':bank,'state':state,'cost':costs,'decisions':decisions,'chain':chain,'keep_streak':streak,
                'residual_gate':residual_gate,'maintenance':extra}

    def encode_case(self, case):
        events=super().encode_case(case)
        if self.flags.get('origin','hard')!='random':
            return events
        if not hasattr(self,'_random_origin_scale'):
            weight=self.model.get_input_embeddings().weight.detach().float()
            self._random_origin_scale=float(weight.std().clamp_min(1e-4))
        out=[]
        case_key=str(case.get('id','') or case.get('group',''))
        for i,event in enumerate(events):
            vals=event['values']
            digest=hashlib.sha256(f'{self.seed}:{case_key}:{i}:{tuple(vals.shape)}'.encode()).digest()
            g=torch.Generator(device=vals.device)
            g.manual_seed(int.from_bytes(digest[:8],'little')%(2**31))
            noise=torch.randn(vals.shape,generator=g,device=vals.device,dtype=vals.dtype)*self._random_origin_scale
            event=dict(event)
            event['values']=noise
            out.append(event)
        return out

    def strategy_loss(self, cases, first_count):
        """QA + action-imitation on a locked deploy rule. Same owner batch as curriculum."""
        mode=self.flags.get('deploy_mode')
        if mode not in ('keep_open','fixed_ratio','length_rule'):
            raise ValueError('strategy_loss_requires_deploy_mode')
        losses=[]; ratios=[]
        for offset,case in enumerate(cases):
            encoded=self.encode_case(case); qs=case['questions']
            with torch.no_grad():
                hard=self.rollout(encoded,'hard')
            rollout=self.rollout(encoded,mode,train=True)
            loss=self.forward([(rollout,q) for q in qs],True).mean()
            for decision in rollout['decisions']:
                risk=decision['prediction']
                logits=torch.cat((risk.new_zeros(1),-risk))
                target=int(decision['action'])
                loss=loss+F.cross_entropy(logits[None],torch.tensor([target],device=logits.device))
            losses.append(loss)
            ratios.append(rollout['cost']/max(1,hard['cost']))
            unkeep=float(any(d['action']!=0 for d in rollout['decisions']))
            self.unkeep_ema=0.95*getattr(self,'unkeep_ema',0.)+0.05*unkeep
            self.search_counts['owners']=self.search_counts.get('owners',0)+1
            self.search_counts['unkeep_ema']=self.unkeep_ema
            self.search_counts['deploy_mode']=mode
            self.search_counts['strategy_draw']=first_count+offset+1
        return torch.stack(losses).mean(),sum(ratios)/len(ratios)

    def identity(self):
        super().identity()
        case=self.data['train'][0]; q=case['questions'][0]
        encoded=self.encode_case(case)
        with torch.no_grad():
            hard=self.rollout(encoded,'hard'); prompt=self.prompt(hard,q['question'])
            aid=self.ids(str(q['answer']))+[self.answer_eos]
            values=torch.cat((prompt,self.embed(self.tensor(aid))))[None]
            self.mode(False)
            logits=self.model(inputs_embeds=values,attention_mask=torch.ones(values.shape[:2],device='cuda',dtype=torch.long),
                              position_ids=torch.arange(values.shape[1],device='cuda')[None],use_cache=False).logits
            per=F.cross_entropy(logits[0,len(prompt)-1:len(prompt)+len(aid)-1].float(),self.tensor(aid),reduction='none')
            self.supervise_eos=True; efficient=self.forward([(hard,q)],False)[0]; self.supervise_eos=False
            content=self.forward([(hard,q)],False)[0]
            combined_error=abs(float(efficient-per.mean())); content_error=abs(float(content-per[:-1].mean()))
            if max(combined_error,content_error)>1e-4: raise RuntimeError('EOS_loss_alignment_failed')
            write(self.out/'EOS_IDENTITY.json',{'native_answer_eos':self.answer_eos,'combined_ce_error':combined_error,
                  'content_ce_error':content_error,'eos_ce':float(per[-1]),'passed':True})

    def codec_fingerprint(self):
        h=hashlib.sha256()
        for name,p in self.memory.named_parameters():
            if not name.startswith('risk.'):
                h.update(name.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        for index,a in enumerate(self.adapters):
            for name,p in a.named_parameters():
                if not name.startswith('base.'):
                    h.update(f'{index}:{name}'.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def controller_fingerprint(self):
        h=hashlib.sha256()
        for name,p in self.memory.risk.named_parameters():
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def freeze_codec(self):
        self.warmup_fingerprint=self.codec_fingerprint()
        for p in self.parameters: p.requires_grad_(False)
        for p in self.memory.risk.parameters(): p.requires_grad_(True)
        trainable=[p for p in self.memory.risk.parameters()]
        writer_open=self.flags['writer']
        if writer_open:
            for p in self.memory.out.parameters(): p.requires_grad_(True)
            for p in self.memory.enc.parameters(): p.requires_grad_(True)
            trainable += [p for p in self.memory.out.parameters()] + [p for p in self.memory.enc.parameters()]
        self.optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=0.)
        self.policy_started=time.time()
        self.policy_span_seconds=max(60., (self.plan['deadline_epoch']-1200)-time.time())
        write(self.out/'WARMUP.json',{'updates':self.warmup_updates,'codec_sha256':self.warmup_fingerprint,
                                    'controller_sha256':self.controller_fingerprint(),
                                    'shared_initializer':self.shared_initializer,
                                    'codec_frozen_for_policy_phase':not writer_open,
                                    'writer_open_on_shrink':writer_open,'eos_supervised':True,
                                    'prompt_slots':self.prompt_slots,
                                    'policy_span_seconds':self.policy_span_seconds})


    def encode_case(self, case):
        encoded=super().encode_case(case)
        labels=case.get('event_labels') or ['']*len(encoded)
        for event,label in zip(encoded,labels):
            event['label']=label
        return encoded

    def select(self, rollout, question, limit=None):
        if limit is None:
            limit=int(getattr(self,'prompt_slots',4))
        return super().select(rollout, question, limit=limit)

    def policy_progress(self):
        target=float(getattr(self,'policy_target_updates',5081) or 5081)
        step=float(getattr(self,'policy_step',0) or 0)
        if step<=0 or target<=0:
            return 0.
        return min(1.,max(0.,step/target))

    def gold_followed(self, rollout, qs):
        follows=[]
        for q in qs:
            prompt=self.prompt(rollout,q['question'])
            aid=self.ids(str(q['answer']))
            if self.supervise_eos: aid=aid+[self.answer_eos]
            if not aid:
                follows.append(False); continue
            values=torch.cat((prompt,self.embed(self.tensor(aid))))[None]
            mask=torch.ones(values.shape[:2],device=values.device,dtype=torch.long)
            pos=torch.arange(values.shape[1],device=values.device)[None]
            self.mode(False)
            logits=self.model(inputs_embeds=values,attention_mask=mask,position_ids=pos,use_cache=False).logits
            pred=logits[0,len(prompt)-1:len(prompt)+len(aid)-1].argmax(-1)
            follows.append(bool(torch.equal(pred,self.tensor(aid))))
        return follows

    def slot_is_noisy(self, case, decision, n_events):
        labels=case.get('event_labels') or []
        targets={q.get('target_label') for q in case.get('questions') or []}
        step=int(decision['step'])
        bank_len=n_events if step>=n_events else step+1
        if bank_len<1 or not labels: return False
        idx=(step-1)//2 % bank_len
        if idx>=len(labels): return False
        label=labels[idx]
        return bool(label) and label not in targets

    def search_loss(self,cases,first_count):
        losses=[]; ratios=[]
        for offset,case in enumerate(cases):
            draw=first_count+offset+1; qs=case['questions']
            with torch.no_grad():
                encoded=self.encode_case(case); hard=self.rollout(encoded,'hard')
                hard_nll=self.forward([(hard,q) for q in qs],False).tolist()
                choice=choose_intervention(draw,len(encoded),2,self.maintenance)
                depth,start=choice['depth'],choice['start']
                position=choice['position_bucket']
                def score(sequence):
                    r=self.rollout(encoded,'beamplan',train=True,prefix_seed=draw,forced_sequence=sequence,
                                   suffix_start=start,keep_prefix_from=choice['keep_prefix_from'])
                    nll=self.forward([(r,q) for q in qs],True).tolist()
                    window=r['decisions'][start-1:start-1+depth]
                    local_streak=0 if window[-1]['action']!=0 else window[-1]['keep_streak_before']+1
                    return {'per_question_nll':nll,'cumulative_cost':r['cost']/max(1,hard['cost']),
                            'realized_sequence':[d['action'] for d in window],
                            'keep_streak_penalty':.02*min(local_streak/3.,1.)}
                result=beam_search(depth,score,hard_teacher_nll=hard_nll)
            arm=getattr(self,'arm','control')
            flags=self.flags
            keep_wrong=False; noisy=False
            if flags['slot']:
                keep_wrong=not all(self.gold_followed(hard,qs))
            records=result.get('records') or []
            pick_shrink=recency_shrink if flags.get('teacher')=='recency' else least_harm_shrink
            weak=None
            progress=self.policy_progress()
            force_rate=force_unkeep_rate(progress, flags['force_unkeep_start'], flags['force_unkeep_end'])
            target_rate=unkeep_target_rate(progress, flags['unkeep_target_start'], flags['unkeep_target_end'])
            forced=False
            if (not result['positive']) and flags['force']:
                forced=random.Random(draw*10007).random()<force_rate
                if forced:
                    weak=pick_shrink(records)
            elif not flags['dynpen']:
                if (not result['positive']) and flags['quota']:
                    weak=pick_shrink(records)
                elif (not result['positive']) and flags['slot'] and keep_wrong:
                    weak=pick_shrink(records,max_harm=0.05)
            if result['positive']:
                winner=result['winner_sequence']; at=result['first_divergence']; target=int(result['action_target']); weak_used=False
            elif weak is not None:
                winner=weak['sequence']; at=weak['divergence']; target=int(weak['action']); weak_used=True
            else:
                winner=result['winner_sequence']; at=0; target=0; weak_used=False
            replay=self.rollout(encoded,'beamplan',train=True,prefix_seed=draw,forced_sequence=winner,
                                suffix_start=start,keep_prefix_from=choice['keep_prefix_from'])
            decision=replay['decisions'][start-1+at]
            if (result['positive'] or weak_used) and decision['action']!=target: raise RuntimeError('positive_target_is_noop')
            if flags['slot']:
                noisy=self.slot_is_noisy(case,decision,len(encoded))
            risk=decision['prediction']; logits=torch.cat((risk.new_zeros(1),-risk))
            predicted=int(logits.detach().argmax().item())
            predicted_key=('preupdate_predicted_keep','preupdate_predicted_shrink','preupdate_predicted_expand')[predicted]
            self.search_counts[predicted_key]=self.search_counts.get(predicted_key,0)+1
            self.search_counts['positive_predicted_nonkeep']=self.search_counts.get('positive_predicted_nonkeep',0)+int(result['positive'] and predicted!=0)
            self.search_counts['negative_predicted_nonkeep']=self.search_counts.get('negative_predicted_nonkeep',0)+int(not result['positive'] and predicted!=0)
            seen_positive=self.search_counts['positive']+int(result['positive'])
            seen_negative=self.search_counts['no_positive']+int(not result['positive'])
            if result['positive']:
                positive_weight=min(16.,max(1.,seen_negative/max(1,seen_positive)))
            elif weak_used and flags['force']:
                positive_weight=0.45*(1.-progress)+0.15*progress
            elif weak_used:
                positive_weight=0.25
            else:
                positive_weight=1.
            if flags['slot'] and (keep_wrong or noisy):
                positive_weight=min(32.,positive_weight*4.)
                self.search_counts['slot_upweighted']=self.search_counts.get('slot_upweighted',0)+1
            self.search_counts['positive_weight_sum']=self.search_counts.get('positive_weight_sum',0.)+(positive_weight if result['positive'] or weak_used else 0.)
            loss=positive_weight*F.cross_entropy(logits[None],torch.tensor([target],device='cuda'))
            penalty_weight=.02*min((decision['keep_streak_before']+1)/3.,1.) if result['positive'] else 0.
            loss=loss+penalty_weight*logits.softmax(-1)[0]
            dyn_penalty=0.
            if flags['force'] or flags['band']:
                side,scale=('none',0.)
                if flags['band']:
                    side,scale=rate_band_penalty(getattr(self,'unkeep_ema',0.),target_rate,flags['unkeep_band'])
                probs=logits.softmax(-1)
                if side=='keep':
                    dyn_penalty=flags['band_weight']*scale
                    loss=loss+dyn_penalty*probs[0]
                elif side=='unkeep':
                    dyn_penalty=flags['band_weight']*scale
                    loss=loss+dyn_penalty*(1.-probs[0])
                self.unkeep_ema=0.95*getattr(self,'unkeep_ema',0.)+0.05*float(predicted!=0)
                self.search_counts['dynamic_keep_penalty_sum']=self.search_counts.get('dynamic_keep_penalty_sum',0.)+dyn_penalty
                self.search_counts['dynamic_keep_penalty_terms']=self.search_counts.get('dynamic_keep_penalty_terms',0)+int(dyn_penalty>0)
                self.search_counts['forced_unkeep']=self.search_counts.get('forced_unkeep',0)+int(forced and weak_used)
                self.search_counts['force_rate']=force_rate
                self.search_counts['unkeep_target']=target_rate
                self.search_counts['unkeep_ema']=self.unkeep_ema
                self.search_counts['policy_progress']=progress
            elif flags['dynpen']:
                dyn_penalty=dynamic_keep_penalty(self.keep_rate_ema,decision['keep_streak_before'])
                loss=loss+dyn_penalty*logits.softmax(-1)[0]
                self.search_counts['dynamic_keep_penalty_sum']=self.search_counts.get('dynamic_keep_penalty_sum',0.)+dyn_penalty
                self.search_counts['dynamic_keep_penalty_terms']=self.search_counts.get('dynamic_keep_penalty_terms',0)+int(dyn_penalty>0)
                self.keep_rate_ema=0.9*self.keep_rate_ema+0.1*float(predicted==0)
                self.search_counts['keep_rate_ema']=self.keep_rate_ema
            if flags['hinge']:
                hinge_seq=None
                realized=list(result.get('winner_realized_sequence') or [])
                if target==1 or (result['positive'] and 1 in realized):
                    hinge_seq=winner
                else:
                    aux=pick_shrink(records,max_harm=0.2)
                    if aux is not None: hinge_seq=aux['sequence']
                if hinge_seq is not None:
                    shrink_r=replay if list(hinge_seq)==list(winner) else self.rollout(
                        encoded,'beamplan',train=True,prefix_seed=draw,forced_sequence=hinge_seq,
                        suffix_start=start,keep_prefix_from=choice['keep_prefix_from'])
                    if shrink_r.get('residual_gate',0)>0:
                        with torch.no_grad():
                            keep_r=self.rollout(encoded,'beamplan',train=True,prefix_seed=draw,
                                                forced_sequence=(0,)*depth,suffix_start=start,
                                                keep_prefix_from=choice['keep_prefix_from'])
                            keep_nll=self.forward([(keep_r,q) for q in qs],True)
                        shrink_nll=self.forward([(shrink_r,q) for q in qs],True)
                        loss=loss+0.5*F.relu(shrink_nll-keep_nll.detach()-1e-4).mean()
                        self.search_counts['codec_hinge_terms']=self.search_counts.get('codec_hinge_terms',0)+1
                        if flags.get('hard_anchor'):
                            # KEEP is itself already softened upstream, so the relative hinge lets
                            # drift accumulate. Anchor the softened path to the hard rollout too.
                            anchor=torch.tensor(hard_nll,device=shrink_nll.device,dtype=shrink_nll.dtype)
                            term=F.relu(shrink_nll-anchor-flags['hard_anchor_margin']).mean()
                            loss=loss+flags['hard_anchor_weight']*term
                            self.search_counts['hard_anchor_terms']=self.search_counts.get('hard_anchor_terms',0)+1
                            self.search_counts['hard_anchor_sum']=self.search_counts.get('hard_anchor_sum',0.)+float(term.detach())
            losses.append(loss); ratios.append(replay['cost']/max(1,hard['cost']))
            self.search_counts['owners']+=1; self.search_counts['positive']+=int(result['positive'])
            self.search_counts['no_positive']+=int(not result['positive']); self.search_counts['evaluations']+=result['evaluated_count']
            self.search_counts['qualified_candidates']+=result['qualified_count']; self.search_counts['keep_penalty_terms']+=int(penalty_weight>0)
            self.search_counts['weak_positive']=self.search_counts.get('weak_positive',0)+int(weak_used)
            self.search_counts['keep_wrong']=self.search_counts.get('keep_wrong',0)+int(keep_wrong)
            self.search_counts['noisy_slot']=self.search_counts.get('noisy_slot',0)+int(noisy)
            self.search_counts['position_buckets'][choice['position_bucket']]+=1
            streak_key='8plus' if decision['keep_streak_before']>=8 else ('3' if decision['keep_streak_before']>=3 else '0')
            self.search_counts['keep_streak_buckets'][streak_key]+=1
            self.search_counts['length_buckets'][case.get('length_bucket','short')]=self.search_counts['length_buckets'].get(case.get('length_bucket','short'),0)+1
            record={'draw':draw,'fit_group':case['group'],'hard_teacher_nll':hard_nll,'keep_streak_before':decision['keep_streak_before'],
                    'suffix_start':start,'window_position_fraction':position,'position_bucket':choice['position_bucket'],
                    'target_keep_streak':choice['target_keep_streak'],'keep_prefix_from':choice['keep_prefix_from'],
                    'supervised_decision_step':decision['step'],'length_bucket':case.get('length_bucket'),
                    'supervised_action':target,'positive_class_weight':positive_weight,'weak_label':weak_used,
                    'keep_wrong':keep_wrong,'noisy_slot':noisy,'arm':arm,
                    'preupdate_predicted_action':predicted,'keep_penalty_weight':penalty_weight,
                    'dynamic_keep_penalty':dyn_penalty,'keep_rate_ema':getattr(self,'keep_rate_ema',1.),
                    'forced_unkeep':forced,'force_rate':force_rate,'unkeep_target':target_rate,
                    'policy_progress':progress,'search':result}
            with (self.out/'SEARCH_TEACHER.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
        return torch.stack(losses).mean(),sum(ratios)/len(ratios)

    def stratified_owner_order(self):
        buckets={'short':[],'medium':[],'long':[]}
        for index,case in enumerate(self.data['train']):
            buckets.setdefault(case.get('length_bucket','short'),[]).append(index)
        for key in buckets:
            self.rng.shuffle(buckets[key])
        order=[]; indexes={key:0 for key in ('short','medium','long')}
        remaining=sum(len(values) for values in buckets.values())
        while remaining:
            progressed=False
            for key in ('short','medium','long'):
                cursor=indexes[key]
                if cursor<len(buckets[key]):
                    order.append(buckets[key][cursor]); indexes[key]+=1; remaining-=1; progressed=True
            if not progressed:
                break
        return order

    def mode(self, enabled, states=None, gates=None):
        for adapter in self.adapters:
            adapter.enabled = enabled
            adapter.context = None if (not enabled or getattr(adapter, 'skip_global', False)) else states
            adapter.gates = gates if enabled else None

    def keep_invariant(self):
        case=self.data['train'][0]; q=case['questions'][0]
        encoded=self.encode_case(case)
        with torch.no_grad():
            hard=self.rollout(encoded,'hard',maintenance=0)
            kept=self.rollout(encoded,'noshrink',maintenance=0)
            if kept['residual_gate']!=0 or any(d['action']!=0 for d in kept['decisions']):
                raise RuntimeError('all_keep_must_remain_zero_residual')
            for entry in kept['bank']:
                if digest_tensor(entry['values'])!=entry['origin_sha']:
                    raise RuntimeError('keep_mutated_hard_embedding')
            prompt_h=self.prompt(hard,q['question']); prompt_k=self.prompt(kept,q['question'])
            if not torch.equal(prompt_h,prompt_k):
                raise RuntimeError('keep_prompt_diverged_from_hard')
            mask=torch.ones(prompt_h[None].shape[:2],device='cuda',dtype=torch.long)
            pos=torch.arange(prompt_h.shape[0],device='cuda')[None]
            self.mode(False)
            a=self.model(inputs_embeds=prompt_h[None],attention_mask=mask,position_ids=pos,use_cache=False).logits
            self.mode(True,kept['state'][None],torch.zeros(1,device='cuda'))
            b=self.model(inputs_embeds=prompt_k[None],attention_mask=mask,position_ids=pos,use_cache=False).logits
            self.mode(False)
            diff=float((a-b).abs().max())
            if diff>1e-4:
                raise RuntimeError(f'keep_hard_logit_invariant_failed:{diff}')
        write(self.out/'KEEP_INVARIANT.json',{'passed':True,'max_logit_error':diff,'source':'fit_only_post_train'})
        self.search_counts['keep_invariant_passed']=True

    def train(self):
        started=time.time()
        locked=bool(self.spec.get('max_updates') or self.plan.get('max_updates'))
        wall=float((self.plan.get('configuration') or {}).get('wall_hours') or 6)
        train_cap=started+(wall*3600 if locked else min(wall, 6)*3600)
        deadline=min(float(self.plan['deadline_epoch']), train_cap)-1200
        self.supervise_eos=True
        self.warmup_updates=1 if self.plan.get('smoke_only') else 384
        self.policy_target_updates=1 if self.plan.get('smoke_only') else int(self.spec.get('policy_updates') or self.plan.get('policy_updates') or 5081)
        self.policy_step=0
        self._band_select=None
        max_updates=2 if self.plan.get('smoke_only') else int(self.spec.get('max_updates') or self.plan.get('max_updates') or 10000)
        order=self.stratified_owner_order()
        slots=int(getattr(self,'prompt_slots',4))
        if self.plan.get('smoke_only'):
            physical,accumulation=8,1
        elif self.backbone_tag in {'14B', '13B'}:
            physical,accumulation=1,8
        elif self.large_model:
            if slots>=12:
                physical,accumulation=1,8
            elif slots>=8:
                physical,accumulation=2,4
            else:
                physical,accumulation=4,2
        else:
            physical,accumulation=8,1
        effective=physical*accumulation
        if effective!=8:
            raise RuntimeError(f'effective_owners_must_be_8_got_{effective}')
        count=0
        for update in range(max_updates):
            if time.time()>deadline: break
            if update==self.warmup_updates:
                self.freeze_codec()
            self.optimizer.zero_grad(set_to_none=True); lossval=0.; cost=0.
            policy=update>=self.warmup_updates
            self.policy_step=0 if not policy else (update-self.warmup_updates+1)
            strategy=self.flags.get('deploy_mode') in ('keep_open','fixed_ratio','length_rule')
            for micro in range(accumulation):
                first=count; cases=[self.data['train'][order[(count+i)%len(order)]] for i in range(physical)]; count+=physical
                self.coverage.update(c['group'] for c in cases)
                if strategy:
                    loss,ratio=self.strategy_loss(cases,first)
                    if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite_loss')
                    lossval+=float(loss.detach())/accumulation; cost+=ratio/accumulation
                    (loss/accumulation).backward(); del loss
                elif policy:
                    for i,case in enumerate(cases):
                        loss,ratio=self.search_loss([case],first+i)
                        if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite_loss')
                        scale=accumulation*physical
                        lossval+=float(loss.detach())/scale; cost+=ratio/scale
                        (loss/scale).backward(); del loss
                else:
                    loss,ratio=self._microbatch_loss(cases,first)
                    if not bool(torch.isfinite(loss)): raise RuntimeError('nonfinite_loss')
                    lossval+=float(loss.detach())/accumulation; cost+=ratio/accumulation
                    (loss/accumulation).backward(); del loss
                del cases
            groups={'writer':self.memory.out.parameters(),'controller':self.memory.risk.parameters(),
                    'global':self.adapters[0].gb.parameters(),'reader':self.adapters[0].rb.parameters()}
            for name,params in groups.items():
                self.gradient_seen[name] |= any(p.grad is not None and bool(p.grad.abs().max()>0) for p in params)
            torch.nn.utils.clip_grad_norm_([p for p in self.parameters if p.requires_grad],1.)
            self.optimizer.step()
            row={'update':update+1,'phase':'policy' if policy else 'warmup','loss':lossval,
                 'seen_owners':len(self.coverage),'compression_cost_ratio':cost}
            self.train_log.append(row)
            write(self.out/'STATUS.json',dict(row,seconds=time.time()-started,search_counts=self.search_counts,
                 physical_owners=physical,gradient_accumulation=accumulation,effective_owners=effective,
                 prompt_slots=slots,arm=self.arm,seed=self.seed,flags=self.flags,
                 backbone_tag=self.backbone_tag,policy_step=self.policy_step,
                 vram_allocated_mb=int(torch.cuda.memory_allocated()/1024/1024),
                 vram_reserved_mb=int(torch.cuda.memory_reserved()/1024/1024)))
            if policy and bool(self.spec.get('select_in_band', self.plan.get('select_in_band'))):
                ema=float(getattr(self,'unkeep_ema',0.))
                if 0.22<=ema<=0.40 and (self._band_select is None or lossval<self._band_select['ce']):
                    torch.save(self._checkpoint_payload(len(self.train_log)), self.out/'checkpoint_selected.pt')
                    self._band_select={'update':len(self.train_log),'ce':float(lossval),'unkeep_ema':ema,
                                       'policy_progress':self.policy_progress()}
        self.optimizer.zero_grad(set_to_none=True)
        if len(self.train_log)<=self.warmup_updates: raise RuntimeError('no_completed_policy_phase')
        if (not self.flags['writer']) and self.codec_fingerprint()!=self.warmup_fingerprint:
            raise RuntimeError('codec_changed_in_policy_phase')
        if any(p.requires_grad for p in self.base_parameters) or [p._version for p in self.base_parameters]!=self.base_versions:
            raise RuntimeError('frozen_base_was_modified')
        last=self._checkpoint_payload(len(self.train_log))
        torch.save(last, self.out/'checkpoint_last.pt')
        selected=self._band_select
        if selected and (self.out/'checkpoint_selected.pt').is_file():
            torch.save(torch.load(self.out/'checkpoint_selected.pt', map_location='cpu', weights_only=True),
                       self.out/'checkpoint.pt')
            selection={'used':'band_best_ce','band':[0.22,0.40],**selected}
        else:
            torch.save(last, self.out/'checkpoint.pt')
            selection={'used':'last_iterate','band':[0.22,0.40],'reason':'no_iterate_in_unkeep_band'}
        write(self.out/'SELECTION.json',selection)
        write(self.out/'TRAINING.json',{'updates':len(self.train_log),'warmup_updates':self.warmup_updates,
              'policy_updates':len(self.train_log)-self.warmup_updates,'seen_owners':len(self.coverage),'pool_owners':len(self.data['train']),
              'seconds':time.time()-started,'effective_owners':effective,'physical_owners':physical,'gradient_accumulation':accumulation,
              'prompt_slots':slots,'policy_target_updates':self.policy_target_updates,
              'progress_basis':'policy_update_over_5081',
              'gradient_seen':self.gradient_seen,'search_counts':self.search_counts,'warmup_codec_sha256':self.warmup_fingerprint,
              'shared_initializer':self.shared_initializer,'warmup_reused':False,'writer_open_on_shrink':self.flags['writer'],
              'flags':self.flags,'selection':selection,
              'final_codec_sha256':self.codec_fingerprint(),'actions':self.actions,'trace':self.train_log})
        self.supervise_eos=False

    def _checkpoint_payload(self, updates):
        return {'memory':self.memory.state_dict(),
                'adapters':[{k:v for k,v in a.state_dict().items() if not k.startswith('base.')} for a in self.adapters],
                'seed':self.seed,'arm':self.arm,'updates':updates,'protocol':'hasmem_step',
                'inference_metadata':inference_metadata(self)}

    def load_trained_checkpoint(self):
        """Restore a finished run's Memory and adapters so evaluation can be
        repeated without retraining. Mirrors _checkpoint_payload."""
        state=torch.load(Path(self.spec['checkpoint']),map_location='cuda',weights_only=True)
        compatibility=validate_checkpoint_metadata(state.get('inference_metadata'), inference_metadata(self),
                    bool(self.plan.get('allow_legacy_checkpoint', False)))
        self.memory.load_state_dict(state['memory'],strict=True)
        if len(state['adapters'])!=len(self.adapters): raise ValueError('checkpoint_adapter_count')
        for adapter,weights in zip(self.adapters,state['adapters']):
            result=adapter.load_state_dict(weights,strict=False)
            if result.unexpected_keys or any(not k.startswith('base.') for k in result.missing_keys):
                raise ValueError('checkpoint_adapter_mismatch')
        if int(state.get('seed',self.seed))!=int(self.seed): raise ValueError('checkpoint_seed_mismatch')
        self.train_log=list(state.get('trace') or [])
        write(self.out/'CHECKPOINT_LOADED.json',{'checkpoint':str(self.spec['checkpoint']),
              'updates':state.get('updates'),'arm':state.get('arm'),'protocol':state.get('protocol'),
              'compatibility':compatibility,
              'validation_cases':len(self.data['validation']),
              'validation_questions':sum(len(c['questions']) for c in self.data['validation'])})

    def evaluate(self):
        self.supervise_eos=False
        return super().evaluate()
