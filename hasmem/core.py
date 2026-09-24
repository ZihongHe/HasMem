"""Frozen-backbone memory, readout adapters, and evaluation."""
from __future__ import annotations
import argparse, copy, hashlib, json, math, os, random, time, traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import VALIDATION_OWNERS, load_data, load_lme
from .stats import summarize
from .metrics import score_prediction

SYSTEM = 'Use only the supplied memory. Answer the question briefly and faithfully. If the memory does not contain the answer, return UNKNOWN.'


def _needs_turn_separators(tokenizer):
    """Mistral-Instruct forbids consecutive user turns after the optional system."""
    try:
        tokenizer.apply_chat_template(
            [{'role': 'system', 'content': 's'},
             {'role': 'user', 'content': 'a'},
             {'role': 'user', 'content': 'b'}],
            tokenize=True, return_dict=False, add_generation_prompt=True,
        )
        return False
    except Exception:
        return True


def write(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def norm(text):
    import re, string
    return re.sub(r'\s+', ' ', re.sub(r'\b(a|an|the)\b', ' ', str(text).lower().translate(str.maketrans('', '', string.punctuation)))).strip()

def scores(pred, gold):
    from collections import Counter
    p, g = norm(pred).split(), norm(gold).split()
    hit = sum((Counter(p) & Counter(g)).values())
    return (2*hit/(len(p)+len(g)) if p and g else float(p==g)), float(norm(pred)==norm(gold))

class ResidualProjection(nn.Module):
    def __init__(self, base, hidden=64, rank=4):
        super().__init__()
        self.base, self.enabled, self.context = base, False, None
        self.ra = nn.Linear(base.in_features, rank, bias=False).float()
        self.rb = nn.Linear(rank, base.out_features, bias=False).float()
        self.ga = nn.Linear(base.in_features, rank, bias=False).float()
        self.gb = nn.Linear(hidden, base.out_features*rank).float()
        self.rank, self.outdim = rank, base.out_features
        self.gates = None
        self.skip_reader = False
        self.skip_global = False
        self.skip_soft = False
        nn.init.zeros_(self.rb.weight)
        nn.init.zeros_(self.gb.weight); nn.init.zeros_(self.gb.bias)

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        if getattr(self, 'skip_reader', False):
            delta = torch.zeros_like(y, dtype=torch.float32)
        else:
            delta = self.rb(self.ra(x.float()))
        if self.context is not None and not getattr(self, 'skip_global', False):
            b = self.gb(self.context.float()).view(-1, self.outdim, self.rank)
            delta = delta + torch.einsum('btr,bor->bto', self.ga(x.float()), b)
        if self.gates is not None:
            delta = delta * self.gates.to(device=delta.device, dtype=delta.dtype).reshape(-1, *([1] * (delta.ndim - 1)))
        return y + delta.to(y.dtype)

class Memory(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,64), nn.GELU())
        self.out = nn.Linear(64, dim)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.gru = nn.GRUCell(64,64)
        self.route = nn.Linear(64,1)
        nn.init.zeros_(self.route.weight); nn.init.constant_(self.route.bias,-4)
        # Learned action advantage. KEEP is reference zero. Negative predicts
        # a beneficial compression under QA+cost objective; no test labels.
        self.risk = nn.Sequential(nn.Linear(64*2+3,64), nn.Tanh(), nn.Linear(64,2))
        nn.init.zeros_(self.risk[-1].weight); nn.init.zeros_(self.risk[-1].bias)

    def reencode(self, values, width, context):
        n = len(values)
        if width == n:
            anchor = values
        elif width < n:
            anchor = F.adaptive_avg_pool1d(values.float().T[None],width)[0].T.to(values.dtype)
        else:
            anchor = F.interpolate(values.float().T[None],size=width,mode='linear',align_corners=False)[0].T.to(values.dtype)
        # Re-encoding sees only retained values, never an original-text handle.
        h = self.enc(anchor.float()) + context[None]
        return anchor + self.out(h).to(anchor.dtype)

def digest_tensor(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()

class Experiment:
    def __init__(self, plan, spec, out):
        self.plan, self.spec, self.out = plan, spec, out
        self.seed = spec['seed']; self.rng = random.Random(self.seed)
        torch.manual_seed(self.seed); torch.cuda.manual_seed_all(self.seed)
        torch.set_num_threads(1)
        self.tok = AutoTokenizer.from_pretrained(spec['model'],local_files_only=bool(plan.get("local_files_only", False)))
        self.model = AutoModelForCausalLM.from_pretrained(spec['model'],local_files_only=bool(plan.get("local_files_only", False)),
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
        for p in self.model.parameters(): p.requires_grad_(False)
        self.base_parameters = list(self.model.parameters())
        self.base_versions = [p._version for p in self.base_parameters]
        self.memory = Memory(self.model.config.hidden_size).cuda()
        self.adapters = []
        for layer in self.model.model.layers[-4:]:
            module = ResidualProjection(layer.self_attn.o_proj).cuda()
            layer.self_attn.o_proj = module
            self.adapters.append(module)
        self.parameters = list(self.memory.parameters())
        for m in self.adapters:
            self.parameters += [p for n,p in m.named_parameters() if not n.startswith('base.')]
        self.optimizer = torch.optim.AdamW(self.parameters,lr=1e-4,weight_decay=0.0)
        self.stops = self.model.generation_config.eos_token_id
        self.stops = sorted(set((self.stops if isinstance(self.stops,list) else [self.stops])+[self.tok.eos_token_id]))
        self.stops = [int(x) for x in self.stops if x is not None]
        self.embed = self.model.get_input_embeddings()
        self.alternate_user_turns = _needs_turn_separators(self.tok)
        self.system_ids = self.tok.apply_chat_template([{'role':'system','content':SYSTEM}],tokenize=True, return_dict=False,add_generation_prompt=False)
        self.data = ({'train': [], 'validation': [], 'lme': [], 'audit': {'inference_only': True}}
                     if plan.get('inference_only') else load_data(self.tok, validation_owners=int(
                         plan.get('configuration', {}).get('validation_owners', VALIDATION_OWNERS))))
        write(out/'DATA_AUDIT.json', self.data['audit'])
        self.coverage = set(); self.actions = {k:0 for k in ('keep','shrink','expand')}
        self.train_log = []; self.gradient_seen = {'writer':False,'controller':False,'global':False,'reader':False}

    def ids(self, s): return self.tok(s,add_special_tokens=False)['input_ids']
    def tensor(self, ids): return torch.tensor(ids,dtype=torch.long,device='cuda')

    def encode_event(self,text):
        # Each event is one native user message. ChatML special-token seams make
        # separate stored messages exactly equal joint native tokenization.
        content = 'Memory record:\n'+text
        system = {'role':'system','content':SYSTEM}
        system_text = self.tok.apply_chat_template([system],tokenize=False,add_generation_prompt=False)
        turn = [system, {'role':'user','content':content}]
        if self.alternate_user_turns:
            turn.append({'role':'assistant','content':''})
        full = self.tok.apply_chat_template(turn,tokenize=False,add_generation_prompt=False)
        if not full.startswith(system_text): raise ValueError('chat_system_not_separable')
        rendered = full[len(system_text):]
        ids = self.ids(rendered)
        offsets = self.tok(rendered,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
        marker = rendered.find('Memory record:')
        if marker < 0:
            raise ValueError('memory_record_marker_missing')
        if text in rendered[marker:]:
            begin = rendered.index(text, marker)
            end = begin+len(text)
        else:
            after = marker + len('Memory record:')
            if after < len(rendered) and rendered[after] == '\n':
                after += 1
            body = self.ids(text)
            lo = next((i for i in range(len(ids) - len(body) + 1) if ids[i:i + len(body)] == body), None)
            if lo is None:
                eligible = [i for i,(a,b) in enumerate(offsets) if a >= after and b > a]
                if not eligible:
                    raise ValueError('empty_memory_body')
                lo, hi = eligible[0], eligible[-1] + 1
                if eligible != list(range(lo, hi)):
                    raise ValueError('noncontiguous_body')
                with torch.no_grad():
                    vals = self.embed(self.tensor(ids[lo:hi])).detach()
                return {'prefix': ids[:lo], 'suffix': ids[hi:], 'values': vals, 'body_ids': ids[lo:hi]}
            begin = offsets[lo][0]
            end = offsets[lo + len(body) - 1][1]
        eligible = [i for i,(a,b) in enumerate(offsets) if a>=begin and b<=end and b>a]
        if not eligible: raise ValueError('empty_memory_body')
        lo,hi = eligible[0],eligible[-1]+1
        if eligible != list(range(lo,hi)): raise ValueError('noncontiguous_body')
        with torch.no_grad(): vals = self.embed(self.tensor(ids[lo:hi])).detach()
        return {'prefix':ids[:lo],'suffix':ids[hi:],'values':vals,'body_ids':ids[lo:hi]}

    def encode_case(self, case):
        return [self.encode_event(s) for s in case['events']]

    def question_ids(self, question):
        # Evaluation prompt never reads answer or age.
        full=self.tok.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':question}],tokenize=True, return_dict=False,add_generation_prompt=True)
        if full[:len(self.system_ids)]!=self.system_ids: raise ValueError('chat_system_token_seam_mismatch')
        return full[len(self.system_ids):]


    def select(self, rollout, question, limit=4):
        bank=rollout['bank']
        with torch.no_grad():
            q=self.embed(self.tensor(self.ids(question))).float().mean(0)
            keys=torch.stack([b['values'].float().mean(0) for b in bank])
            sim=F.cosine_similarity(keys,q[None],dim=-1)
            ids=sim.topk(min(limit,len(bank))).indices.tolist()
        return [bank[i] for i in sorted(ids)]

    def prompt(self, rollout, question):
        parts=[self.embed(self.tensor(self.system_ids))]
        skip_soft=any(getattr(adapter,'skip_soft',False) for adapter in getattr(self,'adapters',[]))
        if not skip_soft:
            for entry in self.select(rollout,question):
                parts.extend((self.embed(self.tensor(entry['prefix'])),entry['values'],self.embed(self.tensor(entry['suffix']))))
        parts.append(self.embed(self.tensor(self.question_ids(question))))
        return torch.cat(parts)

    def mode(self, enabled, states=None, gates=None):
        for a in self.adapters:
            a.enabled=enabled
            a.context=states if enabled else None
            a.gates=gates if enabled else None

    def forward(self, pairs, enabled=True, return_tokens=False):
        values=[]; labels=[]; contexts=[]
        for rollout,q in pairs:
            prompt=self.prompt(rollout,q['question'])
            aid=self.ids(str(q['answer']))
            if getattr(self,'supervise_eos',False): aid=aid+[self.answer_eos]
            if not aid: raise ValueError('empty_answer')
            vals=torch.cat((prompt,self.embed(self.tensor(aid))))
            values.append(vals); labels.append([-100]*len(prompt)+aid); contexts.append(rollout['state'])
        maxlen=max(len(x) for x in values)
        if maxlen>8192: raise ValueError('pilot_context_exceeds_8192_not_truncated')
        padded=torch.stack([F.pad(v,(0,0,0,maxlen-len(v))) for v in values])
        mask=torch.stack([torch.arange(maxlen,device='cuda')<len(v) for v in values]).long()
        lab=torch.tensor([l+[-100]*(maxlen-len(l)) for l in labels],device='cuda')
        pos=(mask.cumsum(-1)-1).clamp_min(0)
        gates=torch.tensor([float(r.get('residual_gate',0.0)) if enabled else 0.0 for r,_ in pairs],
                           device='cuda',dtype=torch.float32)
        self.mode(enabled,torch.stack(contexts),gates)
        valid=lab[:,1:].ne(-100)
        # Same causal decoder/answer CE, without materializing full-vocabulary
        # logits for thousands of unsupervised memory/padding positions.
        hidden=self.model.model(inputs_embeds=padded,attention_mask=mask,position_ids=pos,use_cache=False).last_hidden_state
        selected=hidden[:,:-1][valid]
        targets=lab[:,1:][valid]
        owners=torch.arange(len(pairs),device='cuda')[:,None].expand_as(valid)[valid]
        pieces=[]
        for start in range(0,len(selected),128):
            logits=self.model.lm_head(selected[start:start+128]).float()
            pieces.append(F.cross_entropy(logits,targets[start:start+128],reduction='none'))
        losses=torch.cat(pieces)
        nll=losses.new_zeros(len(pairs)).index_add(0,owners,losses)/valid.sum(-1).clamp_min(1)
        self.mode(False)
        if return_tokens:
            # Per-position gold-token NLL and the gold ids behind it. Storing the
            # span lets any reweighting of the answer be recomputed offline instead
            # of costing another forward pass.
            spans=[losses[owners==i].detach().float().cpu().tolist() for i in range(len(pairs))]
            gold=[[int(t) for t in targets[owners==i].detach().cpu().tolist()] for i in range(len(pairs))]
            return nll,spans,gold
        return nll

    @torch.no_grad()
    def generate(self, rollout, question, enabled=True):
        prompt=self.prompt(rollout,question)[None]
        mask=torch.ones(prompt.shape[:2],device='cuda',dtype=torch.long)
        pos=torch.arange(prompt.shape[1],device='cuda')[None]
        gate=float(rollout.get('residual_gate',0.0)) if enabled else 0.0
        self.mode(enabled,rollout['state'][None],torch.tensor([gate],device='cuda',dtype=torch.float32))
        out=self.model(inputs_embeds=prompt,attention_mask=mask,position_ids=pos,use_cache=True)
        token=out.logits[:,-1].argmax(-1); past=out.past_key_values; generated=[]; stop=False
        for _ in range(64):
            t=int(token.item()); generated.append(t)
            if t in self.stops: stop=True; break
            position=mask.sum(-1,keepdim=True)
            mask=torch.cat((mask,mask.new_ones((1,1))),dim=1)
            out=self.model(input_ids=token[:,None],attention_mask=mask,position_ids=position,past_key_values=past,use_cache=True)
            token=out.logits[:,-1].argmax(-1); past=out.past_key_values
        self.mode(False)
        return self.tok.decode(generated[:-1] if stop else generated,skip_special_tokens=True).strip(),generated,stop

    @torch.no_grad()
    def identity(self):
        records=[]
        for case in self.data['train'][:4]:
            encoded=self.encode_case(case)
            hard=self.rollout(encoded,'hard'); soft=self.rollout(encoded,'noshrink')
            for q in case['questions'][:1]:
                if any(getattr(adapter,'skip_soft',False) for adapter in self.adapters):
                    actual=self.prompt(soft,q['question'])
                    bare=torch.cat((self.embed(self.tensor(self.system_ids)),self.embed(self.tensor(self.question_ids(q['question'])))))
                    if not torch.equal(actual,bare):
                        raise RuntimeError('no_soft_prompt_must_omit_slots')
                    records.append({'skip_soft':True,'prompt_equals_bare':True})
                    continue
                selected=self.select(hard,q['question'])
                # Resolve selected entries by identity, not by gold/evidence id.
                positions=[next(i for i,b in enumerate(hard['bank']) if b is e) for e in selected]
                if self.alternate_user_turns:
                    canonical=list(self.system_ids)
                    for i in positions:
                        ev=encoded[i]
                        canonical += ev['prefix']+ev['body_ids']+ev['suffix']
                    canonical += self.question_ids(q['question'])
                else:
                    messages=[{'role':'system','content':SYSTEM}]
                    for i in positions:
                        messages.append({'role':'user','content':'Memory record:\n'+case['events'][i]})
                    messages.append({'role':'user','content':q['question']})
                    canonical=self.tok.apply_chat_template(messages,tokenize=True, return_dict=False,add_generation_prompt=True)
                hard_values=self.embed(self.tensor(canonical))
                actual=self.prompt(soft,q['question'])
                exact=torch.equal(hard_values,actual)
                if not exact: raise RuntimeError('hard_origin_embedding_or_BPE_identity_failed')
                ids=self.tensor(canonical)[None]; mask=torch.ones_like(ids); pos=torch.arange(ids.shape[1],device='cuda')[None]
                self.mode(False)
                a=self.model(input_ids=ids,attention_mask=mask,position_ids=pos,use_cache=False).logits
                self.mode(True,soft['state'][None],torch.zeros(1,device='cuda'))
                b=self.model(inputs_embeds=actual[None],attention_mask=mask,position_ids=pos,use_cache=False).logits
                diff=float((a-b).abs().max()); same=bool(torch.equal(a.argmax(-1),b.argmax(-1)))
                self.mode(False)
                pa,ia,_=self.generate(hard,q['question'],False); pb,ib,_=self.generate(soft,q['question'],True)
                aid=self.ids(str(q['answer']))
                fullids=self.tensor(canonical+aid)[None]
                fullout=self.model(input_ids=fullids,attention_mask=torch.ones_like(fullids),
                                   position_ids=torch.arange(fullids.shape[1],device='cuda')[None],use_cache=False).logits
                supervised=fullout[0,len(canonical)-1:len(canonical)+len(aid)-1].float()
                reference_ce=F.cross_entropy(supervised,self.tensor(aid))
                efficient_ce=self.forward([(hard,q)],False)[0]
                ce_error=abs(float(reference_ce-efficient_ce))
                record={'embedding_exact':exact,'max_logit_error':diff,'all_argmax_equal':same,'generated_ids_equal':ia==ib,
                        'answer_only_ce_error':ce_error}
                records.append(record)
                good=diff<=0.02 and same and ia==ib and ce_error<=1e-4
                write(self.out/'IDENTITY_PROGRESS.json',{'passed_so_far':good,'records':records})
                if not good: raise RuntimeError('initial_hard_behavior_identity_failed')
        if len(records)!=4: raise RuntimeError('identity_coverage_incomplete')
        write(self.out/'IDENTITY.json',{'passed':True,'records':records,'source':'fit_only','reader_global_on_zero_residual':True})

    def _microbatch_loss(self, cases, first_count):
        """One physical batch, preserving the original effective-batch objective.

        All teacher/rollout/activation locals die at this function boundary;
        only the scalar loss graph survives until its immediate backward call.
        Counterfactual prefix seeds use global owner-draw indices, not microbatch
        indices, so splitting an effective batch cannot alter the action draws.
        """
        widths=[]; packed=[]; teachers=[]; prediction_rows=[]; branch_costs=[]; question_counts=[]
        for offset,case in enumerate(cases):
            draw_count=first_count+offset+1
            encoded=self.encode_case(case); qs=case['questions']
            with torch.no_grad():
                hard=self.rollout(encoded,'hard')
            teachers.extend([(hard,q) for q in qs])
            runs=[self.rollout(encoded,'counterfactual',train=True,forced=action,prefix_seed=draw_count,
                               last_decision_step=len(encoded)-1) for action in (0,1,2)]
            packed.extend([(r,q) for r in runs for q in qs])
            question_counts.append(len(qs))
            full=max(1,hard['cost'])
            branch_costs.append([r['cost']/full for r in runs])
            prediction_rows.append([runs[0]['decisions'][-1]['prediction']] if runs[0]['decisions'] else [])
        with torch.no_grad(): teacher_all=self.forward(teachers,False)
        all_nll=self.forward(packed,True)
        cursor=0; teacher_cursor=0; losses=[]
        for nq,costvalues,predictions in zip(question_counts,branch_costs,prediction_rows):
            question_nll=all_nll[cursor:cursor+3*nq].view(3,nq); cursor+=3*nq
            refs=teacher_all[teacher_cursor:teacher_cursor+nq]; teacher_cursor+=nq
            # Per-question harm penalty: new-fact improvement cannot cancel old
            # harm before the hinge. Every branch uses the actual deployed stack.
            quality=(question_nll+4*F.relu(question_nll-refs[None])).mean(-1)
            costs=torch.tensor(costvalues,device='cuda')
            objective=quality+.05*costs
            target=(objective[1:]-objective[0]).detach()
            ploss=F.smooth_l1_loss(torch.stack(predictions),target[None].expand(len(predictions),-1)) if predictions else objective.sum()*0
            losses.append(quality.mean()+ploss); widths.append(float(costs[1]))
        return torch.stack(losses).mean(), sum(widths)/len(widths)


    @torch.no_grad()
    def evaluate(self):
        rows=[]; audit=[]; complete=True
        modes=list(self.plan.get('configuration',{}).get('eval_modes')
                   or ['hard','noshrink','adaptive','fixed','random'])
        datasets=('validation',) if self.plan.get('configuration',{}).get('skip_lme') else ('validation','lme')
        for dataset in datasets:
            if dataset=='lme':
                # All optimization is finished and checkpoint saved before any
                # LongMemEval labels are loaded. No subsequent model selection.
                cases,lme_audit=load_lme(self.tok)
                self.data['lme']=cases
                write(self.out/'LME_DATA_AUDIT.json',lme_audit)
            cases=self.data[dataset]
            for ci,case in enumerate(cases):
                if time.time()>self.plan['deadline_epoch']-100:
                    complete=False; break
                encoded=self.encode_case(case)
                for mode in modes:
                    rollout=self.rollout(encoded,mode,record_audit=(mode=='adaptive'))
                    qs=case['questions']
                    nll,token_nll,gold_ids=self.forward([(rollout,q) for q in qs],mode!='hard',return_tokens=True)
                    self.supervise_eos=True
                    with_eos=self.forward([(rollout,q) for q in qs],mode!='hard')
                    self.supervise_eos=False
                    answer_lengths=torch.tensor([len(self.ids(str(q['answer']))) for q in qs],device='cuda')
                    eos_nll=with_eos*(answer_lengths+1)-nll*answer_lengths
                    bank=rollout['bank']; vectors=sum(len(e['values'])+len(e['prefix'])+len(e['suffix']) for e in bank)
                    # Only active Soft values and Global hidden/adapter outputs
                    # counted; framing IDs explicit. Shared model params separate.
                    persistent=(sum(8*(len(e['values'])+len(e['prefix'])+len(e['suffix'])) for e in bank) if mode=='hard' else
                                sum(e['values'].numel()*e['values'].element_size()+8*(len(e['prefix'])+len(e['suffix'])) for e in bank)+rollout['state'].numel()*4)
                    for j,q in enumerate(qs):
                        pred,tokens,stopped=self.generate(rollout,q['question'],mode!='hard')
                        f1,em=scores(pred,q['answer'])
                        row={'dataset':case['dataset'],'group':case['group'],'case_id':case['id']+f'/{j}',
                             'condition':mode,'nll':float(nll[j]),'eos_nll':float(eos_nll[j]),'f1':f1,'em':em,'memory_vectors':vectors,
                             'cumulative_vectors':rollout['cost'],'persistent_bytes':persistent,'old':bool((q.get('age',0) or 0)>0),
                             'age_known':q.get('age') is not None,
                             'seed':self.seed,'model':self.spec['label'].rsplit('_s',1)[0],
                             'prediction':pred,'token_ids':tokens,'stopped_on_eos':stopped,
                             'token_nll':[round(x,5) for x in token_nll[j]],'gold_token_ids':gold_ids[j]}
                        if case['dataset'] == 'longmemeval_s':
                            row.update(score_prediction(pred, q['answer'], q.get('question_type', ''), q.get('question_id', '')))
                        rows.append(row)
                        with (self.out/'rows.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    if mode=='adaptive':
                        audit.append({'dataset':case['dataset'],'case_id':case['id'],'actions':rollout['chain']})
                write(self.out/'STATUS.json',{'phase':'evaluation','dataset':dataset,'completed_cases':ci+1,'total_cases':len(cases),'rows':len(rows)})
                del encoded,rollout
            if not complete: break
        write(self.out/'STATE_CHAINS.json',audit)
        summary=summarize(rows)
        summary['question_weighted_metrics'] = {
            dataset: {condition: {
                metric: sum(float(row[metric]) for row in rows if row['dataset'] == dataset and row['condition'] == condition) /
                        sum(1 for row in rows if row['dataset'] == dataset and row['condition'] == condition)
                for metric in ('f1', 'em', 'nll', 'memory_vectors', 'cover', 'brief_cover')
                if all(metric in row for row in rows if row['dataset'] == dataset and row['condition'] == condition)
            } for condition in sorted({row['condition'] for row in rows if row['dataset'] == dataset})}
            for dataset in sorted({row['dataset'] for row in rows})
        }
        summary.update({'complete':complete,'seed':self.seed,'model':self.spec['label'].rsplit('_s',1)[0],
                        'updates':len(self.train_log),'seen_train_owners':len(self.coverage),'gradient_seen':self.gradient_seen,
                        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                        'evaluation_notes': ['F1 and EM are local lexical metrics.',
                                              'Position counts and byte storage are reported separately.']})
        write(self.out/'SUMMARY.json',summary)
        return complete,len(rows)
