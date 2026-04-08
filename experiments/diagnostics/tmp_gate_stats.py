import json, torch
from argparse import Namespace
from _pathfix import ensure_repo_root

ensure_repo_root()

from cd_mambatt.data import load_cmapss_split, fit_normalizer, build_windows, CMAPSSWindowDataset
from train_supervised import build_model, build_loader, infer_device
ROOT='/home/shelterpl/data/CMAPSS'
device=infer_device('auto')

def mk_loader(subset):
    train=load_cmapss_split(ROOT, subset, 'train', rul_clip=125)
    test=load_cmapss_split(ROOT, subset, 'test', rul_clip=125)
    norm=fit_normalizer(train)
    data=build_windows(norm.transform(test),20,stride=1,last_only=True)
    return build_loader(CMAPSSWindowDataset(data),128,False,0)

src_loader=mk_loader('FD001')
tgt_loader=mk_loader('FD003')
modes=[('mixed__token__dt_bc','mixed'),('dual_state__token__dt_bc','dual_state')]
for name,scan in modes:
    args=Namespace(d_model=None,d_state=16,d_conv=8,expand=2,num_mamba_layers=1,num_transformer_layers=3,num_heads=7,dropout=0.5,dim_feedforward=84,transformer_impl='custom',transformer_norm_mode='pre',transformer_inner_dropout=0.0,mamba_block_mode='dd_spd',spd_gate_init_bias=-2.0,spd_scan_mode=scan,spd_gate_mode='token',spd_gate_scheme='dt_bc',spd_predictor_mode='shared_head')
    model=build_model(args,21).to(device)
    ckpt=torch.load(f'/home/shelterpl/cd_mambatt/runs/dual_state_quick_ab/FD001_TO_FD003/{name}/seed_42/cd_stage/best.pt',map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    out={}
    for tag,loader in [('source',src_loader),('target',tgt_loader)]:
        vals={'gate':[],'dt':[],'bc':[]}
        with torch.no_grad():
            for xb,_ in loader:
                aux=model.forward_features_with_aux(xb.to(device))
                vals['gate'].append(float(aux['gate_mean']))
                vals['dt'].append(float(aux['gate_dt_mean']))
                vals['bc'].append(float(aux['gate_bc_mean']))
        out[tag]={k: sum(v)/len(v) for k,v in vals.items()}
    print(name, json.dumps(out))
