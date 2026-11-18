"""Local smoke/batch entry point; the notebooks call the same Experiment."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    from src.cascaded.types import Config
    from src.cascaded.runner import Experiment, prepare_records
    from src.cascaded.dataset import doclaynet_processed_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant',choices=['s','v3'],default='s')
    parser.add_argument('--split',default='valid')
    parser.add_argument('--count',type=int,default=1)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--mock',action='store_true')
    parser.add_argument('--wbf',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/cascaded_layout')
    parser.add_argument('--no-mineru',action='store_true')
    args=parser.parse_args()
    cfg=Config(variant=args.variant,device=args.device,mock=args.mock,enable_wbf=args.wbf,
               enable_mineru=not args.no_mineru)
    records,annotation=prepare_records(ROOT,doclaynet_processed_dir(ROOT,args.split),args.split,args.count,mock=args.mock)
    exp=Experiment(cfg,records,args.output)
    for phase in ('paddle','yolo','mineru'):
        print(exp.run_phase(phase),flush=True)
    print('RESULTS:',exp.root,flush=True)
    if annotation and not cfg.mock:
        from src.cascaded.report import evaluate
        report=evaluate(exp.results(),annotation,exp.root/'report_literal',run_manifest=exp.manifest_path)
        print(report['rows'],flush=True)


if __name__=='__main__':
    main()
