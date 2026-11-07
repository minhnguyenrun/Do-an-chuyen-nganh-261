"""Run measured predictions or live A1-A9 inference on a JSON page dataset."""
import argparse
import asyncio
import json
import time
import sys
from pathlib import Path
import cv2
import numpy as np
from .evaluate_refinement import run_ablation_suite
from ..pipeline import parse_page,load_config
from ..runtime import create_adapters


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Dual-Skip ablation evaluation')
    parser.add_argument('dataset',type=Path)
    parser.add_argument('--configurations',nargs='+',default=['A1','A2','A3','A4','A5','A6','A7','A8','A9'])
    parser.add_argument('--live',action='store_true',help='run real model inference instead of precomputed predictions')
    parser.add_argument('--glm-device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--glm-model', default='zai-org/GLM-OCR')
    parser.add_argument('--output',type=Path,default=Path('outputs/ablation.json'))
    args = parser.parse_args()
    runner = None
    if args.live:
        if 'A0' in args.configurations:
            parser.error('A0 needs official SDK predictions; omit --live to evaluate those records')
        local_config = load_config({'models':{'semantic_parser':args.glm_model},'inference':{'device':args.glm_device}})
        adapters = create_adapters(local_config)
        adapters[-1].initialize()
        prediction_dir = args.output.parent/'predictions'
        prediction_dir.mkdir(parents=True,exist_ok=True)
        def live_runner(record,name,overrides):
            path = args.dataset/record['image']
            image = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'cannot read dataset image: {path}')
            start = time.perf_counter()
            config = load_config(overrides)
            config['models']['semantic_parser'] = args.glm_model
            config['inference']['device'] = args.glm_device
            result = asyncio.run(parse_page(image,config,*adapters))
            output = result.model_dump(mode='json')
            output['latency_seconds'] = time.perf_counter()-start
            # Keep the complete provenance for inspection, not just aggregate scores.
            (prediction_dir/f'{path.stem}-{name}.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
            return output
        runner = live_runner
    result = run_ablation_suite(args.dataset,args.configurations,runner=runner)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(args.output.resolve())


if __name__ == '__main__':
    main()
