import argparse
import asyncio
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import yaml
from .pipeline import parse_page,load_config
from .runtime import create_adapters


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='Parse a page with Semantic Dual Skip v4')
    parser.add_argument('image',type=Path)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs'))
    parser.add_argument('--mock',action='store_true',help='synthetic outputs, no model inference')
    parser.add_argument('--glm-device', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--glm-model', help='Hugging Face model ID or local model directory')
    args = parser.parse_args()
    overrides = yaml.safe_load(args.config.read_text(encoding='utf-8')) if args.config else {}
    cfg = load_config(overrides)
    if args.glm_device:
        cfg['inference']['device'] = args.glm_device
    if args.glm_model:
        cfg['models']['semantic_parser'] = args.glm_model
    try:
        image = cv2.imdecode(np.fromfile(args.image,dtype=np.uint8),cv2.IMREAD_COLOR)
    except (OSError,cv2.error):
        image = None
    if image is None:
        parser.error(f'cannot read image: {args.image}')
    adapters = create_adapters(cfg,mock_mode=args.mock)
    adapters[-1].initialize()
    result = asyncio.run(parse_page(image,cfg,*adapters))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/(args.image.stem+'.json')).write_text(result.model_dump_json(indent=2),encoding='utf-8')
    (args.output/(args.image.stem+'.md')).write_text(result.markdown,encoding='utf-8')
    print(f'Parsed {len(result.final_regions)} regions; GLM requests: {result.glm_requests}; output: {args.output.resolve()}')


if __name__ == '__main__':
    main()
