import json


def unwrap(result):
    if hasattr(result,'json'):
        result = result.json
        if callable(result):
            result = result()
    if isinstance(result,str):
        result = json.loads(result)
    return result.get('res',result)


def freeze_engine(engine):
    if hasattr(engine,'eval'):
        engine.eval()
    if hasattr(engine,'parameters'):
        for parameter in engine.parameters():
            parameter.requires_grad_(False)
