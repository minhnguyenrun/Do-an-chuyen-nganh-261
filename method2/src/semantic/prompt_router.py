PROMPTS = ('Text Recognition:', 'Table Recognition:', 'Formula Recognition:')


def route_prompt(label):
    return {'table':PROMPTS[1],'formula':PROMPTS[2],'display_formula':PROMPTS[2]}.get(label,PROMPTS[0])
