

def parse_primitives(names, all_primitives):
    p = all_primitives if names == 'all' else names if isinstance(names, list
        ) else [names]
    assert set(p) <= set(all_primitives)
    return p
