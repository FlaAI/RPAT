def setup(mode, P):
    fname = f'{P.dataset}_{P.model}_{mode}_{P.augment_type}'

    if mode == 'ce':
        from .ce import train
    elif mode == 'ce_rand':
        from .ce_rand import train
        fname += f'_rand'
    else:
        raise NotImplementedError()

    fname += f'_seed_{P.seed}'
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train, fname
