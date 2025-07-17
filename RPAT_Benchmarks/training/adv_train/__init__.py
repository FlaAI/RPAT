def setup(mode, P):
    fname = f'{P.dataset}_{P.model}_{mode}_{P.adv_method}_{P.distance}_{P.augment_type}'

    if mode == 'adv_train':
        if P.consistency:
            if P.RA:
                from .adv_consistency_ra import train
                fname += f'_consistency_ra'
            else:
                from .adv_consistency import train
                fname += f'_consistency'
        elif P.RA:
            from .adv_train_ra import train
            fname += f'_ra'
        else:
            from .adv_train import train

    elif mode == 'adv_trades':
        if P.RA:
            from .adv_trades_ra import train
            fname += f'_ra'
        else:
            from .adv_trades import train

    elif mode == 'adv_mart':
        if P.RA:
            from .adv_mart_ra import train
            fname += f'_ra'
        else:
            from .adv_mart import train

    else:
        raise NotImplementedError()

    fname += f'_seed_{P.seed}'
    if P.suffix is not None:
        fname += f'_{P.suffix}'

    return train, fname
