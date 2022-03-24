test_params =\
    {
        'mcorr':
            {
                'mcorr_kwargs':
                    {
                        'max_shifts': [24, 24],
                        'strides': [48, 48],
                        'overlaps': [24, 24],
                        'max_deviation_rigid': 3,
                        'border_nan': 'copy',
                        'pw_rigid': True,
                        'gSig_filt': None
                    }
            },

        'cnmf':
            {
                'cnmf_kwargs':
                    {
                        'p': 2,
                        'nb': 1,
                        # raises error: no parameter 'merge_thresh' found
                        # 'merge_thresh': 0.7,
                        'rf': None,
                        'stride': 30,
                        'K': 10,
                        'gSig': [5,5],
                        'ssub': 1,
                        'tsub': 1,
                        'method_init': 'greedy_roi',
                    },
                'eval_kwargs':
                    {
                        'min_SNR': 2.50,
                        'rval_thr': 0.8,
                        'use_cnn': True,
                        'min_cnn_thr': 0.8,
                        'cnn_lowest': 0.1,
                        'decay_time': 1,
                    },
                'refit': True,
            },
        'cnmfe': None,
    }
