from execute import *


def parse_exec_args():
    import argparse
    parser = argparse.ArgumentParser(description='Execute file')

    parser.set_defaults(app='c')
    parser.add_argument('model', help='model_id', choices=model_dict.keys())
    parser.set_defaults(suffix='backbone')

    parser.set_defaults(generate_gcn_files=False)
    parser.add_argument('--gen_gcn', help='generate only the GCN files (requires IAD files to already exist)',
                        dest='generate_gcn_files', action='store_true')
    parser.set_defaults(generate_files=False)
    parser.add_argument('--gen', help='generate_files', dest='generate_files', action='store_true')
    parser.set_defaults(eval_only=False)
    parser.add_argument('--eval', help='evaluate only', dest='eval_only', action='store_true')

    parser.add_argument('--frames', help='number of frames', default=64, type=int)
    parser.add_argument('--repeat', help='repeat code runs', default=1, type=int)
    parser.add_argument('--application', help='application', default="tea_making", choices=application_list)

    return parser.parse_args()


def exec_different_bottleneck_sizes(args, lfd_params):
    for bn in [8, 16, 32, 64]:
        lfd_params.model_save_dir = "saved_models_"+str(bn)
        lfd_params.model.bottleneck_size = bn
        for r in range(args.repeat):
            execute_func(args, lfd_params, r, backbone=True)


if __name__ == '__main__':
    args = parse_exec_args()

    lfd_params = default_model_params()
    lfd_params.set_application(args.application)

    lfd_params.set_model_params(model_dict[args.model], end_point=-1)
    lfd_params.input_frames = args.frames

    exec_different_bottleneck_sizes(args, lfd_params)
