from engine.argdefault import default_argument_parser
from engine.defaults import default_setup
from config.config import get_cfg
from point_rent.config import add_pointrend_config


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg
if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    cfg = setup(args)
    print(cfg)