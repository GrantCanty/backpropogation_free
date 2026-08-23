from scripts.run_sparse_nystrom_rank_sweep import build_parser


def test_rank_sweep_defaults_to_sparse_large_width_development_grid() -> None:
    args = build_parser().parse_args([])

    assert args.widths == (1024,)
    assert args.ranks == (64, 96, 128, 192, 256)
    assert args.regularizations == (0.1, 1.0, 10.0)
    assert args.seeds == (100, 101, 102, 103, 104)
    assert args.sparse_fan_in == 8


def test_rank_sweep_accepts_explicit_comma_separated_grid() -> None:
    args = build_parser().parse_args(
        ["--widths", "512,1024", "--ranks", "64,128", "--regularizations", "0.1,1"]
    )

    assert args.widths == (512, 1024)
    assert args.ranks == (64, 128)
    assert args.regularizations == (0.1, 1.0)
