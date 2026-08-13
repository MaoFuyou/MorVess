"""Stage-II entry point for the MorVess training loop."""

from trainer_hq_parse import trainer_run as _trainer_run


def trainer_run(args, model, snapshot_path, multimask_output, low_res):
    return _trainer_run(args, model, snapshot_path, multimask_output, low_res, stage="stage2")
