# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
A script to run multinode training with submitit.
"""
import os
import uuid
from pathlib import Path
from omegaconf.dictconfig import DictConfig
import submitit
import rich


def get_shared_folder() -> Path:
    p = Path(f"{os.getcwd()}/slurm_log/")
    p.mkdir(exist_ok=True, parents=True)
    return p


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as detection

        self._setup_gpu_args()
        detection.main(self.args)

    def checkpoint(self):

        self.args.dist_url = str(get_init_file().as_uri())
        self.args.resume = True
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        # from pathlib import Path

        job_env = submitit.JobEnvironment()
        # self.args.output_dir = Path(
        #     str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(
            f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}"
        )


def submitit_main(args: DictConfig):
    if not args.job_name:
        args.job_name = args.name

    if args.job_dir == "":
        args.job_dir = str(get_shared_folder() / args.job_name / "%j")

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir,
                                     slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    # args.gpus = args.ngpus

    kwargs = {
        "additional_parameters":
        dict(mail_type="FAIL", mail_user="asrafulashiq@gmail.com")
    }

    executor.update_parameters(job_name=args.job_name,
                               mem_per_cpu=args.mem_per_cpu,
                               num_gpus=num_gpus_per_node,
                               ntasks_per_node=num_gpus_per_node,
                               cpus_per_task=args.cpus_per_task,
                               nodes=nodes,
                               time=args.timeout,
                               slurm_signal_delay_s=120,
                               **kwargs)

    args.dist_url = get_init_file().as_uri()
    args.accelerator = 'ddp'
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("_" * 60)
    print("_" * 60)
    rich.print("job_id :", job.job_id)
    rich.print("sbatch script : ", job.paths.submission_file)
    rich.print("stderr : ", job.paths.stderr)
    rich.print("stdout : ", job.paths.stdout)
