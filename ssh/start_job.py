#!/usr/bin/env python3
import typer

from client_lib import logs, Job

cli = typer.Typer()
cli_jobs = typer.Typer()


@cli_jobs.command(
    name="run",
    help=(
        "Submit job to cluster. "
        "If you want to use advanced options you can import client_lib directly from Python script"
    ),
)
def cli_jobs_run(
        script: str,
        base_image: str,
        instance_type: str,
        n_workers: int = 1,
        type: str = "horovod",
):
    result = Job(
        script=script,
        base_image=base_image,
        instance_type=instance_type,
        n_workers=n_workers,
        type=type
    ).submit()
    print(result)


@cli_jobs.command(
    name="logs",
    help=(
        "Print job logs by name. "
    ),
)
def cli_jobs_logs(job_name):
    logs(job_name)


cli.add_typer(cli_jobs, name="jobs")

if __name__ == "__main__":
    cli()
