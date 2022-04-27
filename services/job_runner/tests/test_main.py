from job_runner.main import MAX_JOBS_PER_DATASET


def test_dummy() -> None:
    assert MAX_JOBS_PER_DATASET > 0
