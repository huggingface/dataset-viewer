import subprocess
from pathlib import Path


def cli_all_projects() -> None:
    all_projects_dir = Path(__file__).resolve().parents[-5]
    projects = [
        p
        for d in all_projects_dir.glob("*")
        if not d.name.startswith(".") and d.is_dir()
        for p in list(d.glob("*")) + [d]
        if (p / "pyproject.toml").exists()
    ]
    projects.sort(key=lambda p: (p.name != "libcommon", "libs" not in p.parts, p))

    print(f"Multi CLI for projects: {', '.join(str(project) for project in projects)}")
    while (command := input(">>> ")) not in ["exit", "exit()"]:
        for project in projects:
            print(f"cd {project}")
            subprocess.run(command.split(" "), cwd=project)
