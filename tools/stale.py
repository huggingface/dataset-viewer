# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors, the AllenNLP library authors.
# All rights reserved.

"""
Script to close stale issue. Taken in part from the AllenNLP repository.
https://github.com/allenai/allennlp.
Copied from https://github.com/huggingface/transformers
"""
from datetime import datetime as dt, timezone
import os

from github import Github
# ^ PyGithub - https://pygithub.readthedocs.io/en/stable/introduction.html


LABELS_TO_EXEMPT_IN_LOWERCASE = [label.lower() for label in [
    "P0",
    "P1",
    "P2"
]]


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("huggingface/datasets-server")
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        now = dt.now(timezone.utc)
        if (
            (now - issue.created_at).days < 30
            or any(label.name.lower() in LABELS_TO_EXEMPT_IN_LOWERCASE for label in issue.get_labels())
        ):
            continue
        comments = sorted(list(issue.get_comments()), key=lambda i: i.created_at, reverse=True)
        last_comment = comments[0] if len(comments) > 0 else None
        if (
            last_comment is not None
            and last_comment.user.login == "github-actions[bot]"
            and (now - issue.updated_at).days > 7
        ):
            # close issue since it has been 7 days of inactivity since bot mention
            issue.edit(state="closed")
        elif (
            (now - issue.updated_at).days > 23
        ):
            #add stale comment
            issue.create_comment(
                "This issue has been automatically marked as stale because it has not had "
                "recent activity. If you think this still needs to be addressed "
                "please comment on this thread.\n\nPlease note that issues that do not follow the "
                "[contributing guidelines](https://github.com/huggingface/datasets-server/blob/main/CONTRIBUTING.md) "
                "are likely to be ignored."
            )


if __name__ == "__main__":
    main()
