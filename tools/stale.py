# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors, the AllenNLP library authors.
# All rights reserved.

"""
Script to close stale issue. Taken in part from the AllenNLP repository.
https://github.com/allenai/allennlp.
Copied from https://github.com/huggingface/transformers
"""
from datetime import datetime as dt
import os

from github import Github


LABELS_TO_EXEMPT = [
    "good first issue",
    "feature request",
    "keep"
]


def main():
    g = Github(os.environ["GITHUB_TOKEN"])
    repo = g.get_repo("huggingface/datasets-server")
    open_issues = repo.get_issues(state="open")

    for issue in open_issues:
        comments = sorted([comment for comment in issue.get_comments()], key=lambda i: i.created_at, reverse=True)
        last_comment = comments[0] if len(comments) > 0 else None
        if (
            last_comment is not None and last_comment.user.login == "github-actions[bot]"
            and (dt.utcnow() - issue.updated_at).days > 7
            and (dt.utcnow() - issue.created_at).days >= 30
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            # print(f"Would close issue {issue.number} since it has been 7 days of inactivity since bot mention.")
            issue.edit(state="closed")
        elif (
            (dt.utcnow() - issue.updated_at).days > 23
            and (dt.utcnow() - issue.created_at).days >= 30
            and not any(label.name.lower() in LABELS_TO_EXEMPT for label in issue.get_labels())
        ):
            # print(f"Would add stale comment to {issue.number}")
            issue.create_comment(
                "This issue has been automatically marked as stale because it has not had "
                "recent activity. If you think this still needs to be addressed "
                "please comment on this thread.\n\nPlease note that issues that do not follow the "
                "[contributing guidelines](https://github.com/huggingface/datasets-server/blob/main/CONTRIBUTING.md) "
                "are likely to be ignored."
            )


if __name__ == "__main__":
    main()
