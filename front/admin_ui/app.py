import json
import os
import urllib.parse
from itertools import product

import duckdb
import gradio as gr
import huggingface_hub as hfh
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests
from libcommon.processing_graph import processing_graph
from tqdm.contrib.concurrent import thread_map

matplotlib.use("SVG")

DEV = os.environ.get("DEV", False)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
PROD_DV_ENDPOINT = os.environ.get(
    "PROD_DV_ENDPOINT", "https://datasets-server.huggingface.co"
)
DEV_DV_ENDPOINT = os.environ.get("DEV_DV_ENDPOINT", "http://localhost:8100")
ADMIN_HF_ORGANIZATION = os.environ.get("ADMIN_HF_ORGANIZATION", "huggingface")
HF_TOKEN = os.environ.get("HF_TOKEN")

DV_ENDPOINT = DEV_DV_ENDPOINT if DEV else PROD_DV_ENDPOINT


# global state (shared with all the user sessions)
pending_jobs_df = None


def healthcheck():
    try:
        response = requests.head(f"{DV_ENDPOINT}/admin/healthcheck", timeout=10)
    except requests.ConnectionError as error:
        return f"❌ Failed to connect to {DV_ENDPOINT} (error {error})"
    if response.status_code == 200:
        return f"*Connected to {DV_ENDPOINT}*"
    else:
        return f"❌ Failed to connect to {DV_ENDPOINT} (error {response.status_code})"


def draw_graph(width, height):
    graph = processing_graph._nx_graph

    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    fig = plt.figure(figsize=(width, height))
    nx.draw_networkx(graph, pos=pos, node_color="#d1b2f8", node_size=500)
    return fig


with gr.Blocks() as demo:
    gr.Markdown("## Datasets-server admin page")
    gr.Markdown(healthcheck)

    with gr.Row(visible=HF_TOKEN is None) as auth_page:
        with gr.Column():
            auth_title = gr.Markdown(
                "Enter your token ([settings](https://huggingface.co/settings/tokens)):"
            )
            token_box = gr.Textbox(
                HF_TOKEN or "", label="token", placeholder="hf_xxx", type="password"
            )
            auth_error = gr.Markdown("", visible=False)

    with gr.Row(visible=HF_TOKEN is not None) as main_page:
        with gr.Column():
            welcome_title = gr.Markdown("### Welcome")
            with gr.Tab("Home dashboard"):
                home_dashboard_fetch_button = gr.Button("Fetch")
                gr.Markdown("### Dataset infos")
                home_dashboard_trending_datasets_infos_by_builder_name_table = (
                    gr.DataFrame(
                        pd.DataFrame(
                            {
                                "Builder name": [],
                                "Count": [],
                                r"% of all datasets with infos": [],
                                r"% of all public datasets": [],
                            }
                        )
                    )
                )
                gr.Markdown("### Trending datasets coverage (is-valid)")
                home_dashboard_trending_datasets_coverage_stats_table = gr.DataFrame(
                    pd.DataFrame(
                        {
                            "Num trending datasets": [],
                            "HTTP Status": [],
                            "Preview": [],
                            "Viewer": [],
                            "Search": [],
                            "Filter": [],
                            "Statistics": [],
                        }
                    )
                )
                home_dashboard_trending_datasets_coverage_table = gr.DataFrame(
                    pd.DataFrame(
                        {
                            "All trending datasets": [],
                            "HTTP Status": [],
                            "Preview": [],
                            "Viewer": [],
                            "Search": [],
                            "Filter": [],
                            "Statistics": [],
                        }
                    )
                )

                def fetch_home_dashboard(token):
                    out = {
                        home_dashboard_trending_datasets_infos_by_builder_name_table: gr.DataFrame(
                            value=None
                        ),
                        home_dashboard_trending_datasets_coverage_stats_table: gr.DataFrame(
                            value=None
                        ),
                        home_dashboard_trending_datasets_coverage_table: gr.DataFrame(
                            value=None
                        ),
                    }
                    headers = {"Authorization": f"Bearer {token}"}
                    response = requests.get(
                        f"{DV_ENDPOINT}/admin/num-dataset-infos-by-builder-name",
                        headers=headers,
                        timeout=60,
                    )
                    if response.status_code == 200:
                        num_infos_by_builder_name = response.json()
                        total_num_infos = sum(num_infos_by_builder_name.values())
                        num_public_datasets = sum(
                            1 for _ in hfh.HfApi(endpoint=HF_ENDPOINT).list_datasets()
                        )
                        out[
                            home_dashboard_trending_datasets_infos_by_builder_name_table
                        ] = gr.DataFrame(
                            visible=True,
                            value=pd.DataFrame(
                                {
                                    "Builder name": list(
                                        num_infos_by_builder_name.keys()
                                    ),
                                    "Count": list(num_infos_by_builder_name.values()),
                                    r"% of all datasets with infos": [
                                        f"{round(100 * num_infos / total_num_infos, 2)}%"
                                        for num_infos in num_infos_by_builder_name.values()
                                    ],
                                    r"% of all public datasets": [
                                        f"{round(100 * num_infos / num_public_datasets, 2)}%"
                                        for num_infos in num_infos_by_builder_name.values()
                                    ],
                                }
                            ),
                        )
                    else:
                        out[
                            home_dashboard_trending_datasets_infos_by_builder_name_table
                        ] = gr.DataFrame(
                            visible=True,
                            value=pd.DataFrame(
                                {
                                    "Error": [
                                        f"❌ Failed to fetch dataset infos from {DV_ENDPOINT} (error {response.status_code})"
                                    ]
                                }
                            ),
                        )
                    response = requests.get(
                        f"{HF_ENDPOINT}/api/trending?type=dataset&limit=20", timeout=60
                    )
                    if response.status_code == 200:
                        trending_datasets = [
                            repo_info["repoData"]["id"]
                            for repo_info in response.json()["recentlyTrending"]
                        ]

                        def get_is_valid_response(dataset: str):
                            return requests.get(
                                f"{DV_ENDPOINT}/is-valid?dataset={dataset}",
                                headers=headers,
                                timeout=60,
                            )

                        is_valid_responses = thread_map(
                            get_is_valid_response,
                            trending_datasets,
                            desc="get_is_valid_response",
                        )
                        trending_datasets_coverage = {"All trending datasets": []}
                        error_datasets = []
                        unauthorized_datasets = []
                        for dataset, is_valid_response in zip(
                            trending_datasets, is_valid_responses
                        ):
                            if is_valid_response.status_code == 200:
                                response_json = is_valid_response.json()
                                trending_datasets_coverage[
                                    "All trending datasets"
                                ].append(dataset)
                                for is_valid_field in response_json:
                                    pretty_field = is_valid_field.replace(
                                        "_", " "
                                    ).capitalize()
                                    if pretty_field not in trending_datasets_coverage:
                                        trending_datasets_coverage[pretty_field] = []
                                    trending_datasets_coverage[pretty_field].append(
                                        "✅"
                                        if response_json[is_valid_field] is True
                                        else "❌"
                                    )
                            elif is_valid_response.status_code == 500:
                                error_datasets.append(dataset)
                            else:
                                unauthorized_datasets.append(dataset)
                        
                        def fill_empty_cells(datasets, sign):
                            trending_datasets_coverage[
                                "All trending datasets"
                            ] += datasets
                            for pretty_field in trending_datasets_coverage:
                                trending_datasets_coverage[pretty_field] += [sign] * (
                                    len(trending_datasets_coverage["All trending datasets"])
                                    - len(trending_datasets_coverage[pretty_field])
                                )    
                        fill_empty_cells(error_datasets, "❌")
                        fill_empty_cells(unauthorized_datasets, "🚫")

                        out[
                            home_dashboard_trending_datasets_coverage_table
                        ] = gr.DataFrame(
                            visible=True, value=pd.DataFrame(trending_datasets_coverage)
                        )
                        trending_datasets_coverage_stats = {
                            "Num trending datasets": [len(trending_datasets)],
                            **{
                                is_valid_field: [
                                    f"{round(100 * sum(1 for coverage in trending_datasets_coverage[is_valid_field] if coverage == '✅') / len(trending_datasets), 2)}%"
                                ]
                                for is_valid_field in trending_datasets_coverage
                                if is_valid_field != "All trending datasets"
                            },
                        }
                        out[
                            home_dashboard_trending_datasets_coverage_stats_table
                        ] = gr.DataFrame(
                            visible=True,
                            value=pd.DataFrame(trending_datasets_coverage_stats),
                        )
                    else:
                        out[
                            home_dashboard_trending_datasets_coverage_table
                        ] = gr.DataFrame(
                            visible=True,
                            value=pd.DataFrame(
                                {
                                    "Error": [
                                        f"❌ Failed to fetch trending datasets from {HF_ENDPOINT} (error {response.status_code})"
                                    ]
                                }
                            ),
                        )
                    return out

                home_dashboard_fetch_button.click(
                    fetch_home_dashboard,
                    inputs=[token_box],
                    outputs=[
                        home_dashboard_trending_datasets_infos_by_builder_name_table,
                        home_dashboard_trending_datasets_coverage_stats_table,
                        home_dashboard_trending_datasets_coverage_table,
                    ],
                )
            with gr.Tab("View pending jobs"):
                fetch_pending_jobs_button = gr.Button("Fetch pending jobs")
                gr.Markdown("### Pending jobs summary")
                pending_jobs_summary_table = gr.DataFrame(
                    pd.DataFrame({"Jobs": [], "Waiting": [], "Started": []})
                )
                gr.Markdown("### Most recent")
                recent_pending_jobs_table = gr.DataFrame()
                gr.Markdown("### Query the pending jobs table")
                pending_jobs_query = gr.Textbox(
                    label="Query pending_jobs_df",
                    placeholder="SELECT * FROM pending_jobs_df WHERE dataset LIKE 'allenai/c4",
                    value="SELECT * FROM pending_jobs_df WHERE dataset LIKE 'allenai/c4'",
                    lines=3,
                )
                query_pending_jobs_button = gr.Button("Run")
                pending_jobs_query_result_df = gr.DataFrame()

                def view_jobs(token):
                    global pending_jobs_df
                    headers = {"Authorization": f"Bearer {token}"}
                    response = requests.get(
                        f"{DV_ENDPOINT}/admin/pending-jobs",
                        headers=headers,
                        timeout=60,
                    )
                    if response.status_code == 200:
                        pending_jobs = response.json()
                        df = pd.DataFrame(
                            [
                                job
                                for job_type in pending_jobs
                                for job_state in pending_jobs[job_type]
                                for job in pending_jobs[job_type][job_state]
                            ]
                        )
                        if "started_at" in df.columns:
                            df["started_at"] = pd.to_datetime(
                                df["started_at"], errors="coerce"
                            )
                        if "last_heartbeat" in df.columns:
                            df["last_heartbeat"] = pd.to_datetime(
                                df["last_heartbeat"],
                                errors="coerce",
                            )
                        if "created_at" in df.columns:
                            df["created_at"] = pd.to_datetime(
                                df["created_at"], errors="coerce"
                            )
                            most_recent = df.nlargest(5, "created_at")
                        else:
                            most_recent = pd.DataFrame()
                        pending_jobs_df = df
                        return {
                            pending_jobs_summary_table: gr.DataFrame(
                                visible=True,
                                value=pd.DataFrame(
                                    {
                                        "Jobs": list(pending_jobs),
                                        "Waiting": [
                                            len(pending_jobs[job_type]["waiting"])
                                            for job_type in pending_jobs
                                        ],
                                        "Started": [
                                            len(pending_jobs[job_type]["started"])
                                            for job_type in pending_jobs
                                        ],
                                    }
                                ),
                            ),
                            recent_pending_jobs_table: gr.DataFrame(value=most_recent),
                        }
                    else:
                        return {
                            pending_jobs_summary_table: gr.DataFrame(
                                visible=True,
                                value=pd.DataFrame(
                                    {
                                        "Error": [
                                            f"❌ Failed to view pending jobs to {DV_ENDPOINT} (error {response.status_code})"
                                        ]
                                    }
                                ),
                            ),
                            recent_pending_jobs_table: gr.DataFrame(value=None),
                        }

                def query_jobs(pending_jobs_query):
                    global pending_jobs_df
                    if pending_jobs_df is None:
                        return {
                            pending_jobs_query_result_df: gr.DataFrame(
                                value=pd.DataFrame(
                                    {
                                        "Error": [
                                            "❌ Please, fetch the pending jobs first"
                                        ]
                                    }
                                )
                            )
                        }
                    try:
                        result = duckdb.query(pending_jobs_query).to_df()
                    except (
                        duckdb.ParserException,
                        duckdb.CatalogException,
                        duckdb.BinderException,
                    ) as error:
                        return {
                            pending_jobs_query_result_df: gr.DataFrame(
                                value=pd.DataFrame({"Error": [f"❌ {str(error)}"]})
                            )
                        }
                    return {pending_jobs_query_result_df: gr.DataFrame(value=result)}

                fetch_pending_jobs_button.click(
                    view_jobs,
                    inputs=token_box,
                    outputs=[recent_pending_jobs_table, pending_jobs_summary_table],
                )
                query_pending_jobs_button.click(
                    query_jobs,
                    inputs=pending_jobs_query,
                    outputs=[pending_jobs_query_result_df],
                )

            with gr.Tab("Refresh dataset step"):
                job_types = [
                    processing_step.job_type
                    for processing_step in processing_graph.get_alphabetically_ordered_processing_steps()
                ]

                def on_change_refresh_job_type(job_type):
                    return processing_graph.get_processing_step(job_type).difficulty

                refresh_type = gr.Dropdown(
                    job_types,
                    multiselect=False,
                    type="value",
                    label="job type",
                    value=job_types[0],
                )
                refresh_dataset_name = gr.Textbox(
                    label="dataset", placeholder="allenai/c4"
                )
                refresh_config_name = gr.Textbox(
                    label="config (optional)", placeholder="en"
                )
                refresh_split_name = gr.Textbox(
                    label="split (optional)", placeholder="train, test"
                )
                gr.Markdown(
                    "*you can select multiple values by separating them with commas, e.g. split='train, test'*"
                )

                refresh_difficulty = gr.Slider(
                    0,
                    100,
                    processing_graph.get_processing_step(job_types[0]).difficulty,
                    step=10,
                    interactive=True,
                    label="difficulty",
                )
                refresh_type.change(
                    on_change_refresh_job_type, refresh_type, refresh_difficulty
                )

                refresh_priority = gr.Dropdown(
                    ["low", "normal", "high"],
                    multiselect=False,
                    label="priority",
                    value="high",
                )
                refresh_dataset_button = gr.Button("Force refresh dataset")
                refresh_dataset_output = gr.Markdown("")

                def refresh_dataset(
                    token,
                    refresh_type,
                    refresh_dataset_names,
                    refresh_config_names,
                    refresh_split_names,
                    refresh_priority,
                    refresh_difficulty,
                ):
                    headers = {"Authorization": f"Bearer {token}"}
                    all_results = ""
                    for (
                        refresh_dataset_name,
                        refresh_config_name,
                        refresh_split_name,
                    ) in product(
                        refresh_dataset_names.split(","),
                        refresh_config_names.split(","),
                        refresh_split_names.split(","),
                    ):
                        refresh_dataset_name = refresh_dataset_name.strip()
                        params = {
                            "dataset": refresh_dataset_name,
                            "priority": refresh_priority,
                        }
                        if refresh_config_name:
                            refresh_config_name = refresh_config_name.strip()
                            params["config"] = refresh_config_name
                        if refresh_split_name:
                            refresh_split_name = refresh_split_name.strip()
                            params["split"] = refresh_split_name
                        if refresh_difficulty:
                            params["difficulty"] = refresh_difficulty
                        params = urllib.parse.urlencode(params)
                        response = requests.post(
                            f"{DV_ENDPOINT}/admin/force-refresh/{refresh_type}?{params}",
                            headers=headers,
                            timeout=60,
                        )
                        if response.status_code == 200:
                            result = f"[{refresh_dataset_name}] ✅ Added processing step to the queue: '{refresh_type}'"
                            if refresh_config_name:
                                result += f", for config '{refresh_config_name}'"
                            if refresh_split_name:
                                result += f", for split '{refresh_split_name}'"
                        else:
                            result = f"[{refresh_dataset_name}] ❌ Failed to add processing step to the queue. Error {response.status_code}"
                            try:
                                if response.json().get("error"):
                                    result += f": {response.json()['error']}"
                            except requests.JSONDecodeError:
                                result += f": {response.content}"
                        all_results += result.strip("\n") + "\n"
                    return "```\n" + all_results + "\n```"

                refresh_dataset_button.click(
                    refresh_dataset,
                    inputs=[
                        token_box,
                        refresh_type,
                        refresh_dataset_name,
                        refresh_config_name,
                        refresh_split_name,
                        refresh_priority,
                        refresh_difficulty,
                    ],
                    outputs=refresh_dataset_output,
                )
            with gr.Tab("Recreate dataset"):
                delete_and_recreate_dataset_name = gr.Textbox(
                    label="dataset", placeholder="imdb"
                )
                delete_and_recreate_priority = gr.Dropdown(
                    ["low", "normal", "high"],
                    multiselect=False,
                    label="priority",
                    value="high",
                )
                gr.Markdown(
                    "Beware: this will delete all the jobs, cache entries and assets for the dataset (for all the revisions). The dataset viewer will be unavailable until the cache is rebuilt."
                )
                delete_and_recreate_dataset_button = gr.Button("Delete and recreate")
                delete_and_recreate_dataset_output = gr.Markdown("")

                def delete_and_recreate_dataset(
                    token,
                    delete_and_recreate_dataset_name,
                    delete_and_recreate_priority,
                ):
                    headers = {"Authorization": f"Bearer {token}"}
                    delete_and_recreate_dataset_name = (
                        delete_and_recreate_dataset_name.strip()
                    )
                    params = {
                        "dataset": delete_and_recreate_dataset_name,
                        "priority": delete_and_recreate_priority,
                    }
                    params = urllib.parse.urlencode(params)
                    response = requests.post(
                        f"{DV_ENDPOINT}/admin/recreate-dataset?{params}",
                        headers=headers,
                        timeout=60,
                    )
                    if response.status_code == 200:
                        result = f"[{delete_and_recreate_dataset_name}] ✅ All the assets have been deleted. A new job has been created to generate the cache again."
                    else:
                        result = f"[{refresh_dataset_name}] ❌ Failed to delete and recreate the dataset. Error {response.status_code}"
                        try:
                            if response.json().get("error"):
                                result += f": {response.json()['error']}"
                        except requests.JSONDecodeError:
                            result += f": {response.content}"
                    return result.strip("\n") + "\n"

                delete_and_recreate_dataset_button.click(
                    delete_and_recreate_dataset,
                    inputs=[
                        token_box,
                        delete_and_recreate_dataset_name,
                        delete_and_recreate_priority,
                    ],
                    outputs=delete_and_recreate_dataset_output,
                )
            with gr.Tab("Dataset status"):
                dataset_name = gr.Textbox(label="dataset", placeholder="allenai/c4")
                dataset_status_button = gr.Button("Get dataset status")
                gr.Markdown("### Pending jobs")
                jobs_table = gr.DataFrame()
                gr.Markdown("### Cached responses")
                cached_responses_table = gr.DataFrame()

                def get_dataset_status(token, dataset):
                    headers = {"Authorization": f"Bearer {token}"}
                    response = requests.get(
                        f"{DV_ENDPOINT}/admin/dataset-status?dataset={dataset}",
                        headers=headers,
                        timeout=60,
                    )
                    if response.status_code == 200:
                        dataset_status = response.json()
                        cached_responses_df = pd.DataFrame(
                            [
                                {
                                    "kind": cached_response["kind"],
                                    "dataset": cached_response["dataset"],
                                    "config": cached_response["config"],
                                    "split": cached_response["split"],
                                    "http_status": cached_response["http_status"],
                                    "error_code": cached_response["error_code"],
                                    "job_runner_version": cached_response[
                                        "job_runner_version"
                                    ],
                                    "dataset_git_revision": cached_response[
                                        "dataset_git_revision"
                                    ],
                                    "progress": cached_response["progress"],
                                    "updated_at": cached_response["updated_at"],
                                    "failed_runs": cached_response["failed_runs"],
                                    "details": json.dumps(cached_response["details"]),
                                }
                                for content in dataset_status.values()
                                for cached_response in content["cached_responses"]
                            ]
                        )
                        jobs_df = pd.DataFrame(
                            [
                                {
                                    "type": job["type"],
                                    "dataset": job["dataset"],
                                    "revision": job["revision"],
                                    "config": job["config"],
                                    "split": job["split"],
                                    "namespace": job["namespace"],
                                    "priority": job["priority"],
                                    "status": job["status"],
                                    "difficulty": job["difficulty"],
                                    "created_at": job["created_at"],
                                    "started_at": job["started_at"],
                                    "last_heartbeat": job["last_heartbeat"],
                                }
                                for content in dataset_status.values()
                                for job in content["jobs"]
                            ]
                        )
                        return {
                            cached_responses_table: gr.DataFrame(
                                value=cached_responses_df
                            ),
                            jobs_table: gr.DataFrame(value=jobs_df),
                        }
                    else:
                        return {
                            cached_responses_table: gr.DataFrame(
                                value=pd.DataFrame(
                                    [
                                        {
                                            "error": f"❌ Failed to get status for {dataset} (error {response.status_code})"
                                        }
                                    ]
                                )
                            ),
                            jobs_table: gr.DataFrame(
                                value=pd.DataFrame([{"content": str(response.content)}])
                            ),
                        }

                dataset_status_button.click(
                    get_dataset_status,
                    inputs=[token_box, dataset_name],
                    outputs=[cached_responses_table, jobs_table],
                )

            with gr.Tab("Processing graph"):
                gr.Markdown(
                    "## 💫 Please, don't forget to rebuild (factory reboot) this space immediately after each deploy 💫"
                )
                gr.Markdown(
                    "### so that we get the 🚀 production 🚀 version of the graph here "
                )
                with gr.Row():
                    width = gr.Slider(1, 30, 19, step=1, label="Width")
                    height = gr.Slider(1, 30, 15, step=1, label="Height")
                output = gr.Plot()
                draw_button = gr.Button("Plot processing graph")
                draw_button.click(draw_graph, inputs=[width, height], outputs=output)

    def auth(token):
        if not token:
            return {auth_error: gr.Markdown(value="", visible=False)}
        try:
            user = hfh.whoami(token=token)
        except requests.HTTPError as err:
            return {auth_error: gr.Markdown(value=f"❌ Error ({err})", visible=True)}
        orgs = [org["name"] for org in user["orgs"]]
        if ADMIN_HF_ORGANIZATION in orgs:
            return {
                auth_page: gr.Row(visible=False),
                welcome_title: gr.Markdown(value=f"### Welcome {user['name']}"),
                main_page: gr.Row(visible=True),
            }
        else:
            return {
                auth_error: gr.Markdown(
                    value=f"❌ Unauthorized (user '{user['name']} is not a member of '{ADMIN_HF_ORGANIZATION}')"
                )
            }

    token_box.change(
        auth,
        inputs=token_box,
        outputs=[auth_error, welcome_title, auth_page, main_page],
    )

if __name__ == "__main__":
    demo.launch()
