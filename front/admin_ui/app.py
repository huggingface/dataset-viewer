import os
import urllib.parse
from itertools import product

import pandas as pd
import requests
import gradio as gr
from libcommon.processing_graph import ProcessingGraph
from libcommon.config import ProcessingGraphConfig
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import huggingface_hub as hfh
import duckdb
import json

matplotlib.use('SVG')

DEV = os.environ.get("DEV", False)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
PROD_DSS_ENDPOINT = os.environ.get("PROD_DSS_ENDPOINT", "https://datasets-server.huggingface.co")
DEV_DSS_ENDPOINT = os.environ.get("DEV_DSS_ENDPOINT", "http://localhost:8100")
ADMIN_HF_ORGANIZATION = os.environ.get("ADMIN_HF_ORGANIZATION", "huggingface")
HF_TOKEN = os.environ.get("HF_TOKEN")

DSS_ENDPOINT = DEV_DSS_ENDPOINT if DEV else PROD_DSS_ENDPOINT

pending_jobs_df = None

def healthcheck():
    try:
        response = requests.head(f"{DSS_ENDPOINT}/admin/pending-jobs", timeout=10)
    except requests.ConnectionError as error:
        return f"❌ Failed to connect to {DSS_ENDPOINT} (error {error})"
    if response.status_code == 401:
        return f"*Connected to {DSS_ENDPOINT}*"

    else:
        return f"❌ Failed to connect to {DSS_ENDPOINT} (error {response.status_code})"


def draw_graph(width, height):
    config = ProcessingGraphConfig()
    graph = ProcessingGraph(config.specification)._nx_graph

    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")
    fig = plt.figure(figsize=(width, height))
    nx.draw_networkx(graph, pos=pos, node_color="#d1b2f8", node_size=500)
    return fig


with gr.Blocks() as demo:
    gr.Markdown(" ## Datasets-server admin page")
    gr.Markdown(healthcheck)

    with gr.Row(visible=HF_TOKEN is None) as auth_page:
        with gr.Column():
            auth_title = gr.Markdown("Enter your token ([settings](https://huggingface.co/settings/tokens)):")
            token_box = gr.Textbox(HF_TOKEN or "", label="token", placeholder="hf_xxx", type="password")
            auth_error = gr.Markdown("", visible=False)
    
    with gr.Row(visible=HF_TOKEN is not None) as main_page:
        with gr.Column():
            welcome_title = gr.Markdown("### Welcome")
            with gr.Tab("View pending jobs"):
                fetch_pending_jobs_button = gr.Button("Fetch pending jobs")
                gr.Markdown("### Pending jobs summary")
                pending_jobs_summary_table = gr.DataFrame(pd.DataFrame({"Jobs": [], "Waiting": [], "Started": []}))
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
            with gr.Tab("Refresh dataset"):
                refresh_type = gr.Textbox(label="Processing step type", placeholder="split-first-rows-from-streaming")
                refresh_dataset_name = gr.Textbox(label="dataset", placeholder="c4")
                refresh_config_name = gr.Textbox(label="config (optional)", placeholder="en")
                refresh_split_name = gr.Textbox(label="split (optional)", placeholder="train, test")
                gr.Markdown("*you can select multiple values by separating them with commas, e.g. split='train, test'*")
                refresh_dataset_button = gr.Button("Force refresh dataset")
                refresh_dataset_output = gr.Markdown("")
            with gr.Tab("Dataset status"):
                dataset_name = gr.Textbox(label="dataset", placeholder="c4")
                dataset_status_button = gr.Button("Get dataset status")
                gr.Markdown("### Cached responses")
                cached_responses_table = gr.DataFrame()
                gr.Markdown("### Pending jobs")
                jobs_table = gr.DataFrame()
                backfill_message = gr.Markdown("", visible=False)
                backfill_plan_table = gr.DataFrame(visible=False)
                backfill_execute_button = gr.Button("Execute backfill plan", visible=False)
                backfill_execute_error = gr.Markdown("", visible=False)
            with gr.Tab("Processing graph"):
                gr.Markdown("## 💫 Please, don't forget to rebuild (factory reboot) this space immediately after each deploy 💫")
                gr.Markdown("### so that we get the 🚀 production 🚀 version of the graph here ")
                with gr.Row():
                    width = gr.Slider(1, 30, 19, step=1, label="Width")
                    height = gr.Slider(1, 30, 15, step=1, label="Height")
                output = gr.Plot()
                draw_button = gr.Button("Plot processing graph")
                draw_button.click(draw_graph, inputs=[width, height], outputs=output)

    def auth(token):
        if not token:
            return {auth_error: gr.update(value="", visible=False)}
        try:
            user = hfh.whoami(token=token)
        except requests.HTTPError as err:
            return {auth_error: gr.update(value=f"❌ Error ({err})", visible=True)}
        orgs = [org["name"] for org in user["orgs"]]
        if ADMIN_HF_ORGANIZATION in orgs:
            return {
                auth_page: gr.update(visible=False),
                welcome_title: gr.update(value=f"### Welcome {user['name']}"),
                main_page: gr.update(visible=True)
            }
        else:
            return {
                auth_error: gr.update(value=f"❌ Unauthorized (user '{user['name']} is not a member of '{ADMIN_HF_ORGANIZATION}')")
            }

    def view_jobs(token):
        global pending_jobs_df
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{DSS_ENDPOINT}/admin/pending-jobs", headers=headers, timeout=60)
        if response.status_code == 200:
            pending_jobs = response.json()
            pending_jobs_df = pd.DataFrame([
                job
                for job_type in pending_jobs
                for job_state in pending_jobs[job_type]
                for job in pending_jobs[job_type][job_state]
            ])
            if "started_at" in pending_jobs_df.columns:
                pending_jobs_df["started_at"] = pd.to_datetime(pending_jobs_df["started_at"], errors="coerce")
            if "finished_at" in pending_jobs_df.columns:
                pending_jobs_df["finished_at"] = pd.to_datetime(pending_jobs_df["finished_at"], errors="coerce")
            if "last_heartbeat" in pending_jobs_df.columns:
                pending_jobs_df["last_heartbeat"] = pd.to_datetime(pending_jobs_df["last_heartbeat"], errors="coerce")
            if "created_at" in pending_jobs_df.columns:
                pending_jobs_df["created_at"] = pd.to_datetime(pending_jobs_df["created_at"], errors="coerce")
                most_recent = pending_jobs_df.nlargest(5, "created_at")
            else:
                most_recent = pd.DataFrame()
            return {
                pending_jobs_summary_table: gr.update(visible=True, value=pd.DataFrame({
                    "Jobs": list(pending_jobs),
                    "Waiting": [len(pending_jobs[job_type]["waiting"]) for job_type in pending_jobs],
                    "Started": [len(pending_jobs[job_type]["started"]) for job_type in pending_jobs],
                })),
                recent_pending_jobs_table: gr.update(value=most_recent)
            }
        else:
            return {
                pending_jobs_summary_table: gr.update(visible=True, value=pd.DataFrame({"Error": [f"❌ Failed to view pending jobs to {DSS_ENDPOINT} (error {response.status_code})"]})),
                recent_pending_jobs_table: gr.update(value=None)
            }

    def get_dataset_status(token, dataset):
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{DSS_ENDPOINT}/admin/dataset-status?dataset={dataset}", headers=headers, timeout=60)
        if response.status_code == 200:
            dataset_status = response.json()
            cached_responses_df = pd.DataFrame([{
                    "type": job_type,
                    "dataset": cached_response["dataset"],
                    "config": cached_response["config"],
                    "split": cached_response["split"],
                    "http_status": cached_response["http_status"],
                    "error_code": cached_response["error_code"],
                    "job_runner_version": cached_response["job_runner_version"],
                    "dataset_git_revision": cached_response["dataset_git_revision"],
                    "progress": cached_response["progress"],
                    "updated_at": cached_response["updated_at"],
                    "details": json.dumps(cached_response["details"]),
                }
                for job_type, content in dataset_status.items()
                for cached_response in content["cached_responses"]
            ])
            jobs_df = pd.DataFrame([{
                    "type": job_type,
                    "dataset": job["dataset"],
                    "config": job["config"],
                    "split": job["split"],
                    "namespace": job["namespace"],
                    "priority": job["priority"],
                    "status": job["status"],
                    "created_at": job["created_at"],
                    "started_at": job["started_at"],
                    "finished_at": job["finished_at"],
                    "last_heartbeat": job["last_heartbeat"]
                }
                for job_type, content in dataset_status.items()
                for job in content["jobs"]
            ])
            return {
                cached_responses_table: gr.update(value=cached_responses_df),
                jobs_table: gr.update(value=jobs_df)
            }
        else:
            return {
                cached_responses_table: gr.update(value=None),
                jobs_table: gr.update(value=None)
            }
    
    def get_backfill_plan(token, dataset):
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{DSS_ENDPOINT}/admin/dataset-state?dataset={dataset}", headers=headers, timeout=60)
        if response.status_code != 200:
            return {
                backfill_message: gr.update(value=f"❌ Failed to get backfill plan for {dataset} (error {response.status_code})", visible=True),
                backfill_plan_table: gr.update(value=None,visible=False),
                backfill_execute_button: gr.update( visible=False),
                backfill_execute_error: gr.update( visible=False)
            }
        dataset_state = response.json()
        tasks_df = pd.DataFrame(dataset_state["plan"])
        has_tasks = len(tasks_df) > 0
        return {
            backfill_message: gr.update(
                value="""### Backfill plan

The cache is outdated or in an incoherent state. Here is the plan to backfill the cache."""
            ,visible=has_tasks),
            backfill_plan_table: gr.update(value=tasks_df,visible=has_tasks),
            backfill_execute_button: gr.update(visible=has_tasks),
            backfill_execute_error: gr.update(visible=False),
        }

    def get_dataset_status_and_backfill_plan(token, dataset):
        return {**get_dataset_status(token, dataset), **get_backfill_plan(token, dataset)}


    def execute_backfill_plan(token, dataset):
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(f"{DSS_ENDPOINT}/admin/dataset-backfill?dataset={dataset}", headers=headers, timeout=60)
        state = get_dataset_status_and_backfill_plan(token, dataset)
        message = (
            "✅ Backfill plan executed"
            if response.status_code == 200
            else f"❌ Failed to execute backfill plan (error {response.status_code})<pre>{response.text}</pre>"
        )
        state[backfill_execute_error] = gr.update(value=message, visible=True)
        return state

    def query_jobs(pending_jobs_query):
        global pending_jobs_df
        try:
            result = duckdb.query(pending_jobs_query).to_df()
        except (duckdb.ParserException, duckdb.CatalogException, duckdb.BinderException) as error:
            return {pending_jobs_query_result_df: gr.update(value=pd.DataFrame({"Error": [f"❌ {str(error)}"]}))}
        return {pending_jobs_query_result_df: gr.update(value=result)}

    def refresh_dataset(token, refresh_types, refresh_dataset_names, refresh_config_names, refresh_split_names):
        headers = {"Authorization": f"Bearer {token}"}
        all_results = ""
        for refresh_type, refresh_dataset_name, refresh_config_name, refresh_split_name in product(
            refresh_types.split(","), refresh_dataset_names.split(","), refresh_config_names.split(","), refresh_split_names.split(",")
        ):
            refresh_types = refresh_types.strip()
            refresh_dataset_name = refresh_dataset_name.strip()
            params = {"dataset": refresh_dataset_name}
            if refresh_config_name:
                refresh_config_name = refresh_config_name.strip()
                params["config"] = refresh_config_name
            if refresh_split_name:
                refresh_split_name = refresh_split_name.strip()
                params["split"] = refresh_split_name
            params = urllib.parse.urlencode(params)
            response = requests.post(f"{DSS_ENDPOINT}/admin/force-refresh{refresh_type}?{params}", headers=headers, timeout=60)
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

    token_box.change(auth, inputs=token_box, outputs=[auth_error, welcome_title, auth_page, main_page])

    fetch_pending_jobs_button.click(view_jobs, inputs=token_box, outputs=[recent_pending_jobs_table, pending_jobs_summary_table])
    query_pending_jobs_button.click(query_jobs, inputs=pending_jobs_query, outputs=[pending_jobs_query_result_df])
    
    refresh_dataset_button.click(refresh_dataset, inputs=[token_box, refresh_type, refresh_dataset_name, refresh_config_name, refresh_split_name], outputs=refresh_dataset_output)
    dataset_status_button.click(get_dataset_status_and_backfill_plan, inputs=[token_box, dataset_name], outputs=[cached_responses_table, jobs_table, backfill_message, backfill_plan_table, backfill_execute_button, backfill_execute_error])
    backfill_execute_button.click(execute_backfill_plan, inputs=[token_box, dataset_name], outputs=[cached_responses_table, jobs_table, backfill_message, backfill_plan_table, backfill_execute_button, backfill_execute_error])


if __name__ == "__main__":
    demo.launch()
