import os

import pandas as pd
import requests
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
import huggingface_hub as hfh
import duckdb

matplotlib.use('SVG')

DEV = os.environ.get("DEV", False)
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
PROD_DSS_ENDPOINT = os.environ.get("PROD_DSS_ENDPOINT", "https://datasets-server.huggingface.co")
DEV_DSS_ENDPOINT = os.environ.get("DEV_DSS_ENDPOINT", "http://localhost:8100")
ADMIN_HF_ORGANIZATION = os.environ.get("ADMIN_HF_ORGANIZATION", "huggingface")

DSS_ENDPOINT = DEV_DSS_ENDPOINT if DEV else PROD_DSS_ENDPOINT

pending_jobs_df = None

def healthcheck():
    response = requests.head(f"{DSS_ENDPOINT}/admin/pending-jobs", timeout=10)
    if response.status_code == 401:
        return f"*Connected to {DSS_ENDPOINT}*"

    else:
        return f"❌ Failed to connect to {DSS_ENDPOINT} (error {response.status_code})"



with gr.Blocks() as demo:
    gr.Markdown(" ## Datasets-server admin page")
    gr.Markdown(healthcheck)

    with gr.Row() as auth_page:
        with gr.Column():
            auth_title = gr.Markdown("Enter your token ([settings](https://huggingface.co/settings/tokens)):")
            token_box = gr.Textbox(label="token", placeholder="hf_xxx", type="password")
            auth_error = gr.Markdown("", visible=False)
    
    with gr.Row(visible=False) as main_page:
        with gr.Column():
            welcome_title = gr.Markdown("")
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
            return f"❌ Unauthorized (user '{user['name']} is not a member of '{ADMIN_HF_ORGANIZATION}')"

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
            pending_jobs_df["created_at"] = pd.to_datetime(pending_jobs_df["created_at"], errors="coerce")
            return {
                pending_jobs_summary_table: gr.update(visible=True, value=pd.DataFrame({
                    "Jobs": list(pending_jobs),
                    "Waiting": [len(pending_jobs[job_type]["waiting"]) for job_type in pending_jobs],
                    "Started": [len(pending_jobs[job_type]["started"]) for job_type in pending_jobs],
                })),
                recent_pending_jobs_table: gr.update(value=pending_jobs_df.nlargest(5, "created_at"))
            }
        else:
            return {
                pending_jobs_summary_table: gr.update(visible=True, value=pd.DataFrame({"Error": [f"❌ Failed to view pending jobs to {DSS_ENDPOINT} (error {response.status_code})"]})),
                recent_pending_jobs_table: gr.update(value=None)
            }
    
    def query_jobs(pending_jobs_query):
        global pending_jobs_df
        try:
            result = duckdb.query(pending_jobs_query).to_df()
        except (duckdb.ParserException, duckdb.CatalogException, duckdb.BinderException) as error:
            return {pending_jobs_query_result_df: gr.update(value=pd.DataFrame({"Error": [f"❌ {str(error)}"]}))}
        return {pending_jobs_query_result_df: gr.update(value=result)}

    token_box.change(auth, inputs=token_box, outputs=[auth_error, welcome_title, auth_page, main_page])
    fetch_pending_jobs_button.click(view_jobs, inputs=token_box, outputs=[recent_pending_jobs_table, pending_jobs_summary_table])
    query_pending_jobs_button.click(query_jobs, inputs=pending_jobs_query, outputs=[pending_jobs_query_result_df])
    


if __name__ == "__main__":
    demo.launch()
