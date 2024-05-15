# Authentication for private and gated datasets

To access private or gated datasets, you need to configure your Hugging Face Token in the DuckDB Secrets Manager.

Visit [Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens) to obtain your access token.

DuckDB supports two providers for managing secrets:

- `CONFIG`: Requires the user to pass all configuration information into the CREATE SECRET statement.
- `CREDENTIAL_CHAIN`: Automatically tries to fetch credentials. For Hugging Face token it will try to get it from  `~/.cache/huggingface/token`

For more information about DuckDB Secrets visit https://duckdb.org/docs/configuration/secrets_manager.html

## Creating a secret with `CONFIG` provider

To create a secret using the CONFIG provider, use the following command:

```bash
CREATE SECRET hf_token (TYPE HUGGINGFACE, token 'your_hf_token');
```

Replace `your_hf_token` with your actual Hugging Face token.