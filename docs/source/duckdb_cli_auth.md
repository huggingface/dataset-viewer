# Authentication for private and gated datasets

To access private or gated datasets, you need to configure your Hugging Face Token in the DuckDB Secrets Manager.

Visit [Hugging Face Settings - Tokens](https://huggingface.co/settings/tokens) to obtain your access token.

DuckDB supports two providers for managing secrets:

- `CONFIG`: Requires the user to pass all configuration information into the CREATE SECRET statement.
- `CREDENTIAL_CHAIN`: Automatically tries to fetch credentials. For the Hugging Face token, it will try to get it from  `~/.cache/huggingface/token`.

For more information about DuckDB Secrets visit the [Secrets Manager](https://duckdb.org/docs/configuration/secrets_manager.html) guide.

## Creating a secret with `CONFIG` provider

To create a secret using the CONFIG provider, use the following command:

```bash
CREATE SECRET hf_token (TYPE HUGGINGFACE, TOKEN 'your_hf_token');
```

Replace `your_hf_token` with your actual Hugging Face token.

## Creating a secret with `CREDENTIAL_CHAIN` provider

To create a secret using the CREDENTIAL_CHAIN provider, use the following command:

```bash
CREATE SECRET hf_token (TYPE HUGGINGFACE, PROVIDER credential_chain);
```

This command automatically retrieves the stored token from `~/.cache/huggingface/token`.

If you haven't configured your token, execute the following command in the terminal:

```bash
huggingface-cli login
```

Alternatively, you can set your Hugging Face token as an environment variable:

```bash
export HF_TOKEN="HF_XXXXXXXXXXXXX"
```

For more information on authentication, see the [Hugging Face authentication](https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication) documentation.
