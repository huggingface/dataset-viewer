# Dataset viewer SSE API

> Server-sent events API for the dataset viewer. It's used to update the Hugging Face Hub's backend cache.

## Configuration

The service can be configured using environment variables. They are grouped by scope.

### Common

See [../../libs/libcommon/README.md](../../libs/libcommon/README.md) for more information about the common configuration.

## Endpoints

- /sse/healthcheck: Ensure the app is running
- /sse/metrics: Return a list of metrics in the Prometheus format
- /sse/hub-cache: Return a dataset information as a Server-Sent Event (SSE) when a dataset is updated. If `?all=true` is passed in the parameters, and if the cache already has some entries, one SSE per cached dataset is sent to the client. Then, a SSE is sent when a dataset is inserted, modified or deleted. The event data is a JSON with the following structure. The `hub_cache` field is null for deleted entries, or when the response is an error. The `num_rows` value is `0` if it could not be determined.

```typescript
{
    dataset: string;
    hub_cache: null | {
        preview: boolean;
        viewer: boolean;
        partial: boolean;
        num_rows: int;
    };
}
```
