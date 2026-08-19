# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

"""Egress guard against SSRF.

The worker fetches URLs that come from dataset content: the `data_files` patterns declared in a
dataset card, and the `path` of the media values stored in a dataset's Parquet files. Without a
guard, a dataset can make the worker request internal endpoints: the cloud metadata service,
in-cluster services, or anything else routable from the pod.

The checks apply to the IP addresses the connections are opened to, not to the hostname, so they
also cover a hostname that resolves to an internal address and a redirect to an internal host. Both
guarded transports connect to an address that has been checked rather than resolving the host again,
so a host that answers differently at connect time does not get through either. The transports that
are guarded are the ones used to fetch dataset content:

- `aiohttp`, the transport of `fsspec`'s `HTTPFileSystem` - which `datasets` uses to stream the data
  files - and of `s3fs`,
- `httpx`, the transport of `huggingface_hub`, which `datasets` also uses to send a HEAD request to
  the URL of a data file before opening it.
"""

import socket
from collections.abc import Iterable
from ipaddress import IPv4Address, IPv6Address, ip_address
from typing import Any, Optional, Union
from urllib.parse import urlsplit

import aiohttp
import aiohttp.client
import httpx
from aiohttp.abc import ResolveResult
from aiohttp.connector import AddrInfoType
from httpcore import SOCKET_OPTION, ConnectError, ConnectTimeout, NetworkStream, SyncBackend
from huggingface_hub import set_client_factory
from huggingface_hub.utils._http import default_client_factory

# The EC2 instance metadata service and the ECS task credentials endpoint. Both are link-local, so
# they are already refused by `is_blocked_address`; they are listed to make the intent explicit.
METADATA_ADDRESSES = frozenset({IPv4Address("169.254.169.254"), IPv4Address("169.254.170.2")})


class BlockedAddressError(OSError):
    """An outbound connection was refused because its target is not a public address.

    It derives from `OSError` so that `aiohttp` and `httpx` report it the same way as any other
    connection failure, instead of leaking through their retry and error handling.
    """


def is_blocked_address(address: str) -> bool:
    """Whether an IP address is outside the public internet, and must not be connected to.

    `is_global` is false for loopback, private (RFC 1918), shared (100.64.0.0/10), link-local
    (169.254.0.0/16 and fe80::/10), unique-local (fc00::/7) and reserved addresses, but it is true
    for multicast ones, hence the extra check.

    Args:
        address (`str`): An IP address, as returned by a resolver.

    Returns:
        `bool`: Whether connecting to the address must be refused.
    """
    ip: Union[IPv4Address, IPv6Address] = ip_address(address)
    if isinstance(ip, IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip in METADATA_ADDRESSES or ip.is_multicast or not ip.is_global


def raise_if_blocked_url(url: str) -> None:
    """Refuse a URL whose host resolves to a non-public address.

    This resolves the host itself, and is therefore only a pre-flight check: a host that changes its
    answer between this call and the connection is stopped by the transports - `GuardedTCPConnector`
    and `GuardedNetworkBackend` - which connect to an address they have checked.

    Raises:
        [~`libcommon.ssrf.BlockedAddressError`]: If the host resolves to a non-public address.
    """
    host = urlsplit(url).hostname
    if host is None:
        return
    for _, _, _, _, sockaddr in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
        address = str(sockaddr[0])
        if is_blocked_address(address):
            raise BlockedAddressError(f"Refusing to connect to {host} ({address}): not a public address")


def guarded_socket_factory(addr_info: AddrInfoType) -> socket.socket:
    """Create the socket `aiohttp` is about to connect, once the address it will use is checked.

    This is the last step before `connect()`, so it is what makes the check apply to the address
    actually used, whatever the URL contained and whatever DNS answered in between.

    Raises:
        [~`libcommon.ssrf.BlockedAddressError`]: If the address is not a public one.
    """
    family, type_, proto, _, sockaddr = addr_info
    address = str(sockaddr[0])
    if is_blocked_address(address):
        raise BlockedAddressError(f"Refusing to connect to {address}: not a public address")
    return socket.socket(family=family, type=type_, proto=proto)


class GuardedTCPConnector(aiohttp.TCPConnector):
    """`TCPConnector` that refuses to connect to non-public addresses."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("socket_factory", guarded_socket_factory)
        super().__init__(*args, **kwargs)

    async def _resolve_host(self, host: str, port: int, traces: Optional[Any] = None) -> list[ResolveResult]:
        # `TCPConnector._resolve_host` returns the addresses that will be connected to, and returns
        # the host as-is - without calling the resolver - when it already is an IP address. Dropping
        # the blocked ones here pins the connection to the addresses that have been checked.
        results = await super()._resolve_host(host, port, traces=traces)
        allowed = [result for result in results if not is_blocked_address(result["host"])]
        if not allowed:
            raise BlockedAddressError(f"Refusing to connect to {host}: it resolves to non-public addresses only")
        return allowed


def _check_httpx_request(request: httpx.Request) -> None:
    raise_if_blocked_url(str(request.url))


def guard_httpx_client(client: httpx.Client) -> httpx.Client:
    """Make an `httpx.Client` refuse the requests whose host resolves to a non-public address.

    This is the pre-flight check: it fails before a socket is opened, and it runs once per request so
    it also runs on every hop of a redirect chain. `GuardedHTTPTransport` is what checks the address
    the connection actually uses.
    """
    client.event_hooks["request"].append(_check_httpx_request)
    return client


class GuardedNetworkBackend(SyncBackend):
    """`httpcore` backend that connects to a checked address instead of resolving the host again.

    `SyncBackend.connect_tcp` hands the hostname to `socket.create_connection`, which resolves it
    itself, so checking the host before the request leaves a window in which DNS can answer an
    internal address at connect time. Resolving here and connecting to one of the checked addresses
    closes it.

    `httpcore` takes the name used for SNI and for the certificate check from the request URL, not
    from what is connected to, so handing it an address leaves TLS unchanged.
    """

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options: Optional[Iterable[SOCKET_OPTION]] = None,
    ) -> NetworkStream:
        resolved = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        allowed = [str(sockaddr[0]) for *_, sockaddr in resolved if not is_blocked_address(str(sockaddr[0]))]
        if not allowed:
            raise BlockedAddressError(f"Refusing to connect to {host}: it resolves to non-public addresses only")
        for address in allowed[:-1]:
            try:
                return self._connect_tcp_to_address(address, port, timeout, local_address, socket_options)
            except (ConnectError, ConnectTimeout):
                # keep the fallback on the next address that `socket.create_connection` would do
                pass
        return self._connect_tcp_to_address(allowed[-1], port, timeout, local_address, socket_options)

    def _connect_tcp_to_address(
        self,
        address: str,
        port: int,
        timeout: Optional[float],
        local_address: Optional[str],
        socket_options: Optional[Iterable[SOCKET_OPTION]],
    ) -> NetworkStream:
        return super().connect_tcp(
            address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
        )


class GuardedHTTPTransport(httpx.HTTPTransport):
    """`HTTPTransport` that connects only to the addresses that have been checked."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # the pool is configured by httpx, TLS context included, so only the backend that opens the
        # connections is replaced
        self._pool._network_backend = GuardedNetworkBackend()


def _guarded_client_factory() -> httpx.Client:
    client = guard_httpx_client(default_client_factory())
    client._transport = GuardedHTTPTransport()
    return client


def install_ssrf_guard() -> None:
    """Make the HTTP clients that fetch dataset content refuse non-public addresses.

    Calling it is idempotent. It must be done before the first request, so that no client is created
    from the unguarded defaults.
    """
    aiohttp.TCPConnector = GuardedTCPConnector  # type: ignore[misc]
    aiohttp.client.TCPConnector = GuardedTCPConnector  # type: ignore[misc]
    # ^ `ClientSession` builds its connector from `aiohttp.client.TCPConnector`, while third-party
    # code - `aiobotocore`, hence `s3fs` - builds it from `aiohttp.TCPConnector`
    set_client_factory(_guarded_client_factory)
