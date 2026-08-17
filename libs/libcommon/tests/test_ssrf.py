# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import asyncio
import socket
from collections.abc import Iterator
from contextlib import nullcontext as does_not_raise
from typing import Any

import aiohttp
import aiohttp.client
import httpx
import pytest
from huggingface_hub import set_client_factory
from huggingface_hub.utils._http import default_client_factory
from pytest import MonkeyPatch

from libcommon.ssrf import (
    BlockedAddressError,
    GuardedHTTPTransport,
    GuardedNetworkBackend,
    GuardedTCPConnector,
    guard_httpx_client,
    guarded_socket_factory,
    install_ssrf_guard,
    is_blocked_address,
    raise_if_blocked_url,
)

# the hosts the fake resolver knows about, so that the tests don't depend on DNS
RESOLVED_HOSTS = {
    "public.example": "93.184.216.34",
    "metadata.example": "169.254.169.254",  # a host that resolves to an internal address
    "intranet.example": "10.0.0.1",
    "loopback.example": "127.0.0.1",
}


@pytest.fixture
def fake_dns(monkeypatch: MonkeyPatch) -> Iterator[None]:
    real_getaddrinfo = socket.getaddrinfo

    def getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> Any:
        if host in RESOLVED_HOSTS:
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (RESOLVED_HOSTS[host], port or 0))]
        return real_getaddrinfo(host, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    yield


@pytest.fixture
def installed_guard(monkeypatch: MonkeyPatch) -> Iterator[None]:
    # setting an attribute to its current value records it, so that it is restored on teardown
    monkeypatch.setattr(aiohttp, "TCPConnector", aiohttp.TCPConnector)
    monkeypatch.setattr(aiohttp.client, "TCPConnector", aiohttp.client.TCPConnector)
    install_ssrf_guard()
    yield
    set_client_factory(default_client_factory)


@pytest.mark.parametrize(
    "address,blocked",
    [
        # the cloud metadata endpoints
        ("169.254.169.254", True),
        ("169.254.170.2", True),
        # loopback
        ("127.0.0.1", True),
        ("::1", True),
        ("::ffff:127.0.0.1", True),
        # private (RFC 1918)
        ("10.0.0.1", True),
        ("172.16.0.1", True),
        ("192.168.1.1", True),
        ("::ffff:10.0.0.1", True),
        # shared address space (CGNAT)
        ("100.64.0.1", True),
        # link-local
        ("169.254.1.1", True),
        ("fe80::1", True),
        # unique-local
        ("fc00::1", True),
        ("fd00::1", True),
        # multicast, unspecified and reserved
        ("224.0.0.1", True),
        ("0.0.0.0", True),
        ("240.0.0.1", True),
        # public
        ("8.8.8.8", False),
        ("93.184.216.34", False),
        ("2606:4700:4700::1111", False),
        ("::ffff:8.8.8.8", False),
    ],
)
def test_is_blocked_address(address: str, blocked: bool) -> None:
    assert is_blocked_address(address) is blocked


@pytest.mark.parametrize(
    "url,expectation",
    [
        ("http://169.254.169.254/latest/meta-data/", pytest.raises(BlockedAddressError)),
        ("http://metadata.example/latest/meta-data/", pytest.raises(BlockedAddressError)),
        ("https://intranet.example/", pytest.raises(BlockedAddressError)),
        ("https://loopback.example/", pytest.raises(BlockedAddressError)),
        ("https://public.example/data.csv", does_not_raise()),
        ("https://93.184.216.34/data.csv", does_not_raise()),
    ],
)
def test_raise_if_blocked_url(url: str, expectation: Any, fake_dns: None) -> None:
    with expectation:
        raise_if_blocked_url(url)


@pytest.mark.parametrize(
    "address,expectation",
    [
        ("169.254.169.254", pytest.raises(BlockedAddressError)),
        ("10.0.0.1", pytest.raises(BlockedAddressError)),
        ("93.184.216.34", does_not_raise()),
    ],
)
def test_guarded_socket_factory(address: str, expectation: Any) -> None:
    with expectation:
        # the socket is created, never connected
        guarded_socket_factory((socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))).close()


def resolve_host(host: str, resolved: list[str]) -> list[str]:
    """Resolve a host with a `GuardedTCPConnector`, `resolved` being the answer of the resolver."""

    async def resolve() -> list[str]:
        connector = GuardedTCPConnector(use_dns_cache=False)

        async def resolver_resolve(host: str, port: int = 0, family: Any = socket.AF_INET) -> Any:
            return [
                {"hostname": host, "host": address, "port": port, "family": family, "proto": 6, "flags": 0}
                for address in resolved
            ]

        connector._resolver.resolve = resolver_resolve  # type: ignore[method-assign]
        try:
            return [result["host"] for result in await connector._resolve_host(host, 443)]
        finally:
            await connector.close()

    return asyncio.run(resolve())


def test_guarded_connector_drops_the_blocked_addresses() -> None:
    assert resolve_host("mixed.example", ["169.254.169.254", "93.184.216.34"]) == ["93.184.216.34"]


def test_guarded_connector_refuses_a_host_resolving_only_to_blocked_addresses() -> None:
    with pytest.raises(BlockedAddressError):
        resolve_host("metadata.example", ["169.254.169.254", "10.0.0.1"])


def test_guarded_connector_refuses_an_address_used_as_a_host() -> None:
    # `TCPConnector._resolve_host` doesn't call the resolver when the host already is an IP address
    with pytest.raises(BlockedAddressError):
        resolve_host("169.254.169.254", [])


def test_guarded_httpx_client_refuses_a_redirect_to_an_internal_host(fake_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "public.example":
            return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})
        return httpx.Response(200, text="the response of the internal host")

    client = guard_httpx_client(httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))
    with client, pytest.raises(BlockedAddressError):
        client.get("https://public.example/data.csv")


def test_guarded_httpx_client_allows_a_public_host(fake_dns: None) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        return httpx.Response(200, text="the response of the public host")

    client = guard_httpx_client(httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True))
    with client:
        assert client.get("https://public.example/data.csv").text == "the response of the public host"


@pytest.fixture
def refuse_to_connect(monkeypatch: MonkeyPatch) -> Iterator[list[Any]]:
    """Record the addresses `httpcore` opens a connection to, and never open one."""
    connected: list[Any] = []

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        connected.append(address)
        raise AssertionError(f"connected to {address}")

    monkeypatch.setattr(socket, "create_connection", create_connection)
    yield connected


def rebinding_dns(monkeypatch: MonkeyPatch, host: str, addresses: list[str]) -> None:
    """Resolve `host` to the next address on each call, to reproduce DNS rebinding."""
    remaining = list(addresses)

    def getaddrinfo(hostname: str, port: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        address = remaining.pop(0) if hostname == host and remaining else addresses[-1]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


def test_guarded_network_backend_connects_to_the_address_it_checked(fake_dns: None, monkeypatch: MonkeyPatch) -> None:
    connected: list[Any] = []

    def create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        connected.append(address)
        return socket.socket()

    monkeypatch.setattr(socket, "create_connection", create_connection)
    GuardedNetworkBackend().connect_tcp("public.example", 443).close()
    # the checked address is connected to, not the host: it cannot resolve to something else in between
    assert connected == [("93.184.216.34", 443)]


def test_guarded_network_backend_refuses_an_internal_host(fake_dns: None, refuse_to_connect: list[Any]) -> None:
    with pytest.raises(BlockedAddressError):
        GuardedNetworkBackend().connect_tcp("metadata.example", 80)
    assert refuse_to_connect == []


def test_guarded_httpx_client_refuses_a_host_that_rebinds_after_the_check(
    monkeypatch: MonkeyPatch, refuse_to_connect: list[Any]
) -> None:
    # the pre-flight check gets a public address, the connection would get an internal one
    rebinding_dns(monkeypatch, "rebinding.example", ["93.184.216.34", "169.254.169.254"])

    client = guard_httpx_client(httpx.Client(transport=GuardedHTTPTransport()))
    with client, pytest.raises(BlockedAddressError):
        client.get("http://rebinding.example/latest/meta-data/")

    # the pre-flight check passed on the first answer, so the transport is what refused the second
    assert refuse_to_connect == []


def test_install_ssrf_guard(installed_guard: None) -> None:
    assert aiohttp.TCPConnector is GuardedTCPConnector
    assert aiohttp.client.TCPConnector is GuardedTCPConnector

    async def get(url: str) -> Any:
        async with aiohttp.ClientSession() as session:
            try:
                await session.get(url, timeout=aiohttp.ClientTimeout(total=5))
            except Exception as err:
                return err.__cause__
        return None

    # a session built from the defaults, as `fsspec` builds it, refuses an internal address
    assert isinstance(asyncio.run(get("http://169.254.169.254/latest/meta-data/")), BlockedAddressError)
