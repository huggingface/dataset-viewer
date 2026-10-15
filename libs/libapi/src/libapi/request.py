# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import ipaddress
import logging
from functools import lru_cache
from typing import Optional, Union

from libcommon.constants import MAX_NUM_ROWS_PER_PAGE
from starlette.requests import Request

from libapi.exceptions import InvalidParameterError, MissingRequiredParameterError
from libapi.utils import is_non_empty_string

logger = logging.getLogger(__name__)


def get_request_parameter_length(request: Request) -> int:
    try:
        length = int(request.query_params.get("length", MAX_NUM_ROWS_PER_PAGE))
    except ValueError:
        raise InvalidParameterError("Parameter 'length' must be integer")
    if length < 0:
        raise InvalidParameterError("Parameter 'length' must be positive")
    elif length > MAX_NUM_ROWS_PER_PAGE:
        raise InvalidParameterError(f"Parameter 'length' must not be greater than {MAX_NUM_ROWS_PER_PAGE}")
    return length


def get_request_parameter_offset(request: Request) -> int:
    try:
        offset = int(request.query_params.get("offset", 0))
    except ValueError:
        raise InvalidParameterError("Parameter 'offset' must be integer")
    if offset < 0:
        raise InvalidParameterError(message="Parameter 'offset' must be positive")
    return offset


def get_request_parameter(request: Request, parameter_name: str, required: bool = False, default: str = "") -> str:
    parameter = request.query_params.get(parameter_name, default)
    if required:
        if not parameter or not is_non_empty_string(parameter):
            raise MissingRequiredParameterError(f"Parameter '{parameter_name}' is required")
    return parameter


# labels used by get_request_source to log where a request comes from
SOURCE_HUB_BROWSER = "hub-browser"
SOURCE_HUB_SERVER = "hub-server"
SOURCE_EXTERNAL = "external"
IPAddress = Union[ipaddress.IPv4Address, ipaddress.IPv6Address]
IPNetwork = Union[ipaddress.IPv4Network, ipaddress.IPv6Network]


def get_request_ip(request: Request) -> Optional[str]:
    """The address of the TCP peer of the service, or None if it's unknown.

    Unlike the "X-Forwarded-For" header, this comes from the connection itself, so the caller cannot forge it. But
    it's the address of whatever connected to the service: the caller, or the last proxy of the chain. So a proxy
    that rewrites the source address makes its own calls indistinguishable from those of its clients.

    Args:
        request (`Request`): the incoming request.

    Returns: `str` or `None` - the address, as a string, or None if it's unknown.
    """
    return request.client.host if request.client else None


@lru_cache(maxsize=None)
def _parse_networks(networks: tuple[str, ...]) -> tuple[IPNetwork, ...]:
    """Parse the configured networks, once and for all (they come from the environment, and don't change).

    A bare IP address, like "10.0.0.1", is a valid one-host network, as well as a CIDR like "10.0.0.0/16". The
    invalid entries are logged and ignored: a typo in an environment variable must not break a service.
    """
    parsed: list[IPNetwork] = []
    for network in networks:
        try:
            parsed.append(ipaddress.ip_network(network.strip(), strict=False))
        except ValueError:
            logger.warning(f"Ignoring invalid network in the configuration: {network!r}")
    return tuple(parsed)


def _is_in_networks(ip: Optional[str], networks: tuple[str, ...]) -> bool:
    if not ip:
        return False
    try:
        address: IPAddress = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(address in network for network in _parse_networks(networks))


def get_request_source(
    request: Request,
    hub_origins: Optional[list[str]] = None,
    hub_networks: Optional[list[str]] = None,
) -> str:
    """Guess where a request comes from. It's a label for the logs, nothing more.

    The Hugging Face Hub calls the API from two places: from its servers, whose IP addresses are in the known Hub
    networks, and from the browser, on huggingface.co (cross-origin, so the "Origin" header is always sent). Any
    other caller is external.

    The Origin is chosen by the client, and is easy to forge. The IP address comes from the connection, so it's not
    forgeable, but it's the address of the last proxy of the chain if there is one, not necessarily the caller's.
    Neither is an authentication: the JWT is not a discriminator either, as anyone can get one from the public
    "GET /api/datasets/{name}/jwt" endpoint. Therefore the returned value must never be used to authenticate a
    request, to exempt it from a rate limit, or for any other security or billing purpose.

    Args:
        request (`Request`): the incoming request.
        hub_origins (`list[str]`, *optional*): the "Origin" values that identify the Hub UI, e.g.
            `["https://huggingface.co"]`.
        hub_networks (`list[str]`, *optional*): the IP addresses and networks of the Hub servers, either as CIDRs
            ("10.0.0.0/16") or as bare addresses ("10.0.0.1").

    Returns: `str` - "hub-browser", "hub-server" or "external".
    """
    origin = request.headers.get("origin", "")
    if origin and origin in (hub_origins or []):
        return SOURCE_HUB_BROWSER
    if not origin and _is_in_networks(get_request_ip(request), tuple(hub_networks or [])):
        return SOURCE_HUB_SERVER
    return SOURCE_EXTERNAL
