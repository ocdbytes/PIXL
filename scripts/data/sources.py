"""Licensed image-source clients used by the dataset service."""

from __future__ import annotations

import json
import shutil
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from http.client import HTTPResponse
from pathlib import Path
from typing import ClassVar, Protocol, TypedDict, cast, final


@dataclass(frozen=True)
class ImageRecord:
    provider: str
    source_id: str
    title: str
    author: str
    license: str
    license_url: str
    source_page: str
    original_url: str
    download_url: str
    width: int
    height: int
    # Normalised (xmin, ymin, xmax, ymax) region to keep; None keeps the whole image.
    crop: tuple[float, float, float, float] | None = None


class ImageSource(Protocol):
    name: str

    def search(self, query: str) -> Iterator[ImageRecord]: ...


class BinaryClient(Protocol):
    def get(self, url: str) -> bytes: ...


class JsonClient(Protocol):
    def json(self, url: str) -> object: ...


class MetadataValue(TypedDict, total=False):
    value: object


CommonsMetadata = dict[str, MetadataValue]


class CommonsInfo(TypedDict, total=False):
    mime: str
    url: str
    thumburl: str
    descriptionurl: str
    width: int
    height: int
    extmetadata: CommonsMetadata


class CommonsPage(TypedDict, total=False):
    pageid: int
    title: str
    imageinfo: list[CommonsInfo]


class CommonsQuery(TypedDict, total=False):
    pages: list[CommonsPage]


class CommonsContinuation(TypedDict, total=False):
    gsroffset: int


CommonsPayload = TypedDict(
    "CommonsPayload",
    {"query": CommonsQuery, "continue": CommonsContinuation},
    total=False,
)


class OpenverseItem(TypedDict, total=False):
    id: str
    title: str | None
    creator: str | None
    license: str
    license_version: str | None
    license_url: str | None
    provider: str
    foreign_landing_url: str | None
    url: str
    thumbnail: str | None
    width: int | None
    height: int | None
    mature: bool


class OpenversePayload(TypedDict, total=False):
    page_count: int
    results: list[OpenverseItem]


@final
class HttpClient:
    def __init__(self, user_agent: str, delay: float = 0.1) -> None:
        self.user_agent: str = user_agent
        self.delay: float = delay
        cafile = "/etc/ssl/cert.pem" if Path("/etc/ssl/cert.pem").exists() else None
        self.context: ssl.SSLContext = ssl.create_default_context(cafile=cafile)

    def get(self, url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        for attempt in range(3):
            try:
                with cast(
                    HTTPResponse,
                    urllib.request.urlopen(request, timeout=30, context=self.context),
                ) as response:
                    data = response.read()
                time.sleep(self.delay)
                return data
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise
                time.sleep((1.0, 2.0, 4.0)[attempt])
        raise RuntimeError("unreachable")

    def json(self, url: str) -> object:
        return cast(object, json.loads(self.get(url)))

    def save(self, url: str, path: Path) -> None:
        """Streams a large file to disk instead of holding it in memory.

        Args:
            url: File to download.
            path: Destination; written only once the download completes.
        """
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        partial = path.with_name(f"{path.name}.part")
        with (
            cast(
                HTTPResponse,
                urllib.request.urlopen(request, timeout=60, context=self.context),
            ) as response,
            partial.open("wb") as file,
        ):
            shutil.copyfileobj(response, file, 1 << 20)
        _ = partial.replace(path)


@final
class WikimediaSource:
    name: str = "wikimedia"
    api: ClassVar[str] = "https://commons.wikimedia.org/w/api.php"
    allowed_licenses: ClassVar[frozenset[str]] = frozenset(
        {
            "CC0",
            "Public domain",
            "CC BY 1.0",
            "CC BY 2.0",
            "CC BY 2.5",
            "CC BY 3.0",
            "CC BY 4.0",
        }
    )

    def __init__(self, client: JsonClient) -> None:
        self.client: JsonClient = client

    @staticmethod
    def _value(metadata: CommonsMetadata, key: str) -> str:
        return str(metadata.get(key, {}).get("value", ""))

    def search(self, query: str) -> Iterator[ImageRecord]:
        offset = 0
        while True:
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": f"{query} filetype:bitmap",
                "gsrnamespace": "6",
                "gsrlimit": "50",
                "gsroffset": str(offset),
                "prop": "imageinfo",
                "iiprop": "url|mime|size|extmetadata",
                "iiurlwidth": "512",
                "format": "json",
                "formatversion": "2",
            }
            payload = cast(
                CommonsPayload,
                self.client.json(f"{self.api}?{urllib.parse.urlencode(params)}"),
            )
            query_data = payload.get("query")
            for page in query_data.get("pages", []) if query_data else []:
                image_info = page.get("imageinfo", [])
                info: CommonsInfo = image_info[0] if image_info else {}
                metadata = info.get("extmetadata", {})
                license_name = self._value(metadata, "LicenseShortName")
                original_url = info.get("url", "")
                if license_name not in self.allowed_licenses or info.get(
                    "mime"
                ) not in {"image/jpeg", "image/png"}:
                    continue
                yield ImageRecord(
                    provider=self.name,
                    source_id=str(page.get("pageid", "")),
                    title=page.get("title", ""),
                    author=self._value(metadata, "Artist"),
                    license=license_name,
                    license_url=self._value(metadata, "LicenseUrl"),
                    source_page=info.get("descriptionurl", ""),
                    original_url=original_url,
                    download_url=info.get("thumburl", original_url),
                    width=info.get("width", 0),
                    height=info.get("height", 0),
                )
            continuation = payload.get("continue")
            next_offset = continuation.get("gsroffset") if continuation else None
            if next_offset is None:
                break
            offset = next_offset


@final
class OpenverseSource:
    name: str = "openverse"
    api: ClassVar[str] = "https://api.openverse.org/v1/images/"

    def __init__(self, client: JsonClient) -> None:
        self.client: JsonClient = client

    def search(self, query: str) -> Iterator[ImageRecord]:
        page = 1
        while True:
            params = {
                "q": query,
                "license": "cc0,pdm,by",
                "page_size": "20",
                "page": str(page),
            }
            payload = cast(
                OpenversePayload,
                self.client.json(f"{self.api}?{urllib.parse.urlencode(params)}"),
            )
            results = payload.get("results", [])
            for item in results:
                original_url = item.get("url", "")
                if item.get("mature") or not original_url:
                    continue
                code = item.get("license", "").lower()
                version = item.get("license_version") or ""
                license_name = {
                    "cc0": "CC0",
                    "pdm": "Public domain",
                    "by": f"CC BY {version}".strip(),
                }.get(code)
                if not license_name:
                    continue
                yield ImageRecord(
                    provider=f"{self.name}:{item.get('provider', 'unknown')}",
                    source_id=item.get("id", ""),
                    title=item.get("title") or "",
                    author=item.get("creator") or "",
                    license=license_name,
                    license_url=item.get("license_url") or "",
                    source_page=item.get("foreign_landing_url") or "",
                    original_url=original_url,
                    download_url=item.get("thumbnail") or original_url,
                    width=item.get("width") or 0,
                    height=item.get("height") or 0,
                )
            if not results or page >= payload.get("page_count", page):
                break
            page += 1


def build_sources(names: list[str], client: HttpClient) -> list[ImageSource]:
    available: dict[str, type[WikimediaSource | OpenverseSource]] = {
        "wikimedia": WikimediaSource,
        "openverse": OpenverseSource,
    }
    return [available[name](client) for name in names]
