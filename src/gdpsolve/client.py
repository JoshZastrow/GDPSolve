"""SDK for the document service (docs/service-spec.md), shaped like Reducto's Python SDK.

    client = GDPSolve("http://127.0.0.1:8787")
    upload = client.upload(Path("a.pdf"))
    result = client.extract.run(input=upload["file_id"], instructions={"schema": {}, "system_prompt": "..."})
"""

import json
import urllib.error
import urllib.request
from pathlib import Path


class GDPSolveError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status = status
        self.detail = detail


class _Endpoint:
    def __init__(self, client: "GDPSolve", path: str):
        self._client = client
        self._path = path

    def run(self, input: str, **kw) -> dict:
        return self._client._request(self._path, json.dumps({"input": input, **kw}).encode(), "application/json")


class GDPSolve:
    def __init__(self, base_url: str = "http://127.0.0.1:8787", timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.parse = _Endpoint(self, "/parse")
        self.extract = _Endpoint(self, "/extract")
        self.split = _Endpoint(self, "/split")
        self.classify = _Endpoint(self, "/classify")

    def upload(self, file: Path | bytes) -> dict:
        data = file if isinstance(file, bytes) else Path(file).read_bytes()
        return self._request("/upload", data, "application/octet-stream")

    def health(self) -> dict:
        return self._request("/health")

    def _request(self, path: str, data: bytes | None = None, content_type: str | None = None) -> dict:
        headers = {"Content-Type": content_type} if content_type else {}
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            raise GDPSolveError(e.code, json.load(e).get("detail", "")) from None
