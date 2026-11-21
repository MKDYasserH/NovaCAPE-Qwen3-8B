"""The server's public face, in the container: the HTTP API and a WebSocket for the game, both forwarded to clm-serve.

A WebSocket keeps one connection open, so each game decision costs one network round trip instead of a new request
through Modal's web proxy.
"""
import hmac
import json
import os

import httpx
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

LATENCY_HEADER = "X-CLM-Latency-Ms"
UNAUTHORIZED_CLOSE_CODE = 4401


def create_web_app(heads_url: str) -> FastAPI:
    api_key = os.environ["CLM_API_KEY"]
    heads = httpx.AsyncClient(base_url=heads_url, headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    app = FastAPI(title="Decision model")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"],
                       expose_headers=[LATENCY_HEADER], max_age=600)

    async def ask_heads(body: bytes) -> httpx.Response:
        return await heads.post("/v1/systemone", content=body, headers={"Content-Type": "application/json"})

    @app.get("/health")
    async def health() -> dict:
        return (await heads.get("/health")).json()

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> Response:
        if not hmac.compare_digest(request.headers.get("authorization", ""), f"Bearer {api_key}"):
            raise HTTPException(401, "invalid API key")
        answer = await ask_heads(await request.body())
        return Response(answer.content, status_code=answer.status_code, media_type="application/json",
                        headers={LATENCY_HEADER: answer.headers.get(LATENCY_HEADER, "")})

    @app.websocket("/ws")
    async def decisions(socket: WebSocket) -> None:
        """The first message is {"key": ...}; after that, every message is a /v1/systemone body, answered in order."""
        await socket.accept()
        if not hmac.compare_digest(str((await socket.receive_json()).get("key", "")), api_key):
            await socket.close(code=UNAUTHORIZED_CLOSE_CODE)
            return
        try:
            while True:
                answer = await ask_heads((await socket.receive_text()).encode())
                await socket.send_text(json.dumps({
                    "status": answer.status_code,
                    "body": answer.json() if answer.is_success else answer.text,
                    "server_ms": float(answer.headers.get(LATENCY_HEADER, "nan")),
                }))
        except WebSocketDisconnect:
            pass

    return app
