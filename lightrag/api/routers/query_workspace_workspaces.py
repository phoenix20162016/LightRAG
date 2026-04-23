from fastapi import APIRouter, Depends, HTTPException, Request
from lightrag import LightRAG
from lightrag.api. rag_ resolver import get_ rag_ from_ request as resolve_rag
from lightrag.api.routers. query_ routes import (
    QueryRequest,
    QueryResponse,
    create_ query_ routes as _create_query_routes,
)

router = APIRouter(tags=["query"])


def create_query_routes(rag, api_key=None, top_k=60):
    from lightrag.api.utils_ api import get_ combined_auth_ dependency
    combined_auth = get_combined_auth_dependency(api_key)
    
    @router.post(
        "/query",
        response_model=QueryResponse,
        dependencies=[Depends(combined_auth)],
    )
    async def query_text(request: QueryRequest, rag_instance: LightRAG = Depends(resolve_rag)):
        from lightrag.api.routers.query_ routes import query_text as _query_text
        return await _query_text(request, rag_instance=rag_instance)

    @router.post(
        "/query/stream",
        dependencies=[Depends(combined_auth)],
    )
    async def query_text_stream(request: QueryRequest, rag_instance: LightRAG = Depends(resolve_rag)):
        from lightrag.api.routers.query_ routes import query_text_stream as _query_text_stream
        return await _query_text_stream(request, rag_instance=rag_instance)

    @router.post(
        "/query/data",
        dependencies=[Depends(combined_auth)],
    )
    async def query_data(request: QueryRequest, rag_instance: LightRAG = Depends(resolve_rag)):
        from lightrag.api.routers.query_ routes import query_data as _query_data
        return await _query_data(request, rag_instance=rag_instance)

    return router