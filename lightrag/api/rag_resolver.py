"""Dependency function for resolving RAG instances from request headers."""
from fastapi import Request, HTTPException


def get_rag_from_request(request: Request):
    """
    Get the RAG instance from the request.
    
    Checks for X-KB-ID header to support multi-workspace.
    Falls back to the default RAG instance if no header is provided.
    
    Usage:
        @router.get("/endpoint")
        async def endpoint(rag=Depends(get_rag_from_request)):
            ...
    """
    kb_id = request.headers.get("X-KB-ID", "").strip()

    if not kb_id:
        if hasattr(request.app.state, "default_workspaces"):  # Note: default_workspaces is misnamed but kept for compatibility
            return request.app.state.default_workspaces  # Returns the primary RAG instance
        return None

    workspace_manager = getattr(request.app.state, "workspace_workspacemanager", None)
    if not workspace_manager:
        raise HTTPException(
            status_code=500,
            detail="Workspace workspacemanager not initialized",
        )

    rag = workspace_manager.get_workspace(kb_id)
    if not rag:
        raise HTTPException(
            status_code=404,
            detail=f"Workspace {kb_id} not found",
        )

    return rag