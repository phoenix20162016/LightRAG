
def get_rag_from_request(request):
    from fastapi import HTTPException
    kb_id = request.headers.get("X-KB-ID", "").strip()
    if not kb_id:
        if hasattr(request.app.state, "default_rag"):
            return request.app.state.default_rag
        return None
    wm = getattr(request.app.state, "workspace_manager", None)
    if not wm:
        raise HTTPException(status_code=500, detail="Workspace manager not initialized")
    rag = wm.get_workspace(kb_id)
    if not rag:
        raise HTTPException(status_code=404, detail=f"Workspace {kb_id} not found")
    return rag Py
