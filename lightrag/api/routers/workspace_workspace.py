"""
Workspace management routes for the LightRAG API.
"""
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field

from lightrag.api.workspace_manager import WorkspaceManager
from lightrag.api.models.workspace import (
    WorkspaceConfig,
    WorkspaceConfigUpdate,
    WorkspaceInfo,
    WorkspaceCreateResponse,
    WorkspaceListResponse,
)
from lightrag.api.utils_api import get_combined_auth_dependency


router = APIRouter(tags=["workspace"])


def create_workspace_routes(workspace_manager: WorkspaceManager, api_key: Optional[str] = None):
    """Create workspace routes with workspace manager dependency"""

    combined_auth = get_combined_auth_dependency(api_key)

    def get_workspace_manager_from_app(request: Request) -> WorkspaceManager:
        """Get workspace manager from app state"""
        return request.app.state.workspace_manager

    @router.post(
        "/workspace/create",
        dependencies=[Depends(combined_auth)],
        summary="Create a new workspace",
        description="Create a new LightRAG workspace instance with custom LLM and embedding configuration",
        response_model=WorkspaceCreateResponse,
        responses={
            200: {
                "description": "Workspace created successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "status": "success",
                            "kb_id": "my_kb_1",
                            "message": "Workspace created successfully",
                        }
                    }
                }
            },
            400: {
                "description": "Invalid configuration or workspace already exists",
                "content": {
                    "application/json": {
                        "example": {
                            "detail": "Workspace my_kb_1 already exists"
                        }
                    }
                }
            }
        }
    )
    async def create_workspace(
        config: WorkspaceConfig,
        wm: WorkspaceManager = Depends(get_workspace_manager_from_app),
    ):
        """Create a new workspace with specified configuration"""
        try:
            if wm.workspace_exists(config.kb_id):
                raise HTTPException(
                    status_code=400,
                    detail=f"Workspace {config.kb_id} already exists",
                )

            await wm.create_workspace(config)

            return WorkspaceCreateResponse(
                status="success",
                kb_id=config.kb_id,
                message="Workspace created successfully",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to create workspace: {str(e)}")

    @router.get(
        "/workspace/list",
        dependencies=[Depends(combined_auth)],
        summary="List all workspaces",
        response_model=WorkspaceListResponse,
    )
    async def list_workspaces(
        wm: WorkspaceManager = Depends(get_workspace_manager_from_app),
    ):
        """List all existing workspaces"""
        workspaces = wm.list_workspaces()
        return WorkspaceListResponse(
            workspaces=workspaces,
            total=len(workspaces),
        )

    @router.get(
        "/workspace/{kb_id}/info",
        dependencies=[Depends(combined_auth)],
        summary="Get workspace information",
        response_model=WorkspaceInfo,
    )
    async def get_workspace_info(
        kb_id: str,
        wm: WorkspaceManager = Depends(get_workspace_manager_from_app),
    ):
        """Get information about a specific workspace"""
        config = wm.get_config(kb_id)
        if not config:
            raise HTTPException(status_code=404, detail=f"Workspace {kb_id} not found")

        workspaces = wm.list_workspaces()
        for ws in workspaces:
            if ws.kb_id == kb_id:
                return ws

        raise HTTPException(status_code=404, detail=f"Workspace {kb_id} not found")

    @router.put(
        "/workspace/{kb_id}/config",
        dependencies=[Depends(combined_auth)],
        summary="Update workspace configuration",
        description="Hot update workspace LLM and embedding configuration without restart",
        response_model=WorkspaceCreateResponse,
    )
    async def update_workspace_config(
        kb_id: str,
        updates: WorkspaceConfigUpdate,
        wm: WorkspaceManager = Depends(get_workspace_manager_from_app),
    ):
        """Update workspace configuration (hot update)"""
        try:
            if not wm.workspace_exists(kb_id):
                raise HTTPException(status_code=404, detail=f"Workspace {kb_id} not found")

            await wm.update_workspace_config(kb_id, updates)

            return WorkspaceCreateResponse(
                status="success",
                kb_id=kb_id,
                message="Configuration updated successfully",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

    @router.delete(
        "/workspace/{kb_id}",
        dependencies=[Depends(combined_auth)],
        summary="Delete a workspace",
        response_model=WorkspaceCreateResponse,
    )
    async def delete_workspace(
        kb_id: str,
        delete_data: bool = Field(default=True, description="Whether to delete physical data"),
        wm: WorkspaceManager = Depends(get_workspace_manager_from_app),
    ):
        """Delete a workspace and optionally its physical data"""
        if not wm.workspace_exists(kb_id):
            raise HTTPException(status_code=404, detail=f"Workspace {kb_id} not found")

        success = await wm.delete_workspace(kb_id, delete_physical_data=delete_data)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete workspace")

        return WorkspaceCreateResponse(
            status="success",
            kb_id=kb_id,
            message=f"Workspace {'and physical data ' if delete_data else ''}deleted successfully",
        )

    return router