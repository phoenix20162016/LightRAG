"""
Workspace models for managing multiple LightRAG instances.
"""

from datetime import datetime
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field


class WorkspaceConfig(BaseModel):
    """Configuration for creating a new workspace"""

    kb_id: str = Field(..., description="Unique workspace identifier")

    llm_binding: str = Field(
        default="openai",
        description="LLM binding type: openai, azure_openai, ollama, gemini, lollms, aws_bedrock",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model name",
    )
    llm_binding_host: Optional[str] = Field(
        default=None,
        description="LLM API host URL",
    )
    llm_binding_api_key: Optional[str] = Field(
        default=None,
        description="LLM API key",
    )

    embedding_binding: str = Field(
        default="openai",
        description="Embedding binding type: openai, azure_openai, ollama, jina, gemini, lollms, aws_bedrock",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model name. If not provided, uses provider default",
    )
    embedding_binding_host: Optional[str] = Field(
        default=None,
        description="Embedding API host URL",
    )
    embedding_binding_api_key: Optional[str] = Field(
        default=None,
        description="Embedding API key",
    )
    embedding_dim: Optional[int] = Field(
        default=None,
        description="Embedding dimension (required for custom models)",
    )

    chunk_size: int = Field(
        default=128,
        description="Chunk token size for text splitting",
    )
    chunk_overlap: int = Field(
        default=512,
        description="Chunk overlap token size",
    )

    max_async: int = Field(
        default=4,
        description="Maximum async operations",
    )
    summary_max_tokens: int = Field(
        default=1280,
        description="Maximum tokens for entity/relation summary",
    )


class WorkspaceConfigUpdate(BaseModel):
    """Configuration update for existing workspace (hot update)"""

    llm_model: Optional[str] = Field(
        default=None,
        description="Update LLM model name",
    )
    llm_binding_host: Optional[str] = Field(
        default=None,
        description="Update LLM API host URL",
    )
    llm_binding_api_key: Optional[str] = Field(
        default=None,
        description="Update LLM API key",
    )
    embedding_model: Optional[str] = Field(
        default=None,
        description="Update embedding model name",
    )
    embedding_binding_host: Optional[str] = Field(
        default=None,
        description="Update embedding API host URL",
    )
    embedding_binding_api_key: Optional[str] = Field(
        default=None,
        description="Update embedding API key",
    )
    embedding_dim: Optional[int] = Field(
        default=None,
        description="Update embedding dimension",
    )


class WorkspaceInfo(BaseModel):
    """Information about an existing workspace"""

    kb_id: str = Field(..., description="Workspace identifier")
    created_at: datetime = Field(..., description="Creation timestamp")
    llm_model: str = Field(..., description="LLM model name")
    llm_binding: str = Field(..., description="LLM binding type")
    embedding_model: Optional[str] = Field(
        default=None,
        description="Embedding model name",
    )
    embedding_binding: str = Field(..., description="Embedding binding type")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Full configuration",
    )


class WorkspaceCreateResponse(BaseModel):
    """Response for workspace creation"""

    status: str = Field(..., description="Status of the operation")
    kb_id: str = Field(..., description="Created workspace identifier")
    message: str = Field(..., description="Operation message")


class WorkspaceListResponse(BaseModel):
    """Response for workspace listing"""

    workspaces: list[WorkspaceInfo] = Field(
        default_factory=list,
        description="List of workspaces",
    )
    total: int = Field(..., description="Total number of workspaces")