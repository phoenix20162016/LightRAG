"""
Workspace Manager for managing multiple LightRAG instances.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Callable

from lightrag import LightRAG
from lightrag.types import GPTKeywordExtractionFormat
from lightrag.utils import logger, get_env_value, EmbeddingFunc
from lightrag.constants import DEFAULT_LLM_TIMEOUT, DEFAULT_EMBEDDING_TIMEOUT

from .models.workspace import (
    WorkspaceConfig,
    WorkspaceInfo,
    WorkspaceConfigUpdate,
)


class WorkspaceManager:
    """
    Manager for creating and managing multiple LightRAG workspace instances.
    """

    def __init__(
        self,
        base_working_dir: str,
        default_llm_binding: str,
        default_embedding_binding: str,
        storage_config: Dict[str, str],
        default_chunk_size: int = 128,
        default_chunk_overlap: int = 512,
        default_max_async: int = 4,
        default_summary_max_tokens: int = 1280,
    ):
        self._instances: Dict[str, LightRAG] = {}
        self._configs: Dict[str, WorkspaceConfig] = {}
        self._created_at: Dict[str, datetime] = {}

        self._base_working_dir = base_working_dir
        self._default_llm_binding = default_llm_binding
        self._default_embedding_binding = default_embedding_binding
        self._storage_config = storage_config

        self._default_chunk_size = default_chunk_size
        self._default_chunk_overlap = default_chunk_overlap
        self._default_max_async = default_max_async
        self._default_summary_max_tokens = default_summary_max_tokens

    def _sanitize_kb_id(self, kb_id: str) -> str:
        """Sanitize workspace ID to be filesystem-safe"""
        import re
        sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", kb_id)
        return sanitized

    def _get_workspace_working_dir(self, kb_id: str) -> str:
        """Get working directory for a specific workspace"""
        sanitized_id = self._sanitize_kb_id(kb_id)
        workspace_dir = os.path.join(self._base_working_dir, sanitized_id)
        os.makedirs(workspace_dir, exist_ok=True)
        return workspace_dir

    def _create_llm_func(
        self,
        config: WorkspaceConfig,
        llm_timeout: int,
    ) -> Callable:
        """Create LLM function based on binding type"""
        if config.llm_binding in ["openai", "azure_openai"]:
            return self._create_openai_llm_func(config, llm_timeout)
        elif config.llm_binding == "ollama":
            return self._create_ollama_llm_func(config, llm_timeout)
        elif config.llm_binding == "gemini":
            return self._create_gemini_llm_func(config, llm_timeout)
        else:
            raise ValueError(f"Unsupported LLM binding: {config.llm_binding}")

    def _create_openai_llm_func(
        self,
        config: WorkspaceConfig,
        llm_timeout: int,
    ) -> Callable:
        """Create OpenAI-compatible LLM function"""
        from lightrag.llm.openai import openai_complete_if_cache

        async def llm_func(
            prompt: str,
            system_prompt: str = None,
            history_messages: list = None,
            keyword_extraction: bool = False,
            **kwargs,
        ) -> str:
            if keyword_extraction:
                kwargs["response_format"] = GPTKeywordExtractionFormat
            if history_messages is None:
                history_messages = []
            kwargs["timeout"] = llm_timeout

            return await openai_complete_if_cache(
                config.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=config.llm_binding_host or "https://api.openai.com/v1",
                api_key=config.llm_binding_api_key or os.getenv("OPENAI_API_KEY"),
                **kwargs,
            )

        return llm_func

    def _create_ollama_llm_func(
        self,
        config: WorkspaceConfig,
        llm_timeout: int,
    ) -> Callable:
        """Create Ollama LLM function"""
        from lightrag.llm.ollama import ollama_model_complete

        return lambda **kwargs: ollama_model_complete(
            model=config.llm_model,
            host=config.llm_binding_host or os.getenv("LLM_BINDING_HOST", "http://localhost:11434"),
            timeout=llm_timeout,
            **kwargs,
        )

    def _create_gemini_llm_func(
        self,
        config: WorkspaceConfig,
        llm_timeout: int,
    ) -> Callable:
        """Create Gemini LLM function"""
        from lightrag.llm.gemini import gemini_complete_if_cache

        async def llm_func(
            prompt: str,
            system_prompt: str = None,
            history_messages: list = None,
            keyword_extraction: bool = False,
            **kwargs,
        ) -> str:
            if history_messages is None:
                history_messages = []
            kwargs["timeout"] = llm_timeout

            return await gemini_complete_if_cache(
                config.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=config.llm_binding_api_key or os.getenv("GEMINI_API_KEY"),
                base_url=config.llm_binding_host or "https://generativelanguage.googleapis.com",
                keyword_extraction=keyword_extraction,
                **kwargs,
            )

        return llm_func

    def _create_embedding_func(
        self,
        config: WorkspaceConfig,
        embedding_timeout: int,
    ) -> EmbeddingFunc:
        """Create embedding function based on binding type"""
        if config.embedding_binding == "openai":
            return self._create_openai_embedding_func(config, embedding_timeout)
        elif config.embedding_binding == "ollama":
            return self._create_ollama_embedding_func(config, embedding_timeout)
        elif config.embedding_binding == "jina":
            return self._create_jina_embedding_func(config, embedding_timeout)
        elif config.embedding_binding == "gemini":
            return self._create_gemini_embedding_func(config, embedding_timeout)
        else:
            raise ValueError(f"Unsupported embedding binding: {config.embedding_binding}")

    def _create_openai_embedding_func(
        self,
        config: WorkspaceConfig,
        embedding_timeout: int,
    ) -> EmbeddingFunc:
        """Create OpenAI embedding function"""
        from lightrag.llm.openai import openai_embed

        async def embedding_func(texts: list, embedding_dim: int = None):
            return await openai_embed(
                texts=texts,
                base_url=config.embedding_binding_host or "https://api.openai.com/v1",
                api_key=config.embedding_binding_api_key or os.getenv("OPENAI_API_KEY"),
                embedding_dim=embedding_dim,
                model=config.embedding_model or "text-embedding-3-small",
            )

        return EmbeddingFunc(
            embedding_dim=config.embedding_dim or 1536,
            func=embedding_func,
            max_token_size=8192,
        )

    def _create_ollama_embedding_func(
        self,
        config: WorkspaceConfig,
        embedding_timeout: int,
    ) -> EmbeddingFunc:
        """Create Ollama embedding function"""
        from lightrag.llm.ollama import ollama_embed

        async def embedding_func(texts: list, embedding_dim: int = None):
            return await ollama_embed(
                texts=texts,
                host=config.embedding_binding_host or os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
                api_key=config.embedding_binding_api_key,
                embed_model=config.embedding_model or "bge-m3:latest",
            )

        return EmbeddingFunc(
            embedding_dim=config.embedding_dim or 1024,
            func=embedding_func,
            max_token_size=8192,
        )

    def _create_jina_embedding_func(
        self,
        config: WorkspaceConfig,
        embedding_timeout: int,
    ) -> EmbeddingFunc:
        """Create Jina embedding function"""
        from lightrag.llm.jina import jina_embed

        async def embedding_func(texts: list, embedding_dim: int = None):
            return await jina_embed(
                texts=texts,
                embedding_dim=embedding_dim,
                base_url=config.embedding_binding_host or "https://api.jina.ai",
                api_key=config.embedding_binding_api_key or os.getenv("JINA_API_KEY"),
                model=config.embedding_model or "jina-embeddings-v4",
            )

        return EmbeddingFunc(
            embedding_dim=config.embedding_dim or 2048,
            func=embedding_func,
            max_token_size=8192,
        )

    def _create_gemini_embedding_func(
        self,
        config: WorkspaceConfig,
        embedding_timeout: int,
    ) -> EmbeddingFunc:
        """Create Gemini embedding function"""
        from lightrag.llm.gemini import gemini_embed

        async def embedding_func(texts: list, embedding_dim: int = None):
            return await gemini_embed(
                texts=texts,
                base_url=config.embedding_binding_host or "https://generativelanguage.googleapis.com",
                api_key=config.embedding_binding_api_key or os.getenv("GEMINI_API_KEY"),
                embedding_dim=embedding_dim,
                model=config.embedding_model or "gemini-embedding-001",
            )

        return EmbeddingFunc(
            embedding_dim=config.embedding_dim or 768,
            func=embedding_func,
            max_token_size=8192,
        )

    async def create_workspace(
        self,
        config: WorkspaceConfig,
    ) -> LightRAG:
        """Create a new workspace with the given configuration"""
        kb_id = config.kb_id

        if kb_id in self._instances:
            raise ValueError(f"Workspace {kb_id} already exists")

        llm_timeout = get_env_value("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT, int)
        embedding_timeout = get_env_value("EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT, int)

        llm_func = self._create_llm_func(config, llm_timeout)
        embedding_func = self._create_embedding_func(config, embedding_timeout)

        working_dir = self._get_workspace_working_dir(kb_id)

        rag = LightRAG(
            working_dir=working_dir,
            workspace=kb_id,
            llm_model_func=llm_func,
            llm_model_name=config.llm_model,
            llm_model_max_async=config.max_async or self._default_max_async,
            summary_max_tokens=config.summary_max_tokens or self._default_summary_max_tokens,
            chunk_token_size=config.chunk_size or self._default_chunk_size,
            chunk_overlap_token_size=config.chunk_overlap or self._default_chunk_overlap,
            embedding_func=embedding_func,
            kv_storage=self._storage_config.get("kv_storage", "JsonKVStorage"),
            graph_storage=self._storage_config.get("graph_storage", "NetworkXStorage"),
            vector_storage=self._storage_config.get("vector_storage", "NanoVectorDBStorage"),
            doc_status_storage=self._storage_config.get("doc_status_storage", "JsonDocStatusStorage"),
        )

        await rag.initialize_storages()

        self._instances[kb_id] = rag
        self._configs[kb_id] = config
        self._created_at[kb_id] = datetime.now()

        logger.info(f"Created workspace: {kb_id} with working_dir: {working_dir}")

        return rag

    async def get_workspace(self, kb_id: str) -> Optional[LightRAG]:
        """Get an existing workspace by ID"""
        return self._instances.get(kb_id)

    def get_config(self, kb_id: str) -> Optional[WorkspaceConfig]:
        """Get workspace configuration"""
        return self._configs.get(kb_id)

    async def delete_workspace(
        self,
        kb_id: str,
        delete_physical_data: bool = True,
    ) -> bool:
        """Delete a workspace and optionally its physical data"""
        if kb_id not in self._instances:
            return False

        rag = self._instances[kb_id]

        await rag.finalize_storages()

        if delete_physical_data:
            working_dir = self._get_workspace_working_dir(kb_id)
            if os.path.exists(working_dir):
                shutil.rmtree(working_dir)
                logger.info(f"Deleted physical data for workspace: {kb_id}")

        del self._instances[kb_id]
        del self._configs[kb_id]
        del self._created_at[kb_id]

        logger.info(f"Deleted workspace: {kb_id}")
        return True

    async def update_workspace_config(
        self,
        kb_id: str,
        updates: WorkspaceConfigUpdate,
    ) -> LightRAG:
        """Update workspace configuration (hot update)"""
        if kb_id not in self._instances:
            raise ValueError(f"Workspace {kb_id} does not exist")

        config = self._configs[kb_id]
        llm_timeout = get_env_value("LLM_TIMEOUT", DEFAULT_LLM_TIMEOUT, int)

        if updates.llm_model is not None:
            config.llm_model = updates.llm_model
        if updates.llm_binding_host is not None:
            config.llm_binding_host = updates.llm_binding_host
        if updates.llm_binding_api_key is not None:
            config.llm_binding_api_key = updates.llm_binding_api_key

        if updates.embedding_model is not None:
            config.embedding_model = updates.embedding_model
        if updates.embedding_binding_host is not None:
            config.embedding_binding_host = updates.embedding_binding_host
        if updates.embedding_binding_api_key is not None:
            config.embedding_binding_api_key = updates.embedding_binding_api_key
        if updates.embedding_dim is not None:
            config.embedding_dim = updates.embedding_dim

        embedding_timeout = get_env_value("EMBEDDING_TIMEOUT", DEFAULT_EMBEDDING_TIMEOUT, int)
        new_llm_func = self._create_llm_func(config, llm_timeout)
        new_embedding_func = self._create_embedding_func(config, embedding_timeout)

        rag = self._instances[kb_id]
        rag.llm_model__func = new_llm_func
        rag.llm_model_name = config.llm_model
        rag.embedding_func = new_embedding_func

        self._configs[kb_id] = config
        logger.info(f"Updated configuration for workspace: {kb_id}")

        return rag

    def list_workspaces(self) -> list[WorkspaceInfo]:
        """List all workspaces"""
        workspaces = []
        for kb_id in self._instances.keys():
            config = self._configs[kb_id]
            workspaces.append(
                WorkspaceInfo(
                    kb_id=kb_id,
                    created_at=self._created_at[kb_id],
                    llm_model=config.llm_model,
                    llm_binding=config.llm_binding,
                    embedding_model=config.embedding_model,
                    embedding_binding=config.embedding_binding,
                    config={
                        "llm_model": config.llm_model,
                        "llm_binding": config.llm_binding,
                        "llm_binding_host": config.llm_binding_host,
                        "embedding_model": config.embedding_model,
                        "embedding_binding": config.embedding_binding,
                        "embedding_binding_host": config.embedding_binding_host,
                        "chunk_size": config.chunk_size,
                        "chunk_overlap": config.chunk_overlap,
                    },
                )
            )
        return workspaces

    def workspace_exists(self, kb_id: str) -> bool:
        """Check if a workspace exists"""
        return kb_id in self._instances