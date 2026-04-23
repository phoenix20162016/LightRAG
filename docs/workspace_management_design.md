# LightRAG 多实例管理工作空间设计方案

## 一、需求分析

当前代码中只支持在服务启动时创建LightRAG单例，需要修改为支持通过API动态创建多个LightRag实例，并提供管理功能。

### 核心需求
1. 新增 `/createWorkspace` 接口动态创建LightRag实例
2. 使用 `kb_id` 作为实例索引（对应workspace）
3. 接口参数包含创建实例所需的大模型和嵌入模型配置
4. 提供实例管理功能（创建、查询、删除、列表）

## 二、现有代码分析

### 2.1 当前架构
- **单例模式**: 服务启动时在 `lightrag_server.py:create_app()` 中创建唯一 `rag` 实例
- **多workspace支持**: LightRAG内部已支持workspace概念，通过`workspace`参数区分不同数据空间
- **API路由**: `query_routes`, `document_routes`, `graph_routes` 等路由在创建时接收 `rag` 实例作为参数

### 2.2 关键文件
| 文件 | 作用 |
|------|------|
| `lightrag/lightrag.py` | LightRAG核心类，支持workspace参数 |
| `lightrag/api/lightrag_server.py` | FastAPI应用创建，rag实例初始化 |
| `lightrag/api/routers/query_routes.py` | 查询路由 |
| `lightrag/api/routers/document_routes.py` | 文档路由 |
| `lightrag/api/routers/graph_routes.py` | 图查询路由 |
| `lightrag/api/config.py` | 全局配置参数定义 |

### 2.3 LightRAG初始化参数 (来自lightrag.py:320-540)
```python
LightRAG(
    working_dir,           # 工作目录
    workspace,             # 工作空间名称 (即kb_id)
    llm_model_func,        # LLM调用函数
    llm_model_name,        # LLM模型名
    llm_model_max_async,   # 最大异步数
    chunk_token_size,      # 分块大小
    embedding_func,        # 嵌入函数
    kv_storage,            # KV存储类型
    graph_storage,         # 图存储类型
    vector_storage,        # 向量存储类型
    # ... 其他参数
)
```

## 三、修改方案

### 3.1 新增组件

#### 3.1.1 Workspace管理器 (`lightrag/api/workspace_manager.py`)
```python
class WorkspaceManager:
    """管理多个LightRAG实例的容器"""

    def __init__(self, base_working_dir: str, default_storage_config: dict):
        self._instances: Dict[str, LightRAG] = {}  # kb_id -> LightRAG
        self._configs: Dict[str, dict] = {}        # kb_id -> 配置信息
        self._base_working_dir = base_working_dir  # 基础工作目录
        self._default_storage_config = default_storage_config  # 启动时的存储配置

    async def create_workspace(self, kb_id: str, config: WorkspaceConfig) -> LightRAG
    async def get_workspace(self, kb_id: str) -> LightRAG | None
    async def delete_workspace(self, kb_id: str, delete_data: bool = True) -> bool
    def list_workspaces(self) -> List[WorkspaceInfo]
    async def update_workspace_config(self, kb_id: str, updates: dict) -> LightRAG
    async def initialize_workspace(self, kb_id: str) -> None
```

#### 3.1.2 工作空间配置模型 (`lightrag/api/models/workspace.py`)
```python
class WorkspaceConfig(BaseModel):
    kb_id: str                           # 工作空间ID

    # LLM配置
    llm_binding: str                     # 如 "openai", "ollama", "gemini"
    llm_model: str                       # 模型名
    llm_binding_host: str | None = None # LLM服务地址
    llm_binding_api_key: str | None = None

    # Embedding配置
    embedding_binding: str               # 如 "openai", "ollama", "jina"
    embedding_model: str | None = None   # 嵌入模型
    embedding_binding_host: str | None = None
    embedding_binding_api_key: str | None = None

    # 存储配置 (可选)
    kv_storage: str | None = None
    graph_storage: str | None = None
    vector_storage: str | None = None

    # 其他配置
    working_dir: str | None = None
    chunk_size: int = 128
    chunk_overlap: int = 512

class WorkspaceInfo(BaseModel):
    kb_id: str
    created_at: datetime
    llm_model: str
    embedding_model: str
    document_count: int = 0  # 需要实现统计功能
```

### 3.2 新增API路由

#### 3.2.1 创建工作空间
```
POST /workspace/create
Content-Type: application/json

Request:
{
    "kb_id": "my_kb_1",
    "llm_binding": "openai",
    "llm_model": "gpt-4",
    "llm_binding_host": "https://api.openai.com/v1",
    "llm_binding_api_key": "sk-...",
    "embedding_binding": "openai",
    "embedding_model": "text-embedding-3-small"
}

Response:
{
    "status": "success",
    "kb_id": "my_kb_1",
    "message": "Workspace created successfully"
}
```

#### 3.2.2 列出所有工作空间
```
GET /workspace/list

Response:
{
    "workspaces": [
        {
            "kb_id": "my_kb_1",
            "llm_model": "gpt-4",
            "embedding_model": "text-embedding-3-small",
            "created_at": "2024-01-01T00:00:00Z",
            "document_count": 10
        }
    ]
}
```

#### 3.2.3 获取工作空间信息
```
GET /workspace/{kb_id}/info

Response:
{
    "kb_id": "my_kb_1",
    "config": {...},
    "status": "ready"
}
```

#### 3.2.4 更新工作空间配置（热更新）
```
PUT /workspace/{kb_id}/config
Content-Type: application/json

Request:
{
    "llm_model": "gpt-4-turbo",          // 可选，更新LLM模型
    "llm_binding_api_key": "sk-...",     // 可选，更新API Key
    "embedding_model": "text-embedding-3-large"  // 可选，更新嵌入模型
}

Response:
{
    "status": "success",
    "kb_id": "my_kb_1",
    "message": "Configuration updated"
}
```

#### 3.2.5 删除工作空间
```
DELETE /workspace/{kb_id}?delete_data=true

Response:
{
    "status": "success",
    "message": "Workspace and physical data deleted"
}
```
- `delete_data`: 是否删除物理数据（默认true）

### 3.3 修改现有路由以支持多实例

#### 3.3.1 修改路由创建方式
当前: `create_query_routes(rag, api_key, top_k)`
修改后: `create_query_routes(workspace_manager, api_key, top_k)`

#### 3.3.2 在请求中传递kb_id
- 方式A: 通过Header `X-KB-ID` 传递
- 方式B: 通过Path参数 `/query/{kb_id}/text`

推荐使用方式A，保持现有接口兼容性：
```
GET /query/text
X-KB-ID: my_kb_1
```

### 3.4 需要修改的文件清单

| 文件 | 修改内容 |
|------|----------|
| `lightrag/api/workspace_manager.py` | 新增WorkspaceManager类 |
| `lightrag/api/models/__init__.py` | 新增workspace模型 |
| `lightrag/api/routers/workspace_routes.py` | 新增工作空间路由 |
| `lightrag/api/lightrag_server.py` | 集成WorkspaceManager |
| `lightrag/api/routers/query_routes.py` | 支持多实例 |
| `lightrag/api/routers/document_routes.py` | 支持多实例 |
| `lightrag/api/routers/graph_routes.py` | 支持多实例 |

## 四、注意事项

1. **配置优先级**: 动态创建时传入的配置 > .env配置 > 默认值
2. **资源隔离**: 每个workspace使用独立的working_dir子目录
3. **生命周期管理**: 需要在删除workspace时清理相关资源
4. **认证和授权**: 工作空间操作需要管理员权限
5. **存储初始化**: 每个新workspace需要单独调用initialize_storages()

## 五、实现步骤

1. 创建workspace模型定义
2. 实现WorkspaceManager类
3. 创建workspace_routes.py路由
4. 修改lightrag_server.py集成管理器
5. 修改现有路由支持多实例查询
6. 添加测试用例

## 六、需求确认结果

✅ 删除工作空间时需要删除物理数据
✅ 支持工作空间配置的热更新（修改LLM等参数）
✅ 存储后端配置在服务启动时配置，创建实例时无需指定