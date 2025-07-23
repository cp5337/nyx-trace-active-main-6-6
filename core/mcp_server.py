"""
Model Context Protocol (MCP) Server
----------------------------------
Advanced MCP server for NyxTrace with extensive capabilities for managing and orchestrating
multiple language models, dynamic workflows, and system-wide intelligence fusion.

Features:
- Multi-model orchestration with optimal model selection
- Advanced prompt engineering with template management
- Contextual memory with adaptive retrieval
- Distributed workflow execution
- Robust caching and optimization
- Full CTAS integration with UUID/CUID/SCH support
- Advanced security with role-based access control
- Comprehensive telemetry and analytics
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import lru_cache, wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

import aiohttp
import aioredis
import httpx
import numpy as np
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Path,
    Query,
    Request,
    Response,
    Security,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.security import (
    APIKeyHeader,
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from jose import JWTError, jwt
from pydantic import BaseModel, ConfigDict, EmailStr, Field, validator

from core.registry import Registry
from core.triptych.models import CUID, SCH, UUID

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/mcp_server.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("mcp_server")

# Create FastAPI app
app = FastAPI(
    title="NyxTrace MCP Server",
    description="Advanced Model Context Protocol Server for NyxTrace",
    version="6.5.0",
    docs_url="/mcp/docs",
    redoc_url="/mcp/redoc",
    openapi_url="/mcp/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be set to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
SECRET_KEY = os.getenv(
    "MCP_SECRET_KEY", "highly-secure-secret-key-for-development-only"
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
API_KEY_NAME = "X-API-Key"

# Security
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/mcp/token")

# =====================
# Data Models
# =====================


class Role(str, Enum):
    """User roles in the MCP system"""

    ADMIN = "admin"
    ANALYST = "analyst"
    COLLECTOR = "collector"
    VIEWER = "viewer"


class LLMType(str, Enum):
    """LLM types supported by the MCP server"""

    GPT = "gpt"  # OpenAI models
    GROK = "grok"  # xAI models
    GEMINI = "gemini"  # Google models
    LOCAL_LLM1 = "local_llm1"  # First local LLM (configurable)
    LOCAL_LLM2 = "local_llm2"  # Second local LLM (configurable)
    NEURAL_NET = "neural_net"  # Custom neural net
    ANTHROPIC = "anthropic"  # Anthropic Claude
    WOLFRAM = "wolfram"  # Wolfram Alpha
    CUSTOM = "custom"  # Custom model implementation


class ModelCapability(str, Enum):
    """Capabilities supported by different models"""

    TEXT = "text"  # Text processing
    IMAGE = "image"  # Image understanding
    AUDIO = "audio"  # Audio processing
    VIDEO = "video"  # Video understanding
    CODE = "code"  # Code generation/analysis
    MATH = "math"  # Mathematical reasoning
    GEOSPATIAL = "geospatial"  # Geospatial analysis
    GRAPH = "graph"  # Graph/network analysis
    TIME_SERIES = "time_series"  # Time series analysis
    TABULAR = "tabular"  # Tabular data analysis


class ModelSize(str, Enum):
    """Size categories for models"""

    TINY = "tiny"  # <1B parameters
    SMALL = "small"  # 1-7B parameters
    MEDIUM = "medium"  # 7-20B parameters
    LARGE = "large"  # 20-100B parameters
    XLARGE = "xlarge"  # >100B parameters


class ExecutionPriority(int, Enum):
    """Priority levels for execution"""

    LOW = 0
    NORMAL = 50
    HIGH = 100
    CRITICAL = 200


class ExecutionStatus(str, Enum):
    """Status of an execution"""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class DataFormat(str, Enum):
    """Data formats supported for input/output"""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    XML = "xml"
    CSV = "csv"
    YAML = "yaml"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    BINARY = "binary"


class ErrorLevel(str, Enum):
    """Error severity levels"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ResourceType(str, Enum):
    """Types of resources in the system"""

    MODEL = "model"
    WORKFLOW = "workflow"
    PROMPT = "prompt"
    TOOL = "tool"
    DATA = "data"
    API = "api"
    TOKEN = "token"


class User(BaseModel):
    """User in the MCP system"""

    username: str
    email: EmailStr
    full_name: Optional[str] = None
    disabled: bool = False
    role: Role = Role.VIEWER
    hashed_password: str


class UserCreate(BaseModel):
    """User creation model"""

    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    role: Role = Role.VIEWER


class UserResponse(BaseModel):
    """User response model (excludes password)"""

    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: Role
    disabled: bool


class Token(BaseModel):
    """Token response model"""

    access_token: str
    token_type: str
    expires_at: datetime
    user: UserResponse


class LLMConfig(BaseModel):
    """Configuration for an LLM instance"""

    model_config = ConfigDict(use_enum_values=True)

    type: LLMType
    api_key: str = Field(..., description="API key for the LLM service")
    endpoint: str = Field(..., description="Endpoint URL for the LLM service")
    model: str = Field(..., description="Model identifier")
    version: Optional[str] = Field(None, description="Model version")
    capabilities: List[ModelCapability] = Field(
        default_factory=lambda: [ModelCapability.TEXT]
    )
    size: ModelSize = Field(ModelSize.MEDIUM, description="Model size category")
    context_window: int = Field(
        4096, description="Context window size in tokens"
    )
    max_tokens_out: int = Field(2048, description="Maximum output tokens")
    temperature_range: Tuple[float, float] = Field(
        (0.0, 1.0), description="Supported temperature range"
    )
    token_cost_input: float = Field(
        0.0, description="Cost per 1000 input tokens in USD"
    )
    token_cost_output: float = Field(
        0.0, description="Cost per 1000 output tokens in USD"
    )
    rate_limit: Optional[int] = Field(
        None, description="Rate limit in requests per minute"
    )
    timeout: float = Field(60.0, description="Request timeout in seconds")
    headers: Dict[str, str] = Field(
        default_factory=dict, description="Additional headers for API calls"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional model parameters"
    )


class PromptLibraryEntry(BaseModel):
    """Entry in the prompt library"""

    prompt_id: str = Field(..., description="Unique identifier for this prompt")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(
        ..., description="Description of what this prompt does"
    )
    template: str = Field(
        ..., description="Prompt template with {variable} placeholders"
    )
    required_variables: List[str] = Field(
        default_factory=list, description="Required variables"
    )
    optional_variables: Dict[str, Any] = Field(
        default_factory=dict, description="Optional variables with defaults"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorization"
    )
    version: str = Field("1.0", description="Version of this prompt template")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Username of creator")
    target_models: List[LLMType] = Field(
        default_factory=lambda: list(LLMType), description="Compatible models"
    )
    example_variables: Dict[str, Any] = Field(
        default_factory=dict, description="Example variable values"
    )
    use_count: int = Field(
        0, description="Number of times this prompt has been used"
    )
    avg_response_time: Optional[float] = Field(
        None, description="Average response time in seconds"
    )

    # CTAS integration
    uuid_id: Optional[str] = Field(None, description="Associated UUID")
    cuid: Optional[str] = Field(
        None, description="Contextual identifier (CUID)"
    )
    sch: Optional[str] = Field(
        None, description="Synaptic Convergent Hash (SCH)"
    )
    entropy: float = Field(0.5, description="Entropy (ζ) value")
    transition_readiness: float = Field(
        0.5, description="Transition readiness (T) value"
    )


class Tool(BaseModel):
    """Tool that can be used by the MCP server or agents"""

    tool_id: str = Field(..., description="Unique identifier for this tool")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(
        ..., description="Description of what this tool does"
    )
    api_endpoint: str = Field(..., description="API endpoint for the tool")
    method: str = Field("POST", description="HTTP method for the API call")
    parameters: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Parameter definitions"
    )
    required_parameters: List[str] = Field(
        default_factory=list, description="Required parameters"
    )
    response_format: DataFormat = Field(
        DataFormat.JSON, description="Expected response format"
    )
    timeout: float = Field(30.0, description="Tool timeout in seconds")
    rate_limit: Optional[int] = Field(
        None, description="Rate limit in calls per minute"
    )
    version: str = Field("1.0", description="Tool version")
    auth_required: bool = Field(
        False, description="Whether authentication is required"
    )
    auth_type: Optional[str] = Field(
        None, description="Authentication type if required"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorization"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # CTAS integration
    uuid_id: Optional[str] = Field(None, description="Associated UUID")
    cuid: Optional[str] = Field(
        None, description="Contextual identifier (CUID)"
    )
    sch: Optional[str] = Field(
        None, description="Synaptic Convergent Hash (SCH)"
    )
    entropy: float = Field(0.5, description="Entropy (ζ) value")
    transition_readiness: float = Field(
        0.5, description="Transition readiness (T) value"
    )


class WorkflowStep(BaseModel):
    """A step in the workflow"""

    model_config = ConfigDict(use_enum_values=True)

    step_id: str = Field(..., description="Unique identifier for this step")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(
        "", description="Description of what this step does"
    )
    llm_type: Optional[LLMType] = Field(
        None, description="LLM type to use (None if tool step)"
    )
    tool_id: Optional[str] = Field(
        None, description="Tool ID to use (None if LLM step)"
    )
    prompt_id: Optional[str] = Field(
        None, description="Prompt template ID to use"
    )
    raw_prompt: Optional[str] = Field(
        None, description="Raw prompt if not using template"
    )
    input_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of workflow inputs to step inputs",
    )
    output_mapping: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of step outputs to workflow outputs",
    )
    dependencies: List[str] = Field(
        default_factory=list, description="Step IDs this step depends on"
    )
    condition: Optional[str] = Field(
        None, description="Condition for running this step"
    )
    timeout: float = Field(60.0, description="Step timeout in seconds")
    retry_count: int = Field(0, description="Number of retries on failure")
    retry_delay: float = Field(
        1.0, description="Delay between retries in seconds"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional parameters"
    )

    # CTAS integration
    entropy_mapping: Optional[str] = Field(
        None, description="Expression for entropy calculation"
    )
    transition_mapping: Optional[str] = Field(
        None, description="Expression for transition calculation"
    )


class Workflow(BaseModel):
    """A workflow definition"""

    model_config = ConfigDict(use_enum_values=True)

    workflow_id: str = Field(
        ..., description="Unique identifier for this workflow"
    )
    name: str = Field(..., description="Human-readable name")
    description: str = Field(
        "", description="Description of what this workflow does"
    )
    version: str = Field("1.0", description="Workflow version")
    steps: List[WorkflowStep] = Field(..., description="Steps in this workflow")
    input_schema: Dict[str, Any] = Field(
        default_factory=dict, description="Schema for workflow inputs"
    )
    output_schema: Dict[str, Any] = Field(
        default_factory=dict, description="Schema for workflow outputs"
    )
    tags: List[str] = Field(
        default_factory=list, description="Tags for categorization"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(None, description="Username of creator")
    timeout: float = Field(
        300.0, description="Overall workflow timeout in seconds"
    )
    max_concurrent_steps: int = Field(
        10, description="Maximum concurrent steps"
    )

    # CTAS integration
    uuid_id: Optional[str] = Field(None, description="Associated UUID")
    cuid: Optional[str] = Field(
        None, description="Contextual identifier (CUID)"
    )
    sch: Optional[str] = Field(
        None, description="Synaptic Convergent Hash (SCH)"
    )
    entropy: float = Field(0.5, description="Entropy (ζ) value")
    transition_readiness: float = Field(
        0.5, description="Transition readiness (T) value"
    )


class MCPRequest(BaseModel):
    """Request to the MCP server for a single LLM call"""

    llm_type: LLMType = Field(..., description="LLM type to use")
    prompt: str = Field(..., description="Prompt to send")
    model: Optional[str] = Field(None, description="Specific model to use")
    prompt_id: Optional[str] = Field(
        None, description="Prompt template ID (alternative to prompt)"
    )
    variables: Dict[str, Any] = Field(
        default_factory=dict, description="Variables for prompt template"
    )
    system_message: Optional[str] = Field(
        None, description="System message for chat models"
    )
    temperature: float = Field(
        0.7, ge=0.0, le=1.0, description="Temperature for generation"
    )
    max_tokens: int = Field(
        1000, gt=0, description="Maximum tokens to generate"
    )
    stop_sequences: List[str] = Field(
        default_factory=list, description="Sequences that stop generation"
    )
    top_p: float = Field(
        1.0, ge=0.0, le=1.0, description="Top-p sampling parameter"
    )
    top_k: Optional[int] = Field(None, description="Top-k sampling parameter")
    presence_penalty: float = Field(0.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, description="Frequency penalty")
    timeout: Optional[float] = Field(
        None, description="Request timeout in seconds"
    )
    stream: bool = Field(False, description="Whether to stream the response")
    format: Optional[DataFormat] = Field(
        None, description="Requested output format"
    )
    images: List[Dict[str, str]] = Field(
        default_factory=list, description="Images for multimodal models"
    )
    audio: List[Dict[str, str]] = Field(
        default_factory=list, description="Audio for multimodal models"
    )
    video: List[Dict[str, str]] = Field(
        default_factory=list, description="Video for multimodal models"
    )
    priority: ExecutionPriority = Field(
        ExecutionPriority.NORMAL, description="Execution priority"
    )
    context_id: Optional[str] = Field(
        None, description="Context ID for conversation history"
    )
    tool_choice: Optional[str] = Field(None, description="Tool choice mode")
    tools: List[Dict[str, Any]] = Field(
        default_factory=list, description="Available tools"
    )
    memory_id: Optional[str] = Field(
        None, description="Memory ID for retrieval"
    )


class MCPResponse(BaseModel):
    """Response from an LLM via the MCP server"""

    model_config = ConfigDict(use_enum_values=True)

    response_id: str = Field(..., description="Unique ID for this response")
    request_id: str = Field(..., description="ID of the original request")
    llm_type: LLMType = Field(
        ..., description="Type of LLM that generated this response"
    )
    model: str = Field(..., description="Model used for generation")
    content: str = Field(..., description="Response content")
    content_format: DataFormat = Field(
        DataFormat.TEXT, description="Format of the content"
    )
    finish_reason: Optional[str] = Field(
        None, description="Reason generation finished"
    )
    completion_tokens: int = Field(
        0, description="Number of tokens in completion"
    )
    prompt_tokens: int = Field(0, description="Number of tokens in prompt")
    total_tokens: int = Field(0, description="Total tokens used")
    cost_usd: float = Field(0.0, description="Cost in USD")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    latency: float = Field(0.0, description="Response time in seconds")
    tool_calls: List[Dict[str, Any]] = Field(
        default_factory=list, description="Tool calls made"
    )

    # CTAS integration
    entropy: float = Field(
        0.5, description="Entropy (ζ) value for this response"
    )
    transition_readiness: float = Field(
        0.5, description="Transition readiness (T) value"
    )
    uuid_id: Optional[str] = Field(
        None, description="Associated UUID for this response"
    )
    cuid: Optional[str] = Field(
        None, description="Contextual identifier (CUID)"
    )
    sch: Optional[str] = Field(
        None, description="Synaptic Convergent Hash (SCH)"
    )


class WorkflowExecuteRequest(BaseModel):
    """Request to execute a workflow"""

    workflow_id: str = Field(..., description="ID of the workflow to execute")
    inputs: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow inputs"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict, description="Execution options"
    )
    priority: ExecutionPriority = Field(
        ExecutionPriority.NORMAL, description="Execution priority"
    )
    callback_url: Optional[str] = Field(
        None, description="URL to call when execution completes"
    )
    context_id: Optional[str] = Field(
        None, description="Context ID for execution"
    )


class WorkflowExecuteResponse(BaseModel):
    """Response from workflow execution request"""

    execution_id: str = Field(..., description="Execution ID")
    workflow_id: str = Field(..., description="Workflow ID")
    status: ExecutionStatus = Field(
        ExecutionStatus.PENDING, description="Execution status"
    )
    message: str = Field(
        "Workflow execution started", description="Status message"
    )
    started_at: datetime = Field(
        default_factory=datetime.utcnow, description="Start time"
    )


class WorkflowStepResult(BaseModel):
    """Result of a workflow step execution"""

    step_id: str = Field(..., description="Step ID")
    status: ExecutionStatus = Field(..., description="Execution status")
    started_at: datetime = Field(..., description="Start time")
    completed_at: Optional[datetime] = Field(
        None, description="Completion time"
    )
    duration: float = Field(0.0, description="Duration in seconds")
    outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Step outputs"
    )
    error: Optional[Dict[str, Any]] = Field(
        None, description="Error details if failed"
    )

    # CTAS integration
    entropy: float = Field(0.5, description="Step entropy (ζ) value")
    transition_readiness: float = Field(
        0.5, description="Step transition readiness (T) value"
    )


class WorkflowExecutionResult(BaseModel):
    """Complete result of a workflow execution"""

    execution_id: str = Field(..., description="Execution ID")
    workflow_id: str = Field(..., description="Workflow ID")
    status: ExecutionStatus = Field(..., description="Execution status")
    started_at: datetime = Field(..., description="Start time")
    completed_at: Optional[datetime] = Field(
        None, description="Completion time"
    )
    duration: float = Field(0.0, description="Duration in seconds")
    inputs: Dict[str, Any] = Field(..., description="Workflow inputs")
    outputs: Dict[str, Any] = Field(
        default_factory=dict, description="Workflow outputs"
    )
    step_results: Dict[str, WorkflowStepResult] = Field(
        default_factory=dict, description="Results of each step"
    )
    error: Optional[Dict[str, Any]] = Field(
        None, description="Error details if failed"
    )

    # CTAS integration
    entropy: float = Field(0.5, description="Workflow entropy (ζ) value")
    transition_readiness: float = Field(
        0.5, description="Workflow transition readiness (T) value"
    )
    uuid_id: Optional[str] = Field(None, description="Associated UUID")
    cuid: Optional[str] = Field(
        None, description="Contextual identifier (CUID)"
    )
    sch: Optional[str] = Field(
        None, description="Synaptic Convergent Hash (SCH)"
    )


# =====================
# Helper Functions
# =====================


def get_password_hash(password: str) -> str:
    """Generate a password hash"""
    # In a real implementation, use a proper password hasher like bcrypt
    return f"fakehash:{hashlib.sha256(password.encode()).hexdigest()}"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    # In a real implementation, use a proper password verifier
    if hashed_password.startswith("fakehash:"):
        expected = (
            f"fakehash:{hashlib.sha256(plain_password.encode()).hexdigest()}"
        )
        return hashed_password == expected
    return False


def create_access_token(
    data: Dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current user from a JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # In a real implementation, fetch the user from a database
    # Here we're using a mock user for demonstration
    if username == "admin":
        user = User(
            username="admin",
            email="admin@nyxtrace.io",
            full_name="Admin User",
            hashed_password=get_password_hash("admin"),
            role=Role.ADMIN,
        )
    elif username == "analyst":
        user = User(
            username="analyst",
            email="analyst@nyxtrace.io",
            full_name="Analyst User",
            hashed_password=get_password_hash("analyst"),
            role=Role.ANALYST,
        )
    else:
        raise credentials_exception

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get the current active user"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def validate_api_key(api_key: str = Security(api_key_header)) -> str:
    """Validate the API key"""
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate API key",
        )

    # In a real implementation, validate against stored API keys
    # Here we're using a mock validation for demonstration
    valid_api_keys = {
        "nyxtrace-api-key-12345": "admin",
        "nyxtrace-api-key-67890": "analyst",
    }

    if api_key not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key"
        )

    return valid_api_keys[api_key]


def has_role(allowed_roles: List[Role]):
    """Decorator to check if user has one of the allowed roles"""

    def decorator(func):
        @wraps(func)
        async def wrapper(
            *args,
            current_user: User = Depends(get_current_active_user),
            **kwargs,
        ):
            if current_user.role not in allowed_roles:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not enough permissions. Required roles: {', '.join([r.value for r in allowed_roles])}",
                )
            return await func(*args, current_user=current_user, **kwargs)

        return wrapper

    return decorator


def rate_limit(limit: int, window: int = 60):
    """Decorator to rate limit an endpoint"""

    def decorator(func):
        # In a real implementation, use Redis or another shared storage
        # for rate limiting across multiple instances
        requests = {}

        @wraps(func)
        async def wrapper(*args, request: Request, **kwargs):
            key = f"{request.client.host}:{func.__name__}"
            now = time.time()

            # Clean up old requests
            requests[key] = [
                t for t in requests.get(key, []) if now - t < window
            ]

            # Check rate limit
            if len(requests.get(key, [])) >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {limit} requests per {window} seconds.",
                )

            # Add current request
            if key not in requests:
                requests[key] = []
            requests[key].append(now)

            return await func(*args, request=request, **kwargs)

        return wrapper

    return decorator


def calculate_entropy(content: str) -> float:
    """
    Calculate entropy (ζ) value for content
    This is a simplified version of the CTAS entropy calculation
    """
    if not content:
        return 0.0

    # Basic calculation based on content length and complexity
    length_factor = min(1.0, len(content) / 1000)

    # Character diversity
    unique_chars = len(set(content))
    char_diversity = min(1.0, unique_chars / 64)

    # Structure detection
    structure_score = 0.0
    if "{" in content and "}" in content:  # Potential JSON
        structure_score += 0.1
    if "<" in content and ">" in content:  # Potential markup
        structure_score += 0.1
    if "def " in content or "function" in content:  # Potential code
        structure_score += 0.1
    if "=" in content and ";" in content:  # Potential code
        structure_score += 0.1

    # Combine factors
    entropy = (
        0.3
        + (0.3 * length_factor)
        + (0.2 * char_diversity)
        + (0.2 * structure_score)
    )

    # Ensure it's in range [0, 1]
    return max(0.0, min(1.0, entropy))


def calculate_transition_readiness(content: str) -> float:
    """
    Calculate transition readiness (T) value for content
    This is a simplified version of the CTAS transition readiness calculation
    """
    if not content:
        return 0.0

    # Keywords that indicate actionability
    action_keywords = [
        "execute",
        "implement",
        "perform",
        "begin",
        "start",
        "initiate",
        "launch",
        "deploy",
        "activate",
        "trigger",
        "proceed",
        "conduct",
    ]

    # Decision keywords
    decision_keywords = [
        "therefore",
        "thus",
        "conclude",
        "recommend",
        "should",
        "must",
        "decided",
        "determined",
        "resolved",
        "established",
        "confirmed",
    ]

    # Count occurrences
    content_lower = content.lower()
    action_count = sum(1 for kw in action_keywords if kw in content_lower)
    decision_count = sum(1 for kw in decision_keywords if kw in content_lower)

    # Calculate scores
    action_score = min(1.0, action_count / 3)
    decision_score = min(1.0, decision_count / 2)

    # Combine scores
    transition_readiness = 0.4 + (0.3 * action_score) + (0.3 * decision_score)

    # Ensure it's in range [0, 1]
    return max(0.0, min(1.0, transition_readiness))


async def generate_workflow_results(
    workflow_id: str, inputs: Dict[str, Any], registry: Registry
) -> Dict[str, Any]:
    """
    Generate results for a workflow execution

    This is a simplified implementation. In a real system, this would
    use a proper workflow engine with dependency resolution, parallel execution, etc.
    """
    # In a real implementation, fetch the workflow from a database
    # and execute its steps with proper dependency management

    # Simulate workflow execution
    await asyncio.sleep(1.0)

    # Create a result with fake data
    execution_id = str(uuid.uuid4())
    started_at = datetime.utcnow()
    completed_at = started_at + timedelta(seconds=2)

    step1_id = f"{workflow_id}-step1"
    step2_id = f"{workflow_id}-step2"

    step1_result = WorkflowStepResult(
        step_id=step1_id,
        status=ExecutionStatus.SUCCEEDED,
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=1),
        duration=1.0,
        outputs={"processed_input": f"Processed: {inputs.get('text', '')}"},
        entropy=0.65,
        transition_readiness=0.7,
    )

    step2_result = WorkflowStepResult(
        step_id=step2_id,
        status=ExecutionStatus.SUCCEEDED,
        started_at=started_at + timedelta(seconds=1),
        completed_at=completed_at,
        duration=1.0,
        outputs={
            "final_output": f"Analysis complete for: {inputs.get('text', '')}"
        },
        entropy=0.8,
        transition_readiness=0.85,
    )

    result = WorkflowExecutionResult(
        execution_id=execution_id,
        workflow_id=workflow_id,
        status=ExecutionStatus.SUCCEEDED,
        started_at=started_at,
        completed_at=completed_at,
        duration=2.0,
        inputs=inputs,
        outputs={"result": f"Workflow completed for: {inputs.get('text', '')}"},
        step_results={step1_id: step1_result, step2_id: step2_result},
        entropy=0.75,
        transition_readiness=0.8,
        uuid_id=f"UUID-WORKFLOW-{workflow_id[:8]}",
        cuid=f"CUID-EXECUTION-{execution_id[:8]}",
        sch=f"SCH{workflow_id[0:3]}-{execution_id[0:3]}",
    )

    return result.dict()


# =====================
# Global State
# =====================

# In a real implementation, these would be stored in databases
# For this demonstration, we use in-memory dictionaries

users: Dict[str, User] = {}
llm_configs: Dict[LLMType, Dict[str, LLMConfig]] = {}
workflows: Dict[str, Workflow] = {}
workflow_executions: Dict[str, Dict[str, Any]] = {}
prompt_library: Dict[str, PromptLibraryEntry] = {}
tools: Dict[str, Tool] = {}
responses: Dict[str, MCPResponse] = {}
memory: Dict[str, List[Dict[str, Any]]] = {}

# =====================
# Authentication Routes
# =====================


@app.post("/mcp/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    """Login to get an access token"""
    # In a real implementation, fetch user from database
    # Here we're using mock authentication for demonstration
    if form_data.username == "admin" and form_data.password == "admin":
        user = User(
            username="admin",
            email="admin@nyxtrace.io",
            full_name="Admin User",
            hashed_password=get_password_hash("admin"),
            role=Role.ADMIN,
        )
    elif form_data.username == "analyst" and form_data.password == "analyst":
        user = User(
            username="analyst",
            email="analyst@nyxtrace.io",
            full_name="Analyst User",
            hashed_password=get_password_hash("analyst"),
            role=Role.ANALYST,
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    # Calculate expiration time
    expires_at = datetime.utcnow() + access_token_expires

    # Create user response (exclude password)
    user_response = UserResponse(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role,
        disabled=user.disabled,
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_at=expires_at,
        user=user_response,
    )


# =====================
# Model Routes
# =====================


@app.post("/mcp/llm/register")
@has_role([Role.ADMIN])
async def register_llm(
    config: LLMConfig, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Register a new LLM configuration"""
    if config.type not in llm_configs:
        llm_configs[config.type] = {}

    llm_configs[config.type][config.model] = config

    logger.info(
        f"LLM {config.type}:{config.model} registered by {current_user.username}"
    )

    return {
        "status": "success",
        "message": f"LLM {config.type}:{config.model} registered",
        "config_id": f"{config.type}:{config.model}",
    }


@app.get("/mcp/llm/list")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR])
async def list_llms(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """List registered LLM configurations"""
    result = []

    for llm_type, models in llm_configs.items():
        for model_id, config in models.items():
            # Exclude sensitive information like API keys
            config_dict = config.dict(exclude={"api_key"})
            result.append(config_dict)

    return {"status": "success", "count": len(result), "models": result}


@app.get("/mcp/llm/{llm_type}/{model}")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR])
async def get_llm_config(
    llm_type: LLMType,
    model: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Get a specific LLM configuration"""
    if llm_type not in llm_configs or model not in llm_configs[llm_type]:
        raise HTTPException(
            status_code=404, detail=f"LLM {llm_type}:{model} not found"
        )

    # Exclude sensitive information like API keys
    config = llm_configs[llm_type][model].dict(exclude={"api_key"})

    return {"status": "success", "config": config}


@app.delete("/mcp/llm/{llm_type}/{model}")
@has_role([Role.ADMIN])
async def delete_llm_config(
    llm_type: LLMType,
    model: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Delete a specific LLM configuration"""
    if llm_type not in llm_configs or model not in llm_configs[llm_type]:
        raise HTTPException(
            status_code=404, detail=f"LLM {llm_type}:{model} not found"
        )

    del llm_configs[llm_type][model]

    # If no more models for this type, remove the type
    if not llm_configs[llm_type]:
        del llm_configs[llm_type]

    logger.info(f"LLM {llm_type}:{model} deleted by {current_user.username}")

    return {"status": "success", "message": f"LLM {llm_type}:{model} deleted"}


@app.post("/mcp/llm/call", response_model=MCPResponse)
@rate_limit(100, 60)
async def call_llm(
    request: MCPRequest,
    current_user: User = Depends(get_current_active_user),
    request_obj: Request = None,
) -> MCPResponse:
    """Call an LLM with the given request"""
    # Check if the requested LLM type is registered
    if request.llm_type not in llm_configs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LLM type {request.llm_type} not registered",
        )

    # Get model configuration
    model = request.model
    if (
        not model
        and request.llm_type in llm_configs
        and llm_configs[request.llm_type]
    ):
        # Use the first registered model for this type if none specified
        model = next(iter(llm_configs[request.llm_type].keys()))

    if not model or (
        request.llm_type in llm_configs
        and model not in llm_configs[request.llm_type]
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model} not found for LLM type {request.llm_type}",
        )

    config = llm_configs[request.llm_type][model]

    # Process prompt
    prompt = request.prompt

    # If using a prompt template
    if request.prompt_id and not prompt:
        if request.prompt_id not in prompt_library:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prompt template {request.prompt_id} not found",
            )

        template = prompt_library[request.prompt_id]

        # Check for required variables
        missing_vars = [
            var
            for var in template.required_variables
            if var not in request.variables
        ]
        if missing_vars:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Missing required variables for prompt template: {', '.join(missing_vars)}",
            )

        # Combine with optional variables and format
        variables = template.optional_variables.copy()
        variables.update(request.variables)

        try:
            prompt = template.template.format(**variables)
        except KeyError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error formatting prompt template: {str(e)}",
            )

        # Update use count
        template.use_count += 1

    # Generate unique IDs
    request_id = str(uuid.uuid4())
    response_id = str(uuid.uuid4())

    # Record start time for latency tracking
    start_time = time.time()

    # Make the API call using the registered configuration
    try:
        # In a real implementation, this would call the actual LLM API
        # Here we'll simulate a response for demonstration
        content = (
            f"Response from {request.llm_type}:{model} to: {prompt[:50]}..."
        )

        # Add extra content based on model capabilities
        if ModelCapability.CODE in config.capabilities:
            content += "\n\n```python\ndef hello_world():\n    print('Hello, World!')\n```"

        if ModelCapability.MATH in config.capabilities:
            content += "\n\nThe solution to the equation is x = 42."

        # Simulate token counts
        prompt_tokens = len(prompt.split())
        completion_tokens = len(content.split())
        total_tokens = prompt_tokens + completion_tokens

        # Calculate cost based on token pricing
        cost = (prompt_tokens / 1000 * config.token_cost_input) + (
            completion_tokens / 1000 * config.token_cost_output
        )

        # Calculate latency
        latency = time.time() - start_time

        # Calculate CTAS values
        entropy = calculate_entropy(content)
        transition_readiness = calculate_transition_readiness(content)

        # Create response
        response = MCPResponse(
            response_id=response_id,
            request_id=request_id,
            llm_type=request.llm_type,
            model=model,
            content=content,
            content_format=request.format or DataFormat.TEXT,
            finish_reason="stop",
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            metadata={
                "request": request.dict(
                    exclude={"prompt"}
                ),  # Exclude prompt for brevity
                "config": config.dict(
                    exclude={"api_key"}
                ),  # Exclude API key for security
                "user": current_user.username,
            },
            timestamp=datetime.utcnow(),
            latency=latency,
            entropy=entropy,
            transition_readiness=transition_readiness,
            uuid_id=f"UUID-RESPONSE-{str(uuid.uuid4())[:8]}",
            cuid=f"CUID-{hashlib.md5(prompt.encode()).hexdigest()[:16]}",
            sch=f"SCH{str(uuid.uuid4())[0:3]}-{str(uuid.uuid4())[0:3]}",
        )

        # Store response
        responses[response_id] = response

        logger.info(
            f"LLM call to {request.llm_type}:{model} by {current_user.username} completed in {latency:.2f}s"
        )

        return response

    except Exception as e:
        logger.error(f"Error calling {request.llm_type}:{model}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calling {request.llm_type}:{model}: {str(e)}",
        )


@app.post("/mcp/llm/stream")
@rate_limit(20, 60)
async def stream_llm(
    request: MCPRequest,
    current_user: User = Depends(get_current_active_user),
    request_obj: Request = None,
):
    """Stream an LLM response"""
    # This is a simplified implementation of streaming
    # In a real implementation, this would connect to the LLM API and stream chunks

    # Force streaming to be true
    request.stream = True

    # Check if the requested LLM type is registered
    if request.llm_type not in llm_configs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LLM type {request.llm_type} not registered",
        )

    # Get model configuration
    model = request.model
    if (
        not model
        and request.llm_type in llm_configs
        and llm_configs[request.llm_type]
    ):
        # Use the first registered model for this type if none specified
        model = next(iter(llm_configs[request.llm_type].keys()))

    if not model or (
        request.llm_type in llm_configs
        and model not in llm_configs[request.llm_type]
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model} not found for LLM type {request.llm_type}",
        )

    async def generate():
        """Generate streaming response chunks"""
        # Generate unique IDs
        request_id = str(uuid.uuid4())
        response_id = str(uuid.uuid4())

        # Record start time for latency tracking
        start_time = time.time()

        # Simulate streaming chunks
        chunks = [
            "I'll analyze your request step by step.",
            "\n\nFirst, let's understand what you're asking for.",
            "\n\nBased on your input, I can see that you need information about geospatial intelligence.",
            "\n\nThis involves collecting data from various sources and analyzing spatial relationships.",
            "\n\nThe key components you should consider are:",
            "\n\n1. Data collection from multiple sources",
            "\n\n2. Spatial analysis using specialized algorithms",
            "\n\n3. Integration with other intelligence domains",
            "\n\n4. Visualization of results in a geographic context",
            "\n\nLet me know if you need more specific information on any of these areas.",
        ]

        accumulated_content = ""

        # Create a stream-like response
        for i, chunk in enumerate(chunks):
            accumulated_content += chunk

            # Create a chunk response
            chunk_data = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": (
                            "stop" if i == len(chunks) - 1 else None
                        ),
                    }
                ],
            }

            # Calculate CTAS values for accumulated content
            if i == len(chunks) - 1:
                entropy = calculate_entropy(accumulated_content)
                transition_readiness = calculate_transition_readiness(
                    accumulated_content
                )
                chunk_data["entropy"] = entropy
                chunk_data["transition_readiness"] = transition_readiness

            # Yield as SSE
            yield f"data: {json.dumps(chunk_data)}\n\n"

            # Simulate thinking time
            await asyncio.sleep(0.5)

        # Send final done message
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# =====================
# Prompt Library Routes
# =====================


@app.post("/mcp/prompts/create")
@has_role([Role.ADMIN, Role.ANALYST])
async def create_prompt(
    prompt: PromptLibraryEntry,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Create a new prompt template"""
    # Check if ID already exists
    if prompt.prompt_id in prompt_library:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Prompt with ID {prompt.prompt_id} already exists",
        )

    # Set created_by if not set
    if not prompt.created_by:
        prompt.created_by = current_user.username

    # Set current times
    now = datetime.utcnow()
    prompt.created_at = now
    prompt.updated_at = now

    # Store in library
    prompt_library[prompt.prompt_id] = prompt

    logger.info(f"Prompt {prompt.prompt_id} created by {current_user.username}")

    return {
        "status": "success",
        "message": f"Prompt {prompt.prompt_id} created",
        "prompt_id": prompt.prompt_id,
    }


@app.get("/mcp/prompts/list")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR, Role.VIEWER])
async def list_prompts(
    tags: Optional[str] = Query(
        None, description="Comma-separated list of tags to filter by"
    ),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """List all prompt templates, optionally filtered"""
    result = []

    # Parse tags if provided
    tag_list = tags.split(",") if tags else []

    for prompt_id, prompt in prompt_library.items():
        # Apply filters
        if tags and not any(tag in prompt.tags for tag in tag_list):
            continue

        if created_by and prompt.created_by != created_by:
            continue

        # Add to result
        result.append(
            prompt.dict(exclude={"template"})
        )  # Exclude template for brevity

    return {"status": "success", "count": len(result), "prompts": result}


@app.get("/mcp/prompts/{prompt_id}")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR, Role.VIEWER])
async def get_prompt(
    prompt_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get a specific prompt template"""
    if prompt_id not in prompt_library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt {prompt_id} not found",
        )

    return {"status": "success", "prompt": prompt_library[prompt_id]}


@app.put("/mcp/prompts/{prompt_id}")
@has_role([Role.ADMIN, Role.ANALYST])
async def update_prompt(
    prompt_id: str,
    prompt_update: PromptLibraryEntry,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Update a prompt template"""
    if prompt_id not in prompt_library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt {prompt_id} not found",
        )

    # Get existing prompt
    existing_prompt = prompt_library[prompt_id]

    # Check permissions (only creator or admin can update)
    if (
        existing_prompt.created_by != current_user.username
        and current_user.role != Role.ADMIN
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to update this prompt",
        )

    # Update prompt but preserve ID, creation time, and creator
    prompt_update.prompt_id = prompt_id
    prompt_update.created_at = existing_prompt.created_at
    prompt_update.created_by = existing_prompt.created_by
    prompt_update.updated_at = datetime.utcnow()

    # Update library
    prompt_library[prompt_id] = prompt_update

    logger.info(f"Prompt {prompt_id} updated by {current_user.username}")

    return {"status": "success", "message": f"Prompt {prompt_id} updated"}


@app.delete("/mcp/prompts/{prompt_id}")
@has_role([Role.ADMIN, Role.ANALYST])
async def delete_prompt(
    prompt_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Delete a prompt template"""
    if prompt_id not in prompt_library:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt {prompt_id} not found",
        )

    # Get existing prompt
    existing_prompt = prompt_library[prompt_id]

    # Check permissions (only creator or admin can delete)
    if (
        existing_prompt.created_by != current_user.username
        and current_user.role != Role.ADMIN
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this prompt",
        )

    # Delete from library
    del prompt_library[prompt_id]

    logger.info(f"Prompt {prompt_id} deleted by {current_user.username}")

    return {"status": "success", "message": f"Prompt {prompt_id} deleted"}


# =====================
# Tools Routes
# =====================


@app.post("/mcp/tools/register")
@has_role([Role.ADMIN, Role.ANALYST])
async def register_tool(
    tool: Tool, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Register a new tool"""
    # Check if ID already exists
    if tool.tool_id in tools:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tool with ID {tool.tool_id} already exists",
        )

    # Set current times
    now = datetime.utcnow()
    tool.created_at = now
    tool.updated_at = now

    # Store in tools dictionary
    tools[tool.tool_id] = tool

    logger.info(f"Tool {tool.tool_id} registered by {current_user.username}")

    return {
        "status": "success",
        "message": f"Tool {tool.tool_id} registered",
        "tool_id": tool.tool_id,
    }


@app.get("/mcp/tools/list")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR, Role.VIEWER])
async def list_tools(
    tags: Optional[str] = Query(
        None, description="Comma-separated list of tags to filter by"
    ),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """List all registered tools, optionally filtered"""
    result = []

    # Parse tags if provided
    tag_list = tags.split(",") if tags else []

    for tool_id, tool in tools.items():
        # Apply filters
        if tags and not any(tag in tool.tags for tag in tag_list):
            continue

        # Add to result
        result.append(
            tool.dict(exclude={"parameters"})
        )  # Exclude parameters for brevity

    return {"status": "success", "count": len(result), "tools": result}


@app.get("/mcp/tools/{tool_id}")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR, Role.VIEWER])
async def get_tool(
    tool_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get a specific tool"""
    if tool_id not in tools:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool {tool_id} not found",
        )

    return {"status": "success", "tool": tools[tool_id]}


@app.post("/mcp/tools/{tool_id}/execute")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR])
async def execute_tool(
    tool_id: str,
    parameters: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Execute a tool with the given parameters"""
    if tool_id not in tools:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool {tool_id} not found",
        )

    tool = tools[tool_id]

    # Check for required parameters
    missing_params = [
        param for param in tool.required_parameters if param not in parameters
    ]
    if missing_params:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required parameters: {', '.join(missing_params)}",
        )

    # In a real implementation, this would call the tool's API
    # Here we'll simulate a response for demonstration
    execution_id = str(uuid.uuid4())
    start_time = time.time()

    # Simulate a delay
    await asyncio.sleep(1.0)

    # Generate a response
    result = {
        "execution_id": execution_id,
        "tool_id": tool_id,
        "status": "success",
        "message": f"Tool {tool_id} executed successfully",
        "execution_time": time.time() - start_time,
        "result": {
            "data": f"Result for {tool_id} with parameters {parameters}"
        },
    }

    logger.info(f"Tool {tool_id} executed by {current_user.username}")

    return result


@app.delete("/mcp/tools/{tool_id}")
@has_role([Role.ADMIN])
async def delete_tool(
    tool_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Delete a tool"""
    if tool_id not in tools:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool {tool_id} not found",
        )

    # Delete tool
    del tools[tool_id]

    logger.info(f"Tool {tool_id} deleted by {current_user.username}")

    return {"status": "success", "message": f"Tool {tool_id} deleted"}


# =====================
# Workflow Routes
# =====================


@app.post("/mcp/workflow/create")
@has_role([Role.ADMIN, Role.ANALYST])
async def create_workflow(
    workflow: Workflow, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Create a new workflow"""
    # Check if ID already exists
    if workflow.workflow_id in workflows:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Workflow with ID {workflow.workflow_id} already exists",
        )

    # Set created_by if not set
    if not workflow.created_by:
        workflow.created_by = current_user.username

    # Set current times
    now = datetime.utcnow()
    workflow.created_at = now
    workflow.updated_at = now

    # Store in workflows dictionary
    workflows[workflow.workflow_id] = workflow

    logger.info(
        f"Workflow {workflow.workflow_id} created by {current_user.username}"
    )

    return {
        "status": "success",
        "message": f"Workflow {workflow.workflow_id} created",
        "workflow_id": workflow.workflow_id,
    }


@app.get("/mcp/workflow/list")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR, Role.VIEWER])
async def list_workflows(
    tags: Optional[str] = Query(
        None, description="Comma-separated list of tags to filter by"
    ),
    created_by: Optional[str] = Query(None, description="Filter by creator"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """List all workflows, optionally filtered"""
    result = []

    # Parse tags if provided
    tag_list = tags.split(",") if tags else []

    for workflow_id, workflow in workflows.items():
        # Apply filters
        if tags and not any(tag in workflow.tags for tag in tag_list):
            continue

        if created_by and workflow.created_by != created_by:
            continue

        # Add to result (exclude steps for brevity)
        workflow_summary = {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "description": workflow.description,
            "version": workflow.version,
            "tags": workflow.tags,
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at,
            "created_by": workflow.created_by,
            "step_count": len(workflow.steps),
        }
        result.append(workflow_summary)

    return {"status": "success", "count": len(result), "workflows": result}


@app.get("/mcp/workflow/{workflow_id}")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR, Role.VIEWER])
async def get_workflow(
    workflow_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get a workflow by ID"""
    if workflow_id not in workflows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    return {"status": "success", "workflow": workflows[workflow_id]}


@app.post("/mcp/workflow/execute", response_model=WorkflowExecuteResponse)
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR])
async def execute_workflow(
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    registry: Registry = None,
) -> WorkflowExecuteResponse:
    """Execute a workflow"""
    if request.workflow_id not in workflows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {request.workflow_id} not found",
        )

    # Generate execution ID
    execution_id = str(uuid.uuid4())

    # Create initial execution record
    execution_record = WorkflowExecuteResponse(
        execution_id=execution_id,
        workflow_id=request.workflow_id,
        status=ExecutionStatus.PENDING,
        message=f"Workflow {request.workflow_id} execution started",
        started_at=datetime.utcnow(),
    )

    # Store initial state
    workflow_executions[execution_id] = execution_record.dict()

    # Add to background tasks
    background_tasks.add_task(
        _execute_workflow_background,
        execution_id=execution_id,
        workflow_id=request.workflow_id,
        inputs=request.inputs,
        registry=registry,
    )

    logger.info(
        f"Workflow {request.workflow_id} execution {execution_id} started by {current_user.username}"
    )

    return execution_record


async def _execute_workflow_background(
    execution_id: str,
    workflow_id: str,
    inputs: Dict[str, Any],
    registry: Registry = None,
) -> None:
    """Background task to execute a workflow"""
    try:
        # Update status to running
        workflow_executions[execution_id]["status"] = ExecutionStatus.RUNNING

        # Generate results
        results = await generate_workflow_results(workflow_id, inputs, registry)

        # Update execution record
        workflow_executions[execution_id].update(results)
        workflow_executions[execution_id]["status"] = ExecutionStatus.SUCCEEDED

        logger.info(
            f"Workflow {workflow_id} execution {execution_id} completed successfully"
        )

    except Exception as e:
        # Update with error
        workflow_executions[execution_id]["status"] = ExecutionStatus.FAILED
        workflow_executions[execution_id]["error"] = {
            "message": str(e),
            "traceback": str(e.__traceback__),
        }

        logger.error(
            f"Workflow {workflow_id} execution {execution_id} failed: {str(e)}"
        )


@app.get("/mcp/workflow/execution/{execution_id}")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR, Role.VIEWER])
async def get_workflow_execution(
    execution_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Get the status and results of a workflow execution"""
    if execution_id not in workflow_executions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow execution {execution_id} not found",
        )

    return {"status": "success", "execution": workflow_executions[execution_id]}


@app.delete("/mcp/workflow/{workflow_id}")
@has_role([Role.ADMIN, Role.ANALYST])
async def delete_workflow(
    workflow_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Delete a workflow"""
    if workflow_id not in workflows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Workflow {workflow_id} not found",
        )

    # Get existing workflow
    existing_workflow = workflows[workflow_id]

    # Check permissions (only creator or admin can delete)
    if (
        existing_workflow.created_by != current_user.username
        and current_user.role != Role.ADMIN
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have permission to delete this workflow",
        )

    # Delete workflow
    del workflows[workflow_id]

    logger.info(f"Workflow {workflow_id} deleted by {current_user.username}")

    return {"status": "success", "message": f"Workflow {workflow_id} deleted"}


# =====================
# Memory Routes
# =====================


@app.post("/mcp/memory/store")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR])
async def store_memory(
    memory_id: str = Query(..., description="Memory ID to store under"),
    data: Dict[str, Any] = None,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Store data in memory"""
    if memory_id not in memory:
        memory[memory_id] = []

    # Add timestamp and user
    data["timestamp"] = datetime.utcnow().isoformat()
    data["user"] = current_user.username

    # Store data
    memory[memory_id].append(data)

    # Keep only the last 50 items
    if len(memory[memory_id]) > 50:
        memory[memory_id] = memory[memory_id][-50:]

    return {
        "status": "success",
        "message": f"Data stored in memory {memory_id}",
        "count": len(memory[memory_id]),
    }


@app.get("/mcp/memory/retrieve")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR, Role.VIEWER])
async def retrieve_memory(
    memory_id: str = Query(..., description="Memory ID to retrieve"),
    limit: int = Query(10, description="Maximum number of items to retrieve"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Retrieve data from memory"""
    if memory_id not in memory:
        return {
            "status": "success",
            "memory_id": memory_id,
            "count": 0,
            "data": [],
        }

    # Get the latest items up to the limit
    data = memory[memory_id][-limit:]

    return {
        "status": "success",
        "memory_id": memory_id,
        "count": len(data),
        "data": data,
    }


@app.delete("/mcp/memory/{memory_id}")
@has_role([Role.ADMIN])
async def clear_memory(
    memory_id: str, current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """Clear memory"""
    if memory_id in memory:
        del memory[memory_id]

    return {"status": "success", "message": f"Memory {memory_id} cleared"}


# =====================
# CTAS Integration Routes
# =====================


@app.post("/mcp/ctas/generate_uuid")
@has_role([Role.ADMIN, Role.ANALYST])
async def generate_uuid(
    type_id: str = Query(..., description="Type identifier"),
    namespace: str = Query("nyxtrace", description="Namespace"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Generate a UUID in CTAS format"""
    # In a real implementation, this would call the CTAS API
    # Here we'll generate a synthetic UUID for demonstration

    uuid_str = (
        f"UUID-{namespace.upper()}-{type_id.upper()}-{str(uuid.uuid4())[:8]}"
    )

    return {"status": "success", "uuid": uuid_str}


@app.post("/mcp/ctas/generate_cuid")
@has_role([Role.ADMIN, Role.ANALYST])
async def generate_cuid(
    data: str = Query(..., description="Data to hash"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Generate a CUID in CTAS format"""
    # In a real implementation, this would call the CTAS API
    # Here we'll generate a synthetic CUID for demonstration

    # Hash the data with a timestamp
    timestamp = datetime.utcnow().isoformat()
    hash_input = f"{data}|{timestamp}"
    hash_value = hashlib.sha256(hash_input.encode()).hexdigest()

    cuid_str = f"CUID-{hash_value[:16]}"

    return {"status": "success", "cuid": cuid_str}


@app.post("/mcp/ctas/generate_sch")
@has_role([Role.ADMIN, Role.ANALYST])
async def generate_sch(
    domain: str = Query(..., description="Domain identifier"),
    task: str = Query(..., description="Task identifier"),
    entropy: float = Query(0.5, ge=0.0, le=1.0, description="Entropy value"),
    transition: float = Query(
        0.5, ge=0.0, le=1.0, description="Transition readiness"
    ),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Generate a SCH in CTAS format"""
    # In a real implementation, this would call the CTAS API
    # Here we'll generate a synthetic SCH for demonstration

    # For demonstration, we'll make the SCH format reflect the inputs
    entropy_hex = hex(int(entropy * 255))[2:].zfill(2)
    transition_hex = hex(int(transition * 255))[2:].zfill(2)

    sch_str = f"SCH{domain}-{task}-{entropy_hex}{transition_hex}"

    return {
        "status": "success",
        "sch": sch_str,
        "entropy": entropy,
        "transition": transition,
    }


@app.post("/mcp/ctas/calculate_activation")
@has_role([Role.ADMIN, Role.ANALYST, Role.COLLECTOR])
async def calculate_activation(
    entropy: float = Query(..., ge=0.0, le=1.0, description="Entropy value"),
    transition: float = Query(
        ..., ge=0.0, le=1.0, description="Transition readiness"
    ),
    tools_available: bool = Query(
        True, description="Whether required tools are available"
    ),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Calculate activation status using CTAS activation function"""
    # Activation function: Φh(ζ,T,tools)=1 if ζ>0.5∧T>0.7∧tools available, 0 otherwise
    activated = entropy > 0.5 and transition > 0.7 and tools_available

    return {
        "status": "success",
        "activated": activated,
        "entropy": entropy,
        "transition": transition,
        "tools_available": tools_available,
        "activation_formula": "Φh(ζ,T,tools)=1 if ζ>0.5∧T>0.7∧tools available, 0 otherwise",
    }


# =====================
# Utility Routes
# =====================


@app.get("/mcp/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "6.5.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/mcp/stats")
@has_role([Role.ADMIN, Role.ANALYST])
async def get_stats(current_user: User = Depends(get_current_active_user)):
    """Get system statistics"""
    return {
        "status": "success",
        "stats": {
            "llm_configs": sum(len(models) for models in llm_configs.values()),
            "workflows": len(workflows),
            "workflow_executions": len(workflow_executions),
            "prompt_library": len(prompt_library),
            "tools": len(tools),
            "responses": len(responses),
            "memory_contexts": len(memory),
        },
    }


@app.get("/mcp", response_class=HTMLResponse)
async def landing_page():
    """HTML landing page for the MCP server"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NyxTrace MCP Server</title>
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #0066cc;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
            }
            h2 {
                color: #0066cc;
                margin-top: 30px;
            }
            a {
                color: #0066cc;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            .status {
                background-color: #f8f9fa;
                border-left: 4px solid #0066cc;
                padding: 15px;
                margin: 20px 0;
            }
            .endpoints {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-top: 20px;
            }
            .endpoint {
                margin-bottom: 10px;
            }
            .method {
                font-family: monospace;
                background-color: #e6f3ff;
                padding: 2px 6px;
                border-radius: 3px;
                margin-right: 5px;
            }
        </style>
    </head>
    <body>
        <h1>NyxTrace MCP Server</h1>
        
        <div class="status">
            <p><strong>Status:</strong> Running</p>
            <p><strong>Version:</strong> 6.5.0</p>
            <p><strong>Time:</strong> <span id="server-time"></span></p>
        </div>
        
        <h2>Documentation</h2>
        <p>
            Explore the API documentation:
            <ul>
                <li><a href="/mcp/docs" target="_blank">Interactive Swagger UI</a></li>
                <li><a href="/mcp/redoc" target="_blank">ReDoc Documentation</a></li>
                <li><a href="/mcp/openapi.json" target="_blank">OpenAPI Specification</a></li>
            </ul>
        </p>
        
        <h2>Key Endpoints</h2>
        <div class="endpoints">
            <div class="endpoint">
                <span class="method">POST</span>
                <a href="/mcp/llm/call">/mcp/llm/call</a> - Call an LLM
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <a href="/mcp/workflow/execute">/mcp/workflow/execute</a> - Execute a workflow
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <a href="/mcp/health">/mcp/health</a> - Health check
            </div>
        </div>
        
        <h2>CTAS Integration</h2>
        <p>
            The MCP server integrates with the CTAS framework, providing:
            <ul>
                <li>UUID, CUID, and SCH generation</li>
                <li>Activation function calculation</li>
                <li>Entropy and transition readiness evaluation</li>
            </ul>
        </p>
        
        <script>
            function updateServerTime() {
                document.getElementById('server-time').textContent = new Date().toISOString();
            }
            updateServerTime();
            setInterval(updateServerTime, 1000);
        </script>
    </body>
    </html>
    """


# =====================
# Server startup and shutdown
# =====================


@app.on_event("startup")
async def startup_event():
    """Initialize on server startup"""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Log startup
    logger.info("MCP Server starting up")

    # Add some example data
    _add_example_data()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on server shutdown"""
    logger.info("MCP Server shutting down")


def _add_example_data():
    """Add example data for demonstration"""
    # Add example LLM configs
    if LLMType.GPT not in llm_configs:
        llm_configs[LLMType.GPT] = {}

    llm_configs[LLMType.GPT]["gpt-4o"] = LLMConfig(
        type=LLMType.GPT,
        api_key="sk-example-key",
        endpoint="https://api.openai.com/v1/chat/completions",
        model="gpt-4o",
        version="2024-05-13",
        capabilities=[
            ModelCapability.TEXT,
            ModelCapability.IMAGE,
            ModelCapability.CODE,
            ModelCapability.MATH,
        ],
        size=ModelSize.XLARGE,
        context_window=128000,
        token_cost_input=0.01,
        token_cost_output=0.03,
    )

    if LLMType.ANTHROPIC not in llm_configs:
        llm_configs[LLMType.ANTHROPIC] = {}

    llm_configs[LLMType.ANTHROPIC]["claude-3-5-sonnet-20241022"] = LLMConfig(
        type=LLMType.ANTHROPIC,
        api_key="sk-example-key",
        endpoint="https://api.anthropic.com/v1/messages",
        model="claude-3-5-sonnet-20241022",
        version="2024-10-22",
        capabilities=[
            ModelCapability.TEXT,
            ModelCapability.IMAGE,
            ModelCapability.CODE,
        ],
        size=ModelSize.LARGE,
        context_window=200000,
        token_cost_input=0.008,
        token_cost_output=0.024,
    )

    # Add example prompt
    prompt_library["geospatial-analysis"] = PromptLibraryEntry(
        prompt_id="geospatial-analysis",
        name="Geospatial Intelligence Analysis",
        description="Analyzes geospatial data for intelligence purposes",
        template="""Analyze the following geospatial intelligence:

Location: {location}
Coordinates: {coordinates}
Time: {time}
Observed Activity: {activity}

Provide a comprehensive analysis including:
1. Significance of the location
2. Pattern analysis relative to historical data
3. Potential implications
4. Recommended actions

Additional context: {context}""",
        required_variables=["location", "coordinates", "time", "activity"],
        optional_variables={"context": "No additional context provided."},
        tags=["geospatial", "intelligence", "analysis"],
        version="1.0",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="system",
        target_models=[LLMType.GPT, LLMType.ANTHROPIC],
        example_variables={
            "location": "Port of Los Angeles",
            "coordinates": "33.7395, -118.2610",
            "time": "2025-05-10T14:30:00Z",
            "activity": "Unusual container movements",
            "context": "Recent cybersecurity alerts for maritime infrastructure.",
        },
    )

    # Add example tool
    tools["osint-lookup"] = Tool(
        tool_id="osint-lookup",
        name="OSINT Data Lookup",
        description="Looks up public information about entities",
        api_endpoint="/api/osint/lookup",
        method="POST",
        parameters={
            "entity": {"type": "string", "description": "Entity to look up"},
            "entity_type": {
                "type": "string",
                "description": "Type of entity",
                "enum": ["person", "organization", "domain", "ip"],
            },
            "depth": {
                "type": "integer",
                "description": "Search depth",
                "default": 1,
                "minimum": 1,
                "maximum": 3,
            },
        },
        required_parameters=["entity", "entity_type"],
        response_format=DataFormat.JSON,
        tags=["osint", "intelligence", "lookup"],
    )

    # Add example workflow
    steps = [
        WorkflowStep(
            step_id="collect-data",
            name="Collect OSINT Data",
            description="Collects data from OSINT sources",
            tool_id="osint-lookup",
            input_mapping={
                "entity": "target",
                "entity_type": "target_type",
                "depth": "search_depth",
            },
            output_mapping={"result": "osint_data"},
            timeout=30.0,
        ),
        WorkflowStep(
            step_id="analyze-data",
            name="Analyze OSINT Data",
            description="Analyzes collected OSINT data",
            llm_type=LLMType.GPT,
            prompt_id="geospatial-analysis",
            input_mapping={
                "location": "target_location",
                "coordinates": "target_coordinates",
                "time": "current_time",
                "activity": "recent_activity",
                "context": "osint_data",
            },
            output_mapping={"content": "analysis_result"},
            dependencies=["collect-data"],
        ),
    ]

    workflows["osint-analysis"] = Workflow(
        workflow_id="osint-analysis",
        name="OSINT Analysis Workflow",
        description="Collects and analyzes OSINT data",
        version="1.0",
        steps=steps,
        input_schema={
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "target_type": {"type": "string"},
                "target_location": {"type": "string"},
                "target_coordinates": {"type": "string"},
                "recent_activity": {"type": "string"},
                "search_depth": {"type": "integer", "default": 1},
            },
            "required": ["target", "target_type", "target_location"],
        },
        output_schema={
            "type": "object",
            "properties": {
                "osint_data": {"type": "object"},
                "analysis_result": {"type": "string"},
            },
        },
        tags=["osint", "analysis", "intelligence"],
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        created_by="system",
    )


# Start server if run directly
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5500)
