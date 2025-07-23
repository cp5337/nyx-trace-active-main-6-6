"""
Model Context Protocol Client
----------------------------
Client for interacting with the Model Context Protocol (MCP) server.
Provides a unified interface for accessing different language models
and orchestrating workflows across the NyxTrace system.

This implementation features advanced caching, retry mechanisms, and
contextual awareness for optimal performance with the CTAS framework.
"""

import logging
import os
import json
import asyncio
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from datetime import datetime
from enum import Enum
from functools import lru_cache

import httpx
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# Define LLM types supported by the MCP system
class LLMType(str, Enum):
    """LLM types supported by the MCP system"""

    GPT = "gpt"
    GROK = "grok"
    GEMINI = "gemini"
    LOCAL_LLM1 = "local_llm1"
    LOCAL_LLM2 = "local_llm2"
    NEURAL_NET = "neural_net"
    ANTHROPIC = "anthropic"
    WOLFRAM = "wolfram"


class PromptTemplate(BaseModel):
    """
    Structured prompt template with variable substitution
    and contextual awareness
    """

    template: str = Field(
        ..., description="Prompt template with {variable} placeholders"
    )
    required_variables: List[str] = Field(
        default_factory=list, description="Required variables for this template"
    )
    optional_variables: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional variables with default values",
    )

    def format(self, **kwargs) -> str:
        """
        Format the template with provided variables

        Args:
            **kwargs: Variables to substitute in the template

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If a required variable is missing
        """
        # Check for required variables
        missing = [v for v in self.required_variables if v not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        # Add optional variables with defaults if not provided
        variables = self.optional_variables.copy()
        variables.update(kwargs)

        # Format the template
        return self.template.format(**variables)


class MCPResponse(BaseModel):
    """Response from an LLM processed by the MCP server"""

    response_id: str = Field(..., description="Unique ID for this response")
    llm_type: LLMType = Field(
        ..., description="Type of LLM that generated this response"
    )
    content: str = Field(..., description="Response content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional response metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )

    # Advanced fields for CTAS integration
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

    class Config:
        arbitrary_types_allowed = True


class MCPClient:
    """
    Client for interacting with the Model Context Protocol (MCP) server.

    Features:
    - Unified interface for multiple LLM types
    - Prompt templating and variable substitution
    - Response caching and deduplication
    - Retry mechanisms with exponential backoff
    - CTAS integration with UUID, CUID, and SCH management
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_keys: Optional[Dict[LLMType, str]] = None,
        cache_size: int = 1000,
        retries: int = 3,
        timeout: float = 30.0,
    ):
        """
        Initialize the MCP client

        Args:
            base_url: Base URL for the MCP server
            api_keys: Dictionary mapping LLM types to API keys
            cache_size: Maximum number of responses to cache
            retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_keys = api_keys or {}
        self.cache_size = cache_size
        self.retries = retries
        self.timeout = timeout

        # Response cache
        self._response_cache: Dict[str, MCPResponse] = {}

        # Frequently used prompt templates
        self._prompt_templates: Dict[str, PromptTemplate] = {}

        # Load any API keys from environment
        self._load_api_keys_from_env()

        logger.info("Initialized MCP client")

    def _load_api_keys_from_env(self) -> None:
        """Load API keys from environment variables"""
        env_mapping = {
            LLMType.GPT: "OPENAI_API_KEY",
            LLMType.GROK: "GROK_API_KEY",
            LLMType.GEMINI: "GEMINI_API_KEY",
            LLMType.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMType.WOLFRAM: "WOLFRAM_APPID",
            LLMType.LOCAL_LLM1: "LOCAL_LLM1_API_KEY",
            LLMType.LOCAL_LLM2: "LOCAL_LLM2_API_KEY",
            LLMType.NEURAL_NET: "NEURAL_NET_API_KEY",
        }

        for llm_type, env_var in env_mapping.items():
            if env_var in os.environ and llm_type not in self.api_keys:
                self.api_keys[llm_type] = os.environ[env_var]

    async def call_llm(
        self,
        llm_type: LLMType,
        prompt: str,
        variables: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None,
        use_cache: bool = True,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> MCPResponse:
        """
        Call an LLM through the MCP server

        Args:
            llm_type: Type of LLM to use
            prompt: Direct prompt or template ID
            variables: Variables for prompt template
            template_id: ID of a registered prompt template
            use_cache: Whether to use the response cache
            model: Specific model identifier to use
            temperature: Temperature parameter (0-1)
            max_tokens: Maximum tokens to generate

        Returns:
            LLM response
        """
        # Format prompt if template and variables provided
        formatted_prompt = prompt

        if template_id and template_id in self._prompt_templates:
            if variables is None:
                variables = {}

            template = self._prompt_templates[template_id]
            formatted_prompt = template.format(**variables)
        elif variables:
            # Simple variable substitution if not using a registered template
            formatted_prompt = prompt.format(**variables)

        # Generate cache key
        if use_cache:
            cache_key = self._generate_cache_key(
                llm_type, formatted_prompt, model, temperature, max_tokens
            )

            # Check cache
            if cache_key in self._response_cache:
                logger.debug(f"Cache hit for key {cache_key}")
                return self._response_cache[cache_key]

        # Prepare request data based on LLM type
        headers, data, url = self._prepare_request(
            llm_type, formatted_prompt, model, temperature, max_tokens
        )

        # Make request with retries
        response_data = await self._make_request_with_retries(
            url, headers, data, llm_type
        )

        # Parse response
        content = self._extract_content(llm_type, response_data)

        # Create response object
        response = MCPResponse(
            response_id=str(uuid.uuid4()),
            llm_type=llm_type,
            content=content,
            metadata=response_data,
            timestamp=datetime.utcnow(),
            # Calculate additional CTAS parameters
            entropy=self._calculate_entropy(content),
            transition_readiness=self._calculate_transition_readiness(content),
        )

        # Update cache if enabled
        if use_cache:
            # Maintain cache size
            if len(self._response_cache) >= self.cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self._response_cache))
                del self._response_cache[oldest_key]

            self._response_cache[cache_key] = response

        return response

    def register_prompt_template(
        self,
        template_id: str,
        template: Union[PromptTemplate, str],
        required_variables: Optional[List[str]] = None,
        optional_variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a prompt template for reuse

        Args:
            template_id: Identifier for the template
            template: Template string or PromptTemplate object
            required_variables: Required variables if providing a string
            optional_variables: Optional variables with defaults if providing a string
        """
        if isinstance(template, str):
            # Convert string to PromptTemplate
            self._prompt_templates[template_id] = PromptTemplate(
                template=template,
                required_variables=required_variables or [],
                optional_variables=optional_variables or {},
            )
        else:
            # Use provided PromptTemplate
            self._prompt_templates[template_id] = template

        logger.info(f"Registered prompt template: {template_id}")

    async def execute_workflow(
        self, workflow_id: str, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a workflow on the MCP server

        Args:
            workflow_id: ID of the workflow to execute
            inputs: Input variables for the workflow

        Returns:
            Workflow execution results
        """
        url = f"{self.base_url}/workflow/execute"

        data = {"workflow_id": workflow_id, "inputs": inputs}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, json=data)

            if response.status_code != 200:
                logger.error(f"Workflow execution failed: {response.text}")
                raise Exception(f"Workflow execution failed: {response.text}")

            return response.json()

    async def get_response(self, response_id: str) -> MCPResponse:
        """
        Get a response by ID from the MCP server

        Args:
            response_id: Response ID

        Returns:
            Response data
        """
        url = f"{self.base_url}/response/{response_id}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)

            if response.status_code != 200:
                logger.error(f"Failed to get response: {response.text}")
                raise Exception(f"Failed to get response: {response.text}")

            data = response.json()
            return MCPResponse(**data)

    def _generate_cache_key(
        self,
        llm_type: LLMType,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a cache key for an LLM request"""
        # Create a string representation of the request
        key_parts = [
            str(llm_type),
            prompt,
            str(model),
            str(temperature),
            str(max_tokens),
        ]

        key_str = "|".join(key_parts)

        # Hash for shorter key
        return hashlib.md5(key_str.encode()).hexdigest()

    def _prepare_request(
        self,
        llm_type: LLMType,
        prompt: str,
        model: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> Tuple[Dict[str, str], Dict[str, Any], str]:
        """Prepare request headers, data and URL based on LLM type"""
        api_key = self.api_keys.get(llm_type)
        if not api_key:
            raise ValueError(f"No API key configured for {llm_type}")

        headers = {}
        data = {}
        url = ""

        if llm_type == LLMType.GPT:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": model or "gpt-4o",  # Default to gpt-4o
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            url = "https://api.openai.com/v1/chat/completions"

        elif llm_type == LLMType.ANTHROPIC:
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            data = {
                "model": model
                or "claude-3-5-sonnet-20241022",  # Default to claude-3-5-sonnet-20241022
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            url = "https://api.anthropic.com/v1/messages"

        elif llm_type == LLMType.GEMINI:
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                },
            }
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            url += f"?key={api_key}"

        elif llm_type == LLMType.GROK:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
            data = {
                "model": model or "grok-2-1212",  # Default to grok-2-1212
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            url = "https://api.x.ai/v1/chat/completions"

        elif llm_type == LLMType.WOLFRAM:
            from urllib.parse import quote

            url = f"https://api.wolframalpha.com/v1/result?i={quote(prompt)}&appid={api_key}"

        elif llm_type == LLMType.LOCAL_LLM1:
            url = os.getenv(
                "LOCAL_LLM1_ENDPOINT", "http://localhost:8001/generate"
            )
            data = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        elif llm_type == LLMType.LOCAL_LLM2:
            url = os.getenv(
                "LOCAL_LLM2_ENDPOINT", "http://localhost:8002/generate"
            )
            data = {
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        elif llm_type == LLMType.NEURAL_NET:
            url = os.getenv(
                "NEURAL_NET_ENDPOINT", "http://localhost:8003/predict"
            )
            data = {"input": prompt, "parameters": {"temperature": temperature}}

        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")

        return headers, data, url

    async def _make_request_with_retries(
        self,
        url: str,
        headers: Dict[str, str],
        data: Dict[str, Any],
        llm_type: LLMType,
    ) -> Dict[str, Any]:
        """Make a request with exponential backoff retries"""
        retry_count = 0
        last_exception = None

        while retry_count <= self.retries:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if llm_type == LLMType.WOLFRAM:
                        # Wolfram uses GET requests
                        response = await client.get(url, headers=headers)
                    else:
                        # Other LLMs use POST requests
                        response = await client.post(
                            url, headers=headers, json=data
                        )

                    if response.status_code != 200:
                        logger.warning(
                            f"Request failed with status {response.status_code}: {response.text}"
                        )
                        raise Exception(
                            f"Request failed with status {response.status_code}: {response.text}"
                        )

                    if llm_type == LLMType.WOLFRAM:
                        # Wolfram returns plain text
                        return {"result": response.text}
                    else:
                        # Other LLMs return JSON
                        return response.json()

            except Exception as e:
                last_exception = e
                retry_count += 1

                if retry_count <= self.retries:
                    # Exponential backoff with jitter
                    import random

                    delay = (2**retry_count) + random.uniform(0, 1)
                    logger.warning(
                        f"Request failed, retrying in {delay:.2f} seconds: {str(e)}"
                    )
                    await asyncio.sleep(delay)

        # All retries failed
        logger.error(f"All retries failed: {str(last_exception)}")
        raise last_exception

    def _extract_content(
        self, llm_type: LLMType, response_data: Dict[str, Any]
    ) -> str:
        """Extract content from response based on LLM type"""
        if llm_type == LLMType.GPT:
            return str(response_data["choices"][0]["message"]["content"])

        elif llm_type == LLMType.ANTHROPIC:
            if "content" in response_data and isinstance(
                response_data["content"], list
            ):
                return str(response_data["content"][0].get("text", ""))
            return ""

        elif llm_type == LLMType.GEMINI:
            if (
                "candidates" in response_data
                and response_data["candidates"]
                and "content" in response_data["candidates"][0]
                and "parts" in response_data["candidates"][0]["content"]
                and response_data["candidates"][0]["content"]["parts"]
            ):
                return str(
                    response_data["candidates"][0]["content"]["parts"][0][
                        "text"
                    ]
                )
            return ""

        elif llm_type == LLMType.GROK:
            return str(response_data["choices"][0]["message"]["content"])

        elif llm_type == LLMType.WOLFRAM:
            return str(response_data["result"])

        elif llm_type in (
            LLMType.LOCAL_LLM1,
            LLMType.LOCAL_LLM2,
            LLMType.NEURAL_NET,
        ):
            return str(response_data.get("result", ""))

        return ""

    def _calculate_entropy(self, content: str) -> float:
        """
        Calculate entropy (ζ) value for a response
        This is a simplified calculation that could be expanded
        """
        # Simple entropy calculation based on content length, diversity, etc.
        if not content:
            return 0.0

        # Factors that increase entropy:
        # - Content length (up to a point)
        # - Character diversity
        # - Presence of structured data (e.g., JSON, code)

        length_factor = min(
            len(content) / 1000, 1.0
        )  # Cap at 1.0 for 1000+ chars

        # Character diversity
        unique_chars = len(set(content))
        total_chars = len(content)
        diversity = unique_chars / min(total_chars, 100)  # Cap denominator

        # Check for structured data
        structured_bonus = 0.0
        if "{" in content and "}" in content:  # Potential JSON
            structured_bonus += 0.1
        if "<" in content and ">" in content:  # Potential XML/HTML
            structured_bonus += 0.1
        if "def " in content or "function" in content:  # Potential code
            structured_bonus += 0.1

        # Combine factors
        entropy = (
            0.3 + (0.4 * length_factor) + (0.2 * diversity) + structured_bonus
        )

        # Ensure in range [0, 1]
        return max(0.0, min(1.0, entropy))

    def _calculate_transition_readiness(self, content: str) -> float:
        """
        Calculate transition readiness (T) value for a response
        This evaluates how ready the content is to trigger a transition
        """
        # Simple calculation based on content characteristics
        if not content:
            return 0.0

        # Factors that indicate transition readiness:
        # - Presence of action words
        # - Decisiveness (e.g., confidence markers)
        # - Completeness

        content_lower = content.lower()

        # Action words
        action_words = [
            "execute",
            "implement",
            "start",
            "begin",
            "launch",
            "deploy",
            "activate",
            "proceed",
            "initiate",
            "run",
        ]
        action_count = sum(1 for word in action_words if word in content_lower)
        action_factor = min(
            action_count / 3, 1.0
        )  # Cap at 1.0 for 3+ action words

        # Decisiveness markers
        decisive_phrases = [
            "therefore",
            "consequently",
            "as a result",
            "conclude",
            "in conclusion",
            "recommend",
            "should",
            "must",
            "will",
        ]
        decisive_count = sum(
            1 for phrase in decisive_phrases if phrase in content_lower
        )
        decisive_factor = min(
            decisive_count / 2, 1.0
        )  # Cap at 1.0 for 2+ decisive phrases

        # Completeness (heuristic: longer responses are usually more complete)
        completeness = min(len(content) / 500, 1.0)  # Cap at 1.0 for 500+ chars

        # Combine factors
        readiness = (
            0.3
            + (0.3 * action_factor)
            + (0.2 * decisive_factor)
            + (0.2 * completeness)
        )

        # Ensure in range [0, 1]
        return max(0.0, min(1.0, readiness))


# Singleton instance for global use
mcp_client = MCPClient()


async def test_mcp_client():
    """Test the MCP client with a simple request"""
    client = MCPClient()

    # Register a template
    client.register_prompt_template(
        "task_analysis",
        "Analyze the following task from a CTAS perspective:\n\nTask: {task}\n\nProvide analysis of entropy and transition readiness.",
        required_variables=["task"],
    )

    # Make a request using the template
    response = await client.call_llm(
        LLMType.GPT,
        prompt="",  # Empty because we're using a template
        template_id="task_analysis",
        variables={"task": "Reconnaissance of a financial target"},
    )

    print(f"Response: {response.content}")
    print(f"Entropy: {response.entropy}")
    print(f"Transition Readiness: {response.transition_readiness}")


if __name__ == "__main__":
    # Example usage
    asyncio.run(test_mcp_client())
