"""
Claude Code API LLM Client Implementation
Encapsulates all Claude Code API interactions using claude-agent-sdk.
"""

import os
import asyncio
from typing import Optional, List, Any, Dict
from dotenv import load_dotenv
from cli import console
from .claude_validation import ClaudeConfig

try:
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock, ResultMessage
    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    console.print_error("claude-agent-sdk nie jest zainstalowany. Zainstaluj: pip install claude-agent-sdk")


class ClaudeChatSessionWrapper:
    """
    Wrapper for Claude chat session that provides universal dictionary-based history format.
    This ensures compatibility with other LLM clients' history format.
    """
    
    def __init__(self, claude_client: 'ClaudeSDKClient', system_instruction: str, history: Optional[List[Dict]] = None):
        """
        Initialize wrapper with Claude SDK client.
        
        Args:
            claude_client: The ClaudeSDKClient instance
            system_instruction: System prompt for the assistant
            history: Previous conversation history (optional)
        """
        self.claude_client = claude_client
        self.system_instruction = system_instruction
        self._history = history or []
        self._session_connected = False
    
    async def _ensure_connected(self):
        """Ensure Claude client is connected."""
        if not self._session_connected:
            # Connect with system instruction if provided
            if self.system_instruction:
                # Convert history to initial prompt if needed
                initial_prompt = self._build_initial_prompt()
                await self.claude_client.connect(initial_prompt)
            else:
                await self.claude_client.connect()
            self._session_connected = True
    
    def _build_initial_prompt(self) -> str:
        """Build initial prompt from system instruction and history."""
        parts = []
        if self.system_instruction:
            parts.append(f"System: {self.system_instruction}")
        
        # Add conversation history
        for message in self._history:
            role = message.get("role", "")
            text = ""
            if "parts" in message and message["parts"]:
                text = message["parts"][0].get("text", "")
            
            if role == "user" and text:
                parts.append(f"User: {text}")
            elif role == "model" and text:
                parts.append(f"Assistant: {text}")
        
        return "\n\n".join(parts) if parts else self.system_instruction
    
    def send_message(self, text: str) -> Any:
        """
        Sends a message to Claude and returns a response object.
        Note: This is a synchronous wrapper around async operations.
        
        Args:
            text: User's message
            
        Returns:
            Response object with .text attribute containing the response
        """
        # Add user message to history
        user_message = {"role": "user", "parts": [{"text": text}]}
        self._history.append(user_message)
        
        try:
            # Run async operation in event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop exists, create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if loop.is_running():
                # If loop is already running, we need to use a different approach
                # Create a new event loop in a thread
                import concurrent.futures
                import threading
                
                def run_in_thread():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    try:
                        return new_loop.run_until_complete(self._send_message_async(text))
                    finally:
                        new_loop.close()
                
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(run_in_thread)
                    response_text = future.result()
            else:
                response_text = loop.run_until_complete(self._send_message_async(text))
            
            # Add assistant response to history
            assistant_message = {"role": "model", "parts": [{"text": response_text}]}
            self._history.append(assistant_message)
            
            # Return response object compatible with other clients
            return ClaudeResponse(response_text)
            
        except Exception as e:
            console.print_error(f"Błąd podczas generowania odpowiedzi Claude: {e}")
            # Return error response
            error_text = "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."
            assistant_message = {"role": "model", "parts": [{"text": error_text}]}
            self._history.append(assistant_message)
            return ClaudeResponse(error_text)
    
    async def _send_message_async(self, text: str) -> str:
        """Async implementation of send_message."""
        await self._ensure_connected()
        
        # Send query
        await self.claude_client.query(text)
        
        # Collect response
        response_parts = []
        async for message in self.claude_client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        response_parts.append(block.text)
            elif isinstance(message, ResultMessage):
                # End of response
                break
        
        return "\n".join(response_parts) if response_parts else ""
    
    def get_history(self) -> List[Dict]:
        """
        Gets conversation history in universal dictionary format.
        
        Returns:
            List of dictionaries with format: {"role": "user|model", "parts": [{"text": "..."}]}
        """
        return self._history
    
    async def close(self):
        """Close the Claude client connection."""
        if self._session_connected:
            await self.claude_client.disconnect()
            self._session_connected = False


class ClaudeResponse:
    """
    Response object that mimics the Gemini/LLaMA response interface.
    Provides a .text attribute containing the response text.
    """
    
    def __init__(self, text: str):
        self.text = text


class ClaudeLLMClient:
    """
    Encapsulates all Claude Code API interactions.
    Provides a clean interface compatible with GeminiLLMClient and LlamaClient.
    """
    
    def __init__(self, model_name: str, api_key: str, allowed_tools: Optional[List[str]] = None):
        """
        Initialize the Claude LLM client with explicit parameters.
        
        Args:
            model_name: Model to use (e.g., 'claude-sonnet-4-5')
            api_key: Anthropic Claude API key
            allowed_tools: List of allowed tools for Claude (optional)
        
        Raises:
            ValueError: If api_key is empty or None
            RuntimeError: If claude-agent-sdk is not installed
        """
        if not CLAUDE_SDK_AVAILABLE:
            raise RuntimeError("claude-agent-sdk nie jest zainstalowany. Zainstaluj: pip install claude-agent-sdk")
        
        if not api_key:
            raise ValueError("API key cannot be empty or None")
        
        self.model_name = model_name
        self.api_key = api_key
        self.allowed_tools = allowed_tools or []
        
        # Set API key in environment for SDK
        os.environ['ANTHROPIC_API_KEY'] = api_key
    
    @staticmethod
    def preparing_for_use_message() -> str:
        """
        Returns a message indicating that Claude client is being prepared.
        
        Returns:
            Formatted preparation message string
        """
        return "🤖 Przygotowywanie klienta Claude Code API..."
    
    @classmethod
    def from_environment(cls) -> 'ClaudeLLMClient':
        """
        Factory method that creates a ClaudeLLMClient instance from environment variables.
        
        Returns:
            ClaudeLLMClient instance initialized with environment variables
            
        Raises:
            ValueError: If required environment variables are not set
        """
        load_dotenv()
    
        # Walidacja z Pydantic
        config = ClaudeConfig(
            model_name=os.getenv('MODEL_NAME', 'claude-sonnet-4-5'),
            claude_api_key=os.getenv('CLAUDE_API_KEY', '')
        )
        
        # Parse allowed tools from environment (comma-separated)
        allowed_tools_str = os.getenv('CLAUDE_ALLOWED_TOOLS', '')
        allowed_tools = [tool.strip() for tool in allowed_tools_str.split(',') if tool.strip()] if allowed_tools_str else []
        
        return cls(
            model_name=config.model_name,
            api_key=config.claude_api_key,
            allowed_tools=allowed_tools
        )
    
    def create_chat_session(self, 
                          system_instruction: str, 
                          history: Optional[List[Dict]] = None,
                          thinking_budget: int = 0) -> ClaudeChatSessionWrapper:
        """
        Creates a new chat session with the specified configuration.
        
        Args:
            system_instruction: System role/prompt for the assistant
            history: Previous conversation history (optional, in universal dict format)
            thinking_budget: Thinking budget for the model (ignored for Claude Code API)
            
        Returns:
            ClaudeChatSessionWrapper with universal dictionary-based interface
        """
        # Create Claude Agent Options
        options = ClaudeAgentOptions(
            model=self.model_name,
            system_prompt=system_instruction,
            allowed_tools=self.allowed_tools if self.allowed_tools else None
        )
        
        # Create Claude SDK Client
        claude_client = ClaudeSDKClient(options=options)
        
        return ClaudeChatSessionWrapper(
            claude_client=claude_client,
            system_instruction=system_instruction,
            history=history or []
        )
    
    def count_history_tokens(self, history: List[Dict]) -> int:
        """
        Counts tokens for the given conversation history.
        Note: This is an approximation since Claude Code API doesn't provide 
        direct token counting for conversations.
        
        Args:
            history: Conversation history in universal dict format
            
        Returns:
            Estimated token count
        """
        if not history:
            return 0
        
        try:
            # Build text from history
            text_parts = []
            for message in history:
                if "parts" in message and message["parts"]:
                    text_parts.append(message["parts"][0].get("text", ""))
            
            full_text = " ".join(text_parts)
            
            # Rough estimation: ~4 characters per token average
            return len(full_text) // 4
            
        except Exception as e:
            console.print_error(f"Błąd podczas liczenia tokenów: {e}")
            # Fallback: rough estimation
            total_chars = sum(len(msg["parts"][0].get("text", "")) for msg in history if "parts" in msg and msg["parts"])
            return total_chars // 4
    
    def get_model_name(self) -> str:
        """Returns the currently configured model name."""
        return self.model_name
    
    def is_available(self) -> bool:
        """
        Checks if the LLM service is available and properly configured.
        
        Returns:
            True if client is properly initialized and has API key
        """
        return CLAUDE_SDK_AVAILABLE and bool(self.api_key)
    
    def ready_for_use_message(self) -> str:
        """
        Returns a ready-to-use message with model info and masked API key.
        
        Returns:
            Formatted message string for display
        """
        # Mask API key - show first 4 and last 4 characters
        if len(self.api_key) <= 8:
            masked_key = "****"
        else:
            masked_key = f"{self.api_key[:4]}...{self.api_key[-4:]}"
        
        tools_info = f", Narzędzia: {', '.join(self.allowed_tools)}" if self.allowed_tools else ""
        
        return f"✅ Klient Claude Code API gotowy do użycia (Model: {self.model_name}, Key: {masked_key}{tools_info})"
    
    @property
    def client(self):
        """
        Provides access to the underlying Claude SDK client for backwards compatibility.
        This property should be used sparingly and eventually removed.
        """
        # Return a new client instance if needed
        options = ClaudeAgentOptions(
            model=self.model_name,
            allowed_tools=self.allowed_tools if self.allowed_tools else None
        )
        return ClaudeSDKClient(options=options)
