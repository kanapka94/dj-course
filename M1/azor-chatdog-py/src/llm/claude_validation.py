from pydantic import BaseModel, Field, validator
from typing import Optional, Literal

class ClaudeConfig(BaseModel):
    engine: Literal["CLAUDE"] = Field(default="CLAUDE")
    model_name: str = Field(default="claude-sonnet-4-5", description="Nazwa modelu Claude")
    claude_api_key: str = Field(..., min_length=1, description="Klucz API Anthropic Claude")
    
    @validator('claude_api_key')
    def validate_api_key(cls, v):
        if not v or v.strip() == "":
            raise ValueError("CLAUDE_API_KEY nie może być pusty")
        return v.strip()
