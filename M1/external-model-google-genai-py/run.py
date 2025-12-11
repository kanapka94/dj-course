from google import genai
from google.genai import types
import os 
import tiktoken

from dotenv import load_dotenv
load_dotenv()

# if set, print first 4 chars and last 4 chars and dots inside, else print NOT SET
print(f"env var \"GEMINI_API_KEY\" is: { os.getenv('GEMINI_API_KEY', '')[:4] + '...' + os.getenv('GEMINI_API_KEY', '')[-4:] if len(os.getenv('GEMINI_API_KEY', '')) > 0 else 'NOT SET' }")
if not os.getenv('GEMINI_API_KEY'):
    raise ValueError("GEMINI_API_KEY environment variable is not set. Please set it to your Google Gemini API key.")

client = genai.Client()

model = "gemini-2.5-flash"
# model = "gemini-1.5-pro"

system_role = "you were Gandalf the Grey in the Lord of the Rings. You answer in max 15 words. Your answers are mysterious and magical."

conversation_history = [
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="What is the best time for coffee?")]
    ),
    types.Content(
        role="model",
        parts=[types.Part.from_text(text="The best time for coffee is in the morning my apprentice.")]
    ),
    types.Content(
        role="user",
        parts=[types.Part.from_text(text="How about tea?")]
    ),
]

# Initialize tiktoken encoder
encoding = tiktoken.get_encoding("cl100k_base")

# Display tokens for each conversation
for i, content in enumerate(conversation_history):
    print(f"\n--- Conversation {i+1} ({content.role}) ---")
    # Extract text from all parts
    text_parts = []
    for part in content.parts:
        if hasattr(part, 'text') and part.text:
            text_parts.append(part.text)
    
    if text_parts:
        full_text = " ".join(text_parts)
        tokens = encoding.encode(full_text)
        print(f"Text: {full_text}")
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")

response = client.models.generate_content(
    model=model,
    contents=conversation_history,
    config=types.GenerateContentConfig(
        system_instruction=system_role,
        thinking_config=types.ThinkingConfig(thinking_budget=0) # Disables thinking
    ),
)

tokens = encoding.encode(response.text)
print("Response:", response.text)
print("Response tokens:", tokens)
print("Response token count:", len(tokens))