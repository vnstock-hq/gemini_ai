# gemini_ai

`gemini_ai` is a Python package that provides an interface to interact with Google’s Gemini AI model. This package enables advanced configuration of the generative model, file handling, media uploading, and response streaming, primarily optimized for Google Colab and Jupyter Notebook environments.

## Features

- **Flexible Model Configuration**: Customize model settings like temperature, top-p, and top-k sampling for generation.
- **File Upload and Media Support**: Supports uploading various file types (image, audio, text) to Google Colab and Gemini API.
- **Chat and Response Management**: Easily manage chat sessions with token counting, response streaming, and history display.
- **Environment-Specific Optimizations**: Automatic detection of Google Colab and Jupyter Notebook environments for optimized performance.

## Installation

To install `gemini_ai` and its dependencies, use:

```bash
pip install gemini_ai
```

This will install the required packages, including `google-generativeai`, `pillow`, `ipywidgets`, and `ipython`.

## Usage

### 1. Initializing GeminiAI

First, you need a Google Gemini API key. Set up your account and get an API key from [Google Cloud](https://cloud.google.com/).

```python
from gemini.gemini import GeminiAI

# Initialize GeminiAI with your API key
gemini = GeminiAI(api_key="YOUR_API_KEY")
```

### 2. Configuring the Model

You can configure the model with parameters such as `temperature`, `top_p`, `top_k`, and `max_output_tokens` for tailored response generation.

```python
gemini.config(temp=0.7, top_p=0.9, top_k=50, max_output_tokens=1024)
```

### 3. Starting a Chat Session

To start a chat session with the AI model, provide an initial instruction. If you’re working in Google Colab, you can also upload files as part of the chat context.

```python
gemini.start_chat(instruction="Tell me about the latest in AI technology.")
```

### 4. Sending Messages and Generating Content

Once a session is started, you can send prompts to the AI model and retrieve responses. The `send_message` function is useful for quick interactions, while `generate` can be used for more complex responses with optional streaming.

```python
# Send a simple message
gemini.send_message("What are the recent advancements in AI?")

# Generate a more elaborate response with optional streaming
gemini.generate(prompt="Can you write a story about space exploration?", stream=True)
```

### 5. Handling File Uploads (Google Colab Only)

In Google Colab, you can upload files directly to the Colab environment or to the Gemini API.

#### Uploading to Colab

```python
file_path = gemini.upload()  # Uploads a file in Colab and returns the file path
```

#### Uploading to Gemini API

```python
file_uri = gemini.upload_to_gemini(path=file_path, mime_type="text/plain")
print(f"File URI: {file_uri}")
```

### 6. Managing Chat History and Token Counts

You can display the chat history or count tokens in the chat session to manage usage effectively.

```python
# Display chat history
gemini.history()

# Count tokens in the chat history
gemini._token_counts()
```

## Environment-Specific Features

`GeminiAI` optimizes certain features based on the runtime environment. Here are some environment-specific details:

- **Google Colab**: Supports file uploads directly to Colab and uses `google.colab` utilities.
- **Jupyter Notebook**: Limits file upload functionality, skipping Colab-specific features gracefully.

## Class and Method Overview

### Class: `GeminiAI`

#### `__init__(api_key: str, gemini_model: str = 'gemini-1.5-flash-latest')`

Initializes the `GeminiAI` object with an API key and model name.

- **api_key** (str): Your API key for Gemini AI.
- **gemini_model** (str): Specifies the model version. Default is `'gemini-1.5-flash-latest'`.

#### `config(temp: Optional[int] = 1, top_p: Optional[float] = 0.95, top_k: Optional[int] = 64, max_output_tokens: Optional[int] = 8192, response_mime_type: str = "text/plain", stream: bool = True, silent: bool = True)`

Configures the model settings with adjustable parameters.

#### `start_chat(instruction: [str], file_path: Optional[str] = None, meme_type: Optional[str]="text/plain")`

Starts a new chat session with the AI, with optional file input.

#### `send_message(prompt: str, stream: bool = False)`

Sends a text prompt to the AI and retrieves a response, with optional streaming.

#### `generate(prompt: str, stream: bool = True, chunk_size: int = 80)`

Generates content from a prompt, with support for chunked streaming.

#### `upload() -> str`

Uploads a file in Google Colab and returns the file path. Raises an error if not in Colab.

#### `upload_to_gemini(path, mime_type=None)`

Uploads the specified file directly to the Gemini API.

#### `history()`

Displays the chat session history.

#### `_token_counts()`

Counts tokens in the entire chat session history for API usage management.

## MIME Types Supported

This package supports various MIME types for file uploads:

- **Image**: `image/jpeg`, `image/png`, `image/gif`, `image/webp`, `image/heic`, `image/heif`
- **Audio**: `audio/wav`, `audio/mp3`, `audio/aiff`, `audio/aac`, `audio/ogg`, `audio/flac`
- **Text**: `text/plain`, `text/html`, `text/css`, `text/javascript`, `application/json`, `text/markdown`

## Running Tests

To test the `gemini_ai` package, use `pytest` with coverage:

```bash
python3.10 -m pytest --cov=gemini --cov-report=term-missing
```

## Example Code

Here’s a complete example demonstrating the initialization, configuration, chat session setup, and file upload:

```python
from gemini.gemini import GeminiAI

# Initialize the AI with your API key
gemini = GeminiAI(api_key="YOUR_API_KEY")

# Configure model settings
gemini.config(temp=0.7, top_p=0.9)

# Start a chat session
gemini.start_chat(instruction="Tell me about recent advancements in AI")

# Send a prompt and generate a response
gemini.send_message("What's the future of AI?")
gemini.generate("Can you explain the role of AI in healthcare?")

# Display chat history
gemini.history()

# Upload a file to Gemini (Colab only)
file_uri = gemini.upload_to_gemini("/path/to/your/file.txt")
print(f"File uploaded to Gemini with URI: {file_uri}")
```

## Contribution

Contributions to `gemini_ai` are welcome! Please feel free to open issues or submit pull requests on the [GitHub repository](https://github.com/vnstock-hq/gemini_ai).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.