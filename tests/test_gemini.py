# Required: pytest, pytest-cov, yaml

import os
import pytest
from gemini_ai.gemini import GeminiAI
from gemini_ai.utils import colab_verify, jupyter_verify
from unittest.mock import MagicMock
from PIL import Image as PILImage
# read GEMINI_TK key from creds.yaml
import yaml
with open("/Users/mrthinh/Library/CloudStorage/OneDrive-Personal/Github/gemini-ai/tests/creds.yaml", 'r') as stream:
    try:
        creds = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Setup: Mock API key for testing purposes
API_KEY = creds['GEMINI_TK']

# Check if we are running on Google Colab or Jupyter Notebook
IS_COLAB = colab_verify()
IS_JUPYTER = jupyter_verify()

# Initialize the GeminiAI instance for testing
@pytest.fixture
def gemini_instance():
    return GeminiAI(api_key=API_KEY)

### Test Environment Verification ###

def test_colab_verify():
    """Test Google Colab environment verification"""
    if IS_COLAB:
        assert colab_verify() is True
    else:
        assert colab_verify() is False

def test_jupyter_verify():
    """Test Jupyter Notebook environment verification"""
    if IS_JUPYTER:
        assert jupyter_verify() is True
    else:
        assert jupyter_verify() is False

### Test Initialization ###

def test_initialization(gemini_instance):
    """Test GeminiAI initialization with an API key"""
    assert gemini_instance.api_key == API_KEY
    assert gemini_instance.gemini_model == 'gemini-1.5-flash-latest'
    assert gemini_instance.is_colab == IS_COLAB
    assert gemini_instance.is_jupyter == IS_JUPYTER

### Test Configuration ###

def test_config(gemini_instance):
    """Test configuration of the generative model"""
    chat_session = gemini_instance.config(temp=0.7, top_p=0.8, top_k=50, silent=False)
    assert chat_session is not None
    assert gemini_instance.model._generation_config['temperature'] == 0.7
    assert gemini_instance.model._generation_config['top_p'] == 0.8
    assert gemini_instance.model._generation_config['top_k'] == 50

### Test Google Colab-Specific Functions ###

@pytest.mark.skipif(not IS_COLAB, reason="Test only applicable in Google Colab")
def test_upload_colab(gemini_instance):
    """Test file upload functionality in Google Colab"""
    files_mock = MagicMock()
    files_mock.upload.return_value = {'test_image.jpg': b'Test Image Data'}
    file_path = gemini_instance.upload()
    assert os.path.basename(file_path) == 'test_image.jpg'
    assert os.path.exists(file_path)

@pytest.mark.skipif(not IS_COLAB, reason="Test only applicable in Google Colab")
def test_dir_cleanup(gemini_instance):
    """Test directory cleanup in Google Colab"""
    # Create test files in Colab environment
    open('/content/test_file_1.txt', 'a').close()
    open('/content/test_file_2.txt', 'a').close()
    gemini_instance._dir_cleanup()
    assert not os.path.exists('/content/test_file_1.txt')
    assert not os.path.exists('/content/test_file_2.txt')

### Test General Functions ###

def test_open_image(gemini_instance):
    """Test opening and displaying an image"""
    # Mock an image for testing purposes
    PILImage.open = MagicMock()
    image_path = "/path/to/test_image.jpg"
    img = gemini_instance.open_image(image_path)
    assert img is not None
    PILImage.open.assert_called_with(image_path)

def test_start_chat(gemini_instance):
    """Test starting a chat session with the AI"""
    gemini_instance.start_chat(instruction="Hello, AI!")
    assert gemini_instance.chat_session is not None
    assert len(gemini_instance.chat_session.history) == 1

def test_send_message(gemini_instance):
    """Test sending a message with mock response"""
    gemini_instance.start_chat(instruction="Start chat session")
    mock_response = MagicMock()
    mock_response.text = "Sample Response"
    gemini_instance.chat_session.send_message = MagicMock(return_value=mock_response)
    gemini_instance.send_message(prompt="How are you?")
    gemini_instance.chat_session.send_message.assert_called_once_with("How are you?")

def test_generate(gemini_instance):
    """Test generating content with AI model with full mock response."""
    gemini_instance.start_chat(instruction="Start generation")
    
    # Mock response with required attributes
    mock_response = MagicMock()
    mock_response.block_reason = None
    mock_response.prompt_feedback = "Sample feedback"
    
    # Mock the generate_content function to return the mocked response
    gemini_instance.model.generate_content = MagicMock(return_value=mock_response)
    gemini_instance.generate(prompt="Tell me a story")
    
    # Check if generate_content was called with the correct prompt
    gemini_instance.model.generate_content.assert_called_once_with("Tell me a story", stream=True)

def test_token_counts(gemini_instance):
    """Test token count in chat history"""
    gemini_instance.start_chat(instruction="Hello, count tokens")
    gemini_instance.chat_session.history = [{"role": "user", "parts": ["Test message"]}]
    gemini_instance.model.count_tokens = MagicMock(return_value=3)
    gemini_instance._token_counts()
    gemini_instance.model.count_tokens.assert_called_once_with(gemini_instance.chat_session.history)

def test_upload_to_gemini(gemini_instance):
    """Test uploading files to Gemini"""
    gemini_instance.model.upload_file = MagicMock(return_value="MockFileURI")
    file_uri = gemini_instance.upload_to_gemini(path="/Users/mrthinh/Downloads/su-dung-tags-trong-obsidian-toi-uu.png", mime_type="text/plain")
    assert file_uri == "MockFileURI"
    gemini_instance.model.upload_file.assert_called_once_with("/Users/mrthinh/Downloads/su-dung-tags-trong-obsidian-toi-uu.png", mime_type="text/plain")

@pytest.mark.skipif(not IS_COLAB, reason="Test only applicable in Google Colab")
def test_wait_for_files_active(gemini_instance):
    """Test waiting for files to become active on Gemini API"""
    mock_file = MagicMock()
    mock_file.name = "test_file"
    mock_file.state.name = "PROCESSING"
    gemini_instance.model.get_file = MagicMock(side_effect=[mock_file, mock_file, MagicMock(state=MagicMock(name="ACTIVE"))])
    gemini_instance.wait_for_files_active([mock_file])
    gemini_instance.model.get_file.assert_called_with("test_file")

### Test Edge Cases ###

def test_config_invalid_temp(gemini_instance):
    """Test config with invalid temperature setting should raise ValueError."""
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1."):
        gemini_instance.config(temp=-1)  # Invalid temperature

def test_send_message_no_chat_session(gemini_instance):
    """Test sending a message without an initialized chat session should raise AttributeError."""
    # Make sure chat_session is not initialized
    if hasattr(gemini_instance, 'chat_session'):
        del gemini_instance.chat_session
    
    with pytest.raises(AttributeError, match="Chat session is not initialized."):
        gemini_instance.send_message(prompt="This should raise an error")

