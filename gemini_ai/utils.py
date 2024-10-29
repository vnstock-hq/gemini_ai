from IPython.display import Markdown
import textwrap
import os

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

def colab_verify() -> bool:
    """
    Verify if the environment is Google Colab.

    Returns:
        bool: True if running in Google Colab, else False.
    """
    if 'COLAB_JUPYTER_IP' in os.environ:
        print("Running on Colab")
        return True
    else:
        print("Not running on Colab")
        return False

def jupyter_verify() -> bool:
    """
    Verify if the environment is a Jupyter notebook.

    Returns:
        bool: True if running in Jupyter, else False.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            print("Running in Jupyter Notebook")
            return True
        elif shell == 'TerminalInteractiveShell':
            print("Running in Terminal or Script")
            return False
        else:
            print("Unknown environment")
            return False
    except NameError:
        print("Not running in an IPython environment")
        return False
