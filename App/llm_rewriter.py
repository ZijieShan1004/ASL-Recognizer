import json
import urllib.request
import urllib.error

from shared_config import OLLAMA_URL, OLLAMA_MODEL, LLM_TIMEOUT_SECONDS


class LocalLLMRewriter:
  # Initialize the local LLM client
  def __init__(self):
    self.url = OLLAMA_URL
    self.model = OLLAMA_MODEL
    self.timeout = LLM_TIMEOUT_SECONDS

  # Build the rewrite prompt for ASL words
  def _build_prompt(self, words):
    joined = " ".join(words)
    return (
      "Convert the following ASL-like recognized words into one short, natural English sentence.\n"
      "Keep the meaning faithful.\n"
      "If the input is already a good phrase, keep it natural.\n"
      "Output only the final sentence.\n"
      f"Words: {joined}"
    )

  # Send one prompt to the local Ollama server
  def _call_ollama(self, prompt):
    payload = {
      "model": self.model,
      "prompt": prompt,
      "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
      self.url,
      data=data,
      headers={"Content-Type": "application/json"},
      method="POST"
    )

    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
      body = resp.read().decode("utf-8")
      obj = json.loads(body)
      return obj.get("response", "").strip()

  # Rewrite a confirmed sign sequence into a natural sentence
  def rewrite(self, words):
    if not words:
      return ""

    try:
      prompt = self._build_prompt(words)
      text = self._call_ollama(prompt)
      if text:
        return text
      return " ".join(words)
    except urllib.error.URLError:
      return " ".join(words)
    except Exception:
      return " ".join(words)