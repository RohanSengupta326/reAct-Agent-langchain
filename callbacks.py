from typing import Dict, Any, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult


# this extends a langchain's object that has all the methods which get triggered
# when some process starts regarding the llm.
class AgentCallbackHandler(BaseCallbackHandler):
    #override
    # triggers when the llm starts.
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("*********")

    #override
    #triggers when the llm generates some response.
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Run when LLM ends running."""
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("*********")
