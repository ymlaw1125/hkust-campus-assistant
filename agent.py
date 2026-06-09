import logging
import os
from typing import Optional
from dotenv import load_dotenv
from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
from agent_interface import AgentInterface
from microsoft_agents.hosting.core import Authorization, TurnContext

load_dotenv()
logger = logging.getLogger(__name__)


class AgentFrameworkAgent(AgentInterface):

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.conversation_history = []
        self._create_client()

    def _create_client(self):
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        endpoint = endpoint.rstrip("/").replace("/openai/v1", "").replace("/openai", "")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

        token_provider = get_bearer_token_provider(
            AzureCliCredential(),
            "https://ai.azure.com/.default"
        )

        # Pass a dummy api_key to satisfy the openai client's validation,
        # the actual auth is done via azure_ad_token_provider
        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_ad_token_provider=token_provider,
            api_version=api_version,
            api_key="placeholder",
        )
        self.deployment = deployment
        logger.info(f"✅ Client created → {deployment}")

    async def initialize(self) -> None:
        logger.info("Agent ready")

    async def process_user_message(
        self,
        message: str,
        auth: Authorization,
        auth_handler_name: Optional[str],
        context: TurnContext,
    ) -> str:
        try:
            self.conversation_history.append({"role": "user", "content": message})
            messages = [
                {"role": "system", "content": "You are a helpful HKUST campus assistant."}
            ] + self.conversation_history
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=messages,
            )
            reply = response.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": reply})
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            return reply
        except Exception as e:
            logger.error(f"Error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"

    async def cleanup(self) -> None:
        self.conversation_history = []
        logger.info("Cleanup done")