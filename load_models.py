import google.auth
import google.auth.transport.requests
import openai
from helper import retry_if_fail
from anthropic import AnthropicVertex
from environment import LOCATION, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_API_VERSION

def refresh_creds(creds):
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

# Setup client with Google Cloud credentials
def load_claude():
    creds, project_id = google.auth.default()
    refresh_creds(creds)

    client = AnthropicVertex(
        project_id=project_id, 
        region=LOCATION,
        credentials=creds
    )
    return client, creds

def load_gpt4o():
    client = openai.AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_API_VERSION
    )
    return client

@retry_if_fail
def call_gpt4o(client, prompt, seed):
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = prompt,
        seed = seed
    ) 
    answer = (response.choices[0].message.content)
    return answer

@retry_if_fail
def call_claude(client, creds, system_msg, prompt):
    if (creds.expired):
        refresh_creds(creds)
        client.api_key = creds.token

    response = client.messages.create(
        model = 'claude-3-5-sonnet-v2@20241022',
        messages = prompt,
        system = system_msg,
        max_tokens = 512,
    )
    answer = response.content[0].text
    return answer