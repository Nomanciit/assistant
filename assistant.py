from openai import OpenAI
import cohere
import os
import json
from datetime import datetime
import yaml
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import logging
import chromadb

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Assistant:
    def __init__(self):

        # Client for embeddings using the original API key
        self.client_openai = OpenAI(api_key=config['api_keys']['open_ai_embeddings'])
        self.MODEL = "text-embedding-3-small"

        chroma_client = chromadb.HttpClient(host='94.72.110.93', port=8000)
        self.index = chroma_client.get_collection(name="knowledgebase")
        self.co = cohere.Client(config['api_keys']['cohere_ai'])

        # New completions client for Deepseek completions
        self.client_deepseek = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=config['api_keys']['deepseek'],
        )

        # Initialize chat history store
        self.store = {}

    def update_chat_history(self, session_id, role, message):
        # Append the role and message to the chat history
        self.chat_history[session_id].append((role, message))

    def get_embeddings(self, text):
        try:
            response = self.client_openai.embeddings.create(
                input=text,
                model=self.MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return None

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def get_docs(self, query: str, top_k: int,category):

      filter_dict = {}

      xq = self.get_embeddings(query)

      res = self.index.query(query_embeddings=[xq],
                                 n_results=top_k,
                                where={"category": category})

      docs = {x: i for i, x in enumerate(res['documents'][0])}
      return docs

    def compare(self, query: str, top_k: int, top_n: int, category):
        # Get documents
        docs = self.get_docs(query, top_k=top_k, category=category)

        # Ensure docs is a dictionary with proper key-value pairs
        if not isinstance(docs, dict):
            logger.error("docs should be a dictionary")
            raise ValueError("docs should be a dictionary")


        # Perform reranking
        rerank_docs = self.co.rerank(
            query=query, documents=list(docs.keys()), top_n=top_n, model="rerank-english-v3.0"
        )


        # Check if rerank_docs.results is empty
        if not rerank_docs.results:
            print("No results returned by rerank.")
            return []

        # Collect reranked documents
        reranked_docs = []
        for doc in rerank_docs.results:
            # Fetch document text based on index
            text = list(docs.keys())[doc.index]  # Assumes `docs` keys are ordered consistently
            reranked_docs.append(text)

        return reranked_docs


    def create_chat_history_prompt(self, context, query):
        prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is.

Chat History:
{}

Latest user question:
{}
""".format(context, query)

        completion = self.client_deepseek.chat.completions.create(
            extra_body={},
            model="deepseek/deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        response = completion.choices[0].message.content
        logger.debug(f"Chat history prompt created for query: {query}")
        return response

    def create_prompt(self, context, query):
        header = """Answer the question as truthfully as possible using the provided context. If the answer is not contained within the text and requires the latest information to be updated,
please follow these guidelines:
- If you cannot find the answer, print 'Sorry, Not Sufficient Context to Answer Query.'

Context:
{}

Query:
{}
""".format(context, query)
        return header

    def generate_answer(self, context, query, model="deepseek/deepseek-chat"):
        response = self.client_deepseek.chat.completions.create(
            extra_body={},
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": context
                },
                {
                    "role": "assistant",
                    "content": query
                }
            ],
            temperature=1,
            max_tokens=3000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response.choices[0].message.content

    def clear_usercache(self, session_id: str):
        session_history = self.get_session_history(session_id)
        session_history.clear()

    def get_final_answer(self, session_id, query, category, model="deepseek/deepseek-chat",):
        try:
            # Retrieve session history
            session_history = self.get_session_history(session_id)
            stored_messages = session_history.get_messages()

            if len(stored_messages) <= 14:
                print("stored messages", stored_messages)
                if len(stored_messages) > 1:
                    query = self.create_chat_history_prompt(stored_messages, query)
                else:
                    query = query

                context = self.compare(query, 5, 2,category)
                context = " ".join(context)
                prompt = self.create_prompt(context, query)
                reply = self.generate_answer(context, query, model)

                # Update session history with user question and assistant's reply
                session_history.add_message(role="Human", content=query)
                session_history.add_message(role="Assistant", content=reply)

                return reply
            else:
                # Clear stored messages
                session_history.clear()

                # Keep only the last 6 messages
                last_six_messages = stored_messages[-6:]
                for role, content in last_six_messages:
                    session_history.add_message(role, content)

                stored_messages = session_history.get_messages()

                query = self.create_chat_history_prompt(stored_messages, query)
                context = self.compare(query, 5, 2,category)
                context = " ".join(context)
                prompt = self.create_prompt(context, query)
                reply = self.generate_answer(context, query, model)

                # Update session history with user question and assistant's reply
                session_history.add_message(role="Human", content=query)
                session_history.add_message(role="Assistant", content=reply)

                return reply
        except AttributeError as attr_err:
            print("Attribute Error:", attr_err)
        except IndexError as index_err:
            print("Index Error:", index_err)
        except Exception as e:
            session_history.clear()
            logger.error(f"Error generating final answer for session {session_id} with query {query}: {e}")
            return "An error occurred while processing your request."


class InMemoryChatMessageHistory(BaseChatMessageHistory):
    def __init__(self):
        self.messages = []

    def add_message(self, role, content):
        self.messages.append((role, content))

    def get_messages(self):
        return self.messages

    def clear(self):
        self.messages = []
