import streamlit as st
import json
import os
import uuid
from datetime import datetime

# Langchain imports
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import pypdf  
import docx   
# import toml

# Langchain Tools Imports
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools import DuckDuckGoSearchRun

# Constants
HISTORY_FILE = 'chat_history.json'
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

def set_custom_styles():
    """Apply custom CSS styles to enhance UI."""
    st.markdown("""
    <style>
    
    .stTitle {
        color: #2c3e50;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 20px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 6px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: scale(1.05);
    }
    .stSidebar .stSubheader {
        color: #34495e;
        font-weight: bold;
    }
    .chat-message {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .human-message {
        background-color: #e6f2ff;
    }
    .ai-message {
        background-color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)

class FileProcessor:
    @staticmethod
    def read_txt(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def read_pdf(file_path):
        reader = pypdf.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    @staticmethod
    def read_docx(file_path):
        doc = docx.Document(file_path)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    @staticmethod
    def read_md(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

class LangchainToolManager:
    """
    Manages Langchain tools and provides methods for tool selection and execution
    """
    @staticmethod
    def get_available_tools():
        """
        Returns a dictionary of available Langchain tools
        """
        return {
            'None': None,
            'Wikipedia Search': 'wikipedia',
            'Web Search (SerpAPI)': 'serpapi',
            'DuckDuckGo Search': 'duckduckgo',
            'Web Page Loader': 'webpage_loader'
        }
    
    @staticmethod
    def create_tools():
        """
        Create and return a list of Langchain tools
        """
        tools = []
        if 'Wikipedia Search' in st.session_state.selected_tool:
    
            # Wikipedia Tool
            wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
            tools.append(
                Tool(
                    name="Wikipedia",
                    func=wikipedia_tool.run,
                    description="Useful for searching Wikipedia for factual information."
                )
            )
        
        if 'Web Search (SerpAPI)' in st.session_state.selected_tool:
            # SerpAPI Search Tool
            if os.environ.get("SERPAPI_API_KEY"):
                search = SerpAPIWrapper()
                tools.append(
                    Tool(
                        name="Web Search",
                        func=search.run,
                        description="Useful for searching the internet for current information."
                    )
                )
        
        if 'DuckDuckGo Search' in st.session_state.selected_tool:
            # DuckDuckGo Search Tool
            ddg_search = DuckDuckGoSearchRun()
            tools.append(
                Tool(
                    name="DuckDuckGo Search",
                    func=ddg_search.run,
                    description="Alternative web search tool using DuckDuckGo."
                )
            )
        
        if 'Web Page Loader' in st.session_state.selected_tool:
            # Web Page Loader Tool
            tools.append(
                Tool(
                    name="Web Page Loader",
                    func=lambda url: WebBaseLoader(url).load_and_split() if url else "No URL provided",
                    description="Useful for loading and extracting text from web pages."
                )
            )
            
        return tools

class AIAssistant:
    def __init__(self):
        # Initialize session state variables
        if 'conversations' not in st.session_state:
            st.session_state.conversations = self.load_conversations()
        
        if 'current_conversation_id' not in st.session_state:
            st.session_state.current_conversation_id = None
        
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        if 'selected_llm' not in st.session_state:
            st.session_state.selected_llm = 'GPT-4 Mini'
        
        if 'uploaded_file' not in st.session_state:
            st.session_state.uploaded_file = None
        
        # Initialize tool-related session state
        if 'selected_tool' not in st.session_state:
            st.session_state.selected_tool = 'None'
        
        # Initialize Langchain Tool Manager
        self.tool_manager = LangchainToolManager()

        # LLM Configuration
        self.llm_providers = {
            'GPT-4 Mini': self.get_openai_llm,
            'Ollama': self.get_ollama_llm
        }

    def get_openai_llm(self):
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    def get_ollama_llm(self):
        return ChatOpenAI(
                model="ollama/llama3.2",
                base_url="http://localhost:11434"
            )

    def load_conversations(self):
        try:
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, 'r') as f:
                    conversations = json.load(f)
                
                # Ensure each conversation has an ID
                for conv in conversations:
                    if 'id' not in conv:
                        conv['id'] = str(uuid.uuid4())
                
                return conversations
        except (json.JSONDecodeError, IOError):
            st.warning("Error loading chat history. Starting with a fresh conversation.")
        
        return []

    def save_conversations(self):
        try:
            with open(HISTORY_FILE, 'w') as f:
                json.dump(st.session_state.conversations, f)
        except IOError:
            st.error("Failed to save conversations.")

    def create_or_update_conversation(self):
        # Add timestamp to messages
        for message in st.session_state.messages:
            if 'timestamp' not in message:
                message['timestamp'] = datetime.now().isoformat()

        # If no current conversation, create a new one
        if st.session_state.current_conversation_id is None:
            new_conversation = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'title': (st.session_state.messages[0]['content'][:50] + '...') 
                         if st.session_state.messages else 'New Conversation',
                'messages': st.session_state.messages
            }
            st.session_state.conversations.append(new_conversation)
            st.session_state.current_conversation_id = new_conversation['id']
        else:
            # Update existing conversation
            for conversation in st.session_state.conversations:
                if conversation.get('id') == st.session_state.current_conversation_id:
                    conversation['messages'] = st.session_state.messages
                    break
        
        # Limit conversations to last 10
        if len(st.session_state.conversations) > 10:
            st.session_state.conversations = st.session_state.conversations[-10:]
        
        # Save conversations
        self.save_conversations()

    def start_new_conversation(self):
        # Reset current conversation
        st.session_state.current_conversation_id = None
        st.session_state.messages = []
        st.session_state.uploaded_file = None
        st.session_state.selected_tool = 'None'
        st.toast("🆕 New conversation started!")

    def load_conversation(self, conversation_id):
        for conversation in st.session_state.conversations:
            if conversation.get('id') == conversation_id:
                st.session_state.messages = conversation.get('messages', [])
                st.session_state.current_conversation_id = conversation_id
                st.toast("💬 Conversation loaded successfully!")
                return
        
        # If no conversation found, start a new one
        self.start_new_conversation()

    def render_sidebar(self):
        st.sidebar.title("🤖 AI Assistant")
        
        # New Chat Button
        if st.sidebar.button("🆕 New Chat", key="new_chat_btn", help="Start a fresh conversation"):
            self.start_new_conversation()
        
        # File Upload
        uploaded_file = st.sidebar.file_uploader(
            "Upload a file", 
            type=['txt', 'pdf', 'docx', 'md'],
            help="Max file size: 10MB"
        )
        
        # Handle file upload
        if uploaded_file is not None:
            # Check file size
            if uploaded_file.size > MAX_FILE_SIZE:
                st.sidebar.error(f"File too large. Max size is {MAX_FILE_SIZE/1024/1024} MB.")
                st.session_state.uploaded_file = None
            else:
                st.session_state.uploaded_file = uploaded_file
                st.sidebar.success(f"Uploaded: {uploaded_file.name}")
        
        # LLM Selection
        st.sidebar.header("🧠 LLM Provider")
        llm_options = {
            'GPT-4 Mini': '🤖 OpenAI',
            'Ollama': '🌐 Local LLM'
        }
        
        st.session_state.selected_llm = st.sidebar.radio(
            "Select Language Model", 
            list(llm_options.keys()),
            format_func=lambda x: llm_options[x],
            index=list(llm_options.keys()).index(st.session_state.selected_llm)
        )
        if st.session_state.selected_llm == 'GPT-4 Mini':
            api_key = st.sidebar.text_input("OpenAI API Key", key="openai_api_key", type="password")
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            else:
                st.sidebar.error("Please enter OpenAI API Key")
        # Tools Section
        st.sidebar.header("🛠️ Langchain Tools")
        tool_options = self.tool_manager.get_available_tools()
        
        st.session_state.selected_tool = st.sidebar.multiselect(
            "Select a Langchain Tool",
            list(tool_options.keys()),
            default=["None"]
            # index=list(tool_options.keys()).index(st.session_state.selected_tool)
        )
        if 'Web Search (SerpAPI)' in st.session_state.selected_tool:
            serp_api_key = st.sidebar.text_input("SerpAPI API Key", key="serp_api_key", type="password")
            if serp_api_key:
                os.environ["SERPAPI_API_KEY"] = serp_api_key
            else:
                st.sidebar.error("Please enter SerpAPI API Key")
        
        # Optional tool-specific input based on selected tool
        # if 'Web Page Loader' in st.session_state.selected_tool:
        #     st.session_state.webpage_url = st.sidebar.text_input("Enter URL if you want scrape content", key="webpage_url")
        
        # Conversation History with More Details
        st.sidebar.subheader("📜 Chat History")
        for i, conversation in enumerate(reversed(st.session_state.conversations)):
            col1, col2 = st.sidebar.columns([3, 1])
            
            conv_date = datetime.fromisoformat(conversation.get('timestamp', datetime.now().isoformat()))
            display_title = conversation.get('title', f'Conversation {i+1}')
            display_date = conv_date.strftime("%m/%d %H:%M")
            
            col1.button(f"{display_title} ({display_date})", 
                        key=f"conv_btn_{conversation.get('id', i)}",
                        on_click=lambda cid=conversation.get('id'): self.load_conversation(cid))
            
            # Optional delete button
            if col2.button("❌", key=f"del_btn_{conversation.get('id', i)}"):
                # Remove the conversation
                st.session_state.conversations = [
                    c for c in st.session_state.conversations 
                    if c.get('id') != conversation.get('id')
                ]
                self.save_conversations()

    def process_uploaded_file(self, uploaded_file):
        """
        Process the uploaded file and return its content.
        If the file is long, it will be summarized.
        """
        # Ensure a temporary directory exists
        os.makedirs('temp', exist_ok=True)
        
        # Generate a unique temporary filename
        temp_filename = os.path.join('temp', f"temp_{uuid.uuid4()}")
        full_path = temp_filename + os.path.splitext(uploaded_file.name)[1]
        
        try:
            # Save uploaded file temporarily
            with open(full_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Read file based on extension
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            if file_ext == '.txt':
                content = FileProcessor.read_txt(full_path)
            elif file_ext == '.pdf':
                content = FileProcessor.read_pdf(full_path)
            elif file_ext == '.docx':
                content = FileProcessor.read_docx(full_path)
            elif file_ext == '.md':
                content = FileProcessor.read_md(full_path)
            else:
                return f"Unsupported file type: {file_ext}"
            
            # If document is too long, summarize
            if len(content) > 3000:  # Threshold for summarization
                # Initialize LLM for summarization
                llm = self.llm_providers[st.session_state.selected_llm]()
                
                # Text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                # Split the content into documents
                split_texts = text_splitter.create_documents([content])
                
                # Load summarization chain
                chain = load_summarize_chain(
                    llm, 
                    chain_type="map_reduce",
                    verbose=True
                )
                
                # Summarize the document
                summary = chain.run(split_texts)
                return f"File Summary of {uploaded_file.name}:\n{summary}"
            else:
                # Return full content for shorter files
                return content
        except Exception as e:
            return f"Error processing file: {str(e)}"
        finally:
            if os.path.exists(full_path):
                os.remove(full_path)
            
            # Remove temporary directory if empty
            try:
                os.rmdir('temp')
            except OSError:
                # Directory not empty or other issue - ignore
                pass

    def process_query_with_tool(self, prompt):
        """
        Process user query using the selected Langchain tool
        """
        # Create tools
        tools = self.tool_manager.create_tools()
        
        # Initialize LLM for agent
        llm = self.llm_providers[st.session_state.selected_llm]()
        
        # Initialize agent
        agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # Tool-specific modifications
        # if st.session_state.selected_tool == 'Web Page Loader':
        #     # If Web Page Loader is selected and URL is provided
        #     webpage_url = st.session_state.get('webpage_url', '')
        #     if webpage_url:
        #         # Modify prompt to include URL context
        #         prompt = f"URL: {webpage_url}\n\nQuery: {prompt}"
        
        # Run agent
        try:
            response = agent.run(prompt)
            return response
        except Exception as e:
            return f"Error using tool: {str(e)}"

    def main(self):
        # Set custom styles
        set_custom_styles()

        # Display welcome toast
        st.toast("Welcome to AI Assistant! 🎉")

        st.title("🤖 AI Assistant")

        # Render conversation history and file upload sidebar
        self.render_sidebar()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message.get("role", "human"), 
                                 avatar=("🧑" if message.get("role") == "human" else "🤖")):
                st.markdown(message.get("content", ""))
                
                # Optional timestamp
                if message.get("timestamp"):
                    st.caption(f"Sent at {datetime.fromisoformat(message['timestamp']).strftime('%m/%d %H:%M')}")

        # User input
        if prompt := st.chat_input("What would you like to chat about?"):
            # Check if there's an uploaded file and first message includes file context
            if st.session_state.uploaded_file and len(st.session_state.messages) == 0:
                # Process the uploaded file
                file_content = self.process_uploaded_file(st.session_state.uploaded_file)
                
                # Modify prompt to include file context
                prompt = f"File Context: {file_content}\n\nUser Query: {prompt}"

            # Add human message to session state
            st.session_state.messages.append({
                "role": "human", 
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display human message
            with st.chat_message("human", avatar="🧑"):
                st.markdown(prompt)

            # Display AI response
            with st.chat_message("assistant", avatar="🤖"):
                message_placeholder = st.empty()
                
                # Check if a tool is selected
                if st.session_state.selected_tool and st.session_state.selected_tool != 'None':
                    # Use tool-based response generation
                    full_response = self.process_query_with_tool(prompt)
                else:
                    # Use standard LLM-based response generation
                    # Get selected LLM
                    llm = self.llm_providers[st.session_state.selected_llm]()
                    
                    # Prepare conversation history for LLM
                    conversation_history = [
                        HumanMessage(content=msg['content']) if msg['role'] == 'human' 
                        else AIMessage(content=msg['content']) 
                        for msg in st.session_state.messages[:-1]
                    ]
                    
                    # Get AI response
                    full_response = ""
                    for chunk in llm.stream(conversation_history + [HumanMessage(content=prompt)]):
                        full_response += chunk.content
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add AI response to session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Save or update conversation
            self.create_or_update_conversation()

def main():
    assistant = AIAssistant()
    assistant.main()

if __name__ == "__main__":
    main()