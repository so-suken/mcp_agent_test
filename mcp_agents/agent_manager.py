from typing import Dict, List, Optional, Any, Callable
import asyncio
import importlib

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import TextMentionTermination, MaxMessageTermination

class AgentManager:
    """
    Manages the creation, configuration, and selection of agents for the chat system.
    """
    # NOTE: AgentManagerクラスについて:
    # このクラスはマルチエージェントシステムの中心的なクラスで、以下の主要な役割を持ちます:
    # 1. エージェントの定義と設定の管理 (どのエージェントを有効/無効にするか)
    # 2. 必要に応じた遅延ロードによるエージェントの初期化
    # 3. SelectorGroupChatの作成と設定
    # 4. 終了条件やセレクタープロンプトの提供
    # 
    # これにより、メインのアプリケーションコードはエージェントの実装詳細から分離され、
    # 新しいエージェントの追加や設定の変更が容易になります。
    
    def __init__(self, model_client):
        """
        Initialize the AgentManager.
        
        Args:
            model_client: The model client to use for all agents
        """
        self.model_client = model_client
        self.available_agents = {}
        
        # Define available agent types and their module paths for lazy loading
        # NOTE: agent_typesについて:
        # このディクショナリは利用可能なエージェントタイプを定義し、遅延ロードに必要な情報を保持します:
        # - module: エージェント作成関数が定義されているモジュールのパス
        # - function: 実際にエージェントを作成する関数の名前
        # - loaded: このエージェントタイプがすでにロードされているかどうか
        # - create_fn: ロード済みの場合、キャッシュされた関数オブジェクト
        #
        # 遅延ロードにより、エージェントは実際に必要になった時点でのみロードされます。
        # これにより、使用しないエージェントのモジュールは読み込まれず、メモリとロード時間が節約されます。
        self.agent_types = {
            "dialogue_agent": {
                "module": "mcp_agents.dialogue_agent",
                "function": "create_dialogue_agent",
                "loaded": False,
                "create_fn": None
            },
            "postgres_agent": {
                "module": "mcp_agents.postgres_agent",
                "function": "create_postgres_agent", 
                "loaded": False,
                "create_fn": None
            },
            "formatter_agent": {
                "module": "mcp_agents.formatter_agent",
                "function": "create_formatter_agent",
                "loaded": False,
                "create_fn": None
            }
            # Custom agents can be added here
        }
        
        # Default configuration - which agents are enabled
        self.agent_config = {
            "dialogue_agent": True,
            "postgres_agent": True,
            "formatter_agent": False
        }
    
    def _load_agent_function(self, agent_type: str) -> Optional[Callable]:
        """
        Dynamically load the agent creation function if it's not already loaded.
        
        Args:
            agent_type: The type of agent to load
            
        Returns:
            The creation function or None if it couldn't be loaded
        """
        agent_info = self.agent_types.get(agent_type)
        if not agent_info:
            print(f"Unknown agent type: {agent_type}")
            return None
            
        # Return cached function if already loaded
        if agent_info["loaded"] and agent_info["create_fn"]:
            return agent_info["create_fn"]
            
        # Otherwise, load the module and function
        try:
            module = importlib.import_module(agent_info["module"])
            create_fn = getattr(module, agent_info["function"])
            
            # Cache the loaded function
            self.agent_types[agent_type]["loaded"] = True
            self.agent_types[agent_type]["create_fn"] = create_fn
            
            print(f"Loaded agent function {agent_info['function']} from {agent_info['module']}")
            return create_fn
        except Exception as e:
            print(f"Error loading agent function for {agent_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def register_agent_type(self, agent_name: str, module_path: str, function_name: str, enabled: bool = False):
        """
        Register a new agent type with the system.
        
        Args:
            agent_name: The name of the agent type
            module_path: The module path where the creation function resides
            function_name: The name of the function that creates the agent
            enabled: Whether the agent should be enabled by default
        """
        self.agent_types[agent_name] = {
            "module": module_path,
            "function": function_name,
            "loaded": False,
            "create_fn": None
        }
        self.agent_config[agent_name] = enabled
        print(f"Registered new agent type: {agent_name} (enabled: {enabled})")
    
    def configure_agents(self, config: Dict[str, bool]):
        """
        Configure which agents are enabled.
        
        Args:
            config: Dictionary of agent_name: is_enabled
        """
        self.agent_config.update(config)
        print(f"Updated agent configuration: {self.agent_config}")
    
    async def initialize_agents(self) -> List[AssistantAgent]:
        """
        Initialize all enabled agents based on the current configuration.
        
        Returns:
            List of initialized agents including the planner
        """
        print("Initializing agents based on configuration...")
        worker_agents = []
        
        # Initialize each enabled agent type
        for agent_type, is_enabled in self.agent_config.items():
            if is_enabled:
                try:
                    # Get the creation function for this agent type
                    create_fn = self._load_agent_function(agent_type)
                    
                    if create_fn:
                        # Create the agent (creation functions may be async)
                        agent = await create_fn(self.model_client) if asyncio.iscoroutinefunction(create_fn) else create_fn(self.model_client)
                        
                        if agent:  # Only add if agent was successfully created
                            worker_agents.append(agent)
                            self.available_agents[agent_type] = agent
                            print(f"Added {agent_type} to available agents")
                    else:
                        print(f"Could not load creation function for {agent_type}")
                    
                except Exception as e:
                    print(f"Error initializing {agent_type}: {e}")
                    import traceback
                    traceback.print_exc()
        
        if not worker_agents:
            print("Warning: No worker agents were successfully initialized!")
            return []
            
        # Get the names of initialized agents for the planner
        agent_names = [agent.name for agent in worker_agents]
        
        # Create and add the planner agent - load dynamically
        planner_fn = self._load_agent_function("planner_agent") or self._load_planner()
        if planner_fn:
            planner = planner_fn(self.model_client, agent_names)
            worker_agents.append(planner)
        else:
            print("Error: Could not load planner agent!")
        
        return worker_agents
    
    def _load_planner(self):
        """
        Load the planner agent creation function directly if not in agent_types.
        This is a fallback method.
        """
        try:
            # Add planner to agent_types
            self.agent_types["planner_agent"] = {
                "module": "mcp_agents.planner_agent",
                "function": "create_planner_agent",
                "loaded": False,
                "create_fn": None
            }
            return self._load_agent_function("planner_agent")
        except Exception as e:
            print(f"Error loading planner module: {e}")
            import traceback
            traceback.print_exc()
            
            # As a last resort, try direct import
            try:
                from mcp_agents.planner_agent import create_planner_agent
                return create_planner_agent
            except:
                print("Critical error: Failed to load planner agent!")
                return None
    
    def create_selector_prompt(self) -> str:
        """Create the selector prompt for SelectorGroupChat"""
        # NOTE: SelectorGroupChatで使用される変数の説明:
        # {roles}: 各エージェントの役割説明の文字列。自動的に "agent_name: agent_description" の形式で置換される。
        #   例: "dialogue_agent: Creates conversations between characters\npostgres_agent: Retrieves data from databases"
        # {participants}: チャットに参加している全エージェントの名前のリスト。配列としてモデルに提供される。
        #   例: ["dialogue_agent", "postgres_agent", "planner"]
        # {history}: これまでの会話履歴。各メッセージは "agent_name: message_content" の形式。
        #   これにより、どのエージェントが次に回答すべきかを決定するための文脈が提供される。
        return """
        Below is a conversation where a user has made a request, and a planner is coordinating specialized agents to fulfill it.
        The planner has already created a plan with specific tasks for each agent based on their capabilities.
        
        Team members and their roles:
        {roles}
        
        Based on the current conversation and tasks being discussed, which team member should respond next?
        Select one team member from {participants} who is best suited to handle the current task or situation.
        Return only the name of the selected team member.
        
        {history}
        """
    
    def create_termination_condition(self, keyword: str = "[TERMINATE_ALL]", max_turns: int = 15):
        """Create a termination condition for the chat"""
        # NOTE: 終了条件の説明:
        # TextMentionTermination: 特定のテキスト（ここでは "[TERMINATE_ALL]"）が会話に現れると会話を終了する
        # MaxMessageTermination: 指定したメッセージ数（ここでは15）に達すると会話を終了する
        # | 演算子: 論理OR。どちらかの条件が満たされた場合に終了する
        # これらの条件を組み合わせることで、明示的な終了指示または最大ターン数に基づいて会話を終了できる
        
        # Use Autogen's built-in termination conditions
        text_termination = TextMentionTermination(text=keyword)
        max_msg_termination = MaxMessageTermination(max_messages=max_turns)
        return text_termination | max_msg_termination
    
    async def create_chat(self) -> Optional[SelectorGroupChat]:
        """
        Create a SelectorGroupChat with all initialized agents.
        
        Returns:
            SelectorGroupChat instance or None if no agents were initialized
        """
        # NOTE: SelectorGroupChatの概要:
        # SelectorGroupChatは複数のエージェント間の会話を管理するためのクラス
        # - participants: 会話に参加するエージェントのリスト
        # - model_client: エージェント選択に使用するモデル
        # - selector_prompt: どのエージェントが次に回答すべきかを決定するためのプロンプト
        # - termination_condition: 会話を終了するための条件
        # このクラスは各ターンで最適なエージェントを選び、そのエージェントに返答を生成させる
        
        # Initialize all enabled agents
        all_agents = await self.initialize_agents()
        
        if not all_agents:
            print("Error: No agents available. Cannot create chat.")
            return None
        
        # Create the SelectorGroupChat
        chat = SelectorGroupChat(
            participants=all_agents,
            model_client=self.model_client,
            selector_prompt=self.create_selector_prompt(),
            termination_condition=self.create_termination_condition()
        )
        
        print(f"Created SelectorGroupChat with {len(all_agents)} agents")
        return chat 
