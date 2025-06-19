# The Ultimate Google Agent Development Kit (ADK) Python Cheatsheet - Expanded Edition

This document serves as a long-form, comprehensive reference for building, orchestrating, and deploying AI agents using the Python Agent Development Kit (ADK). It aims to cover every significant aspect with greater detail, more code examples, and in-depth best practices.

## Table of Contents

1.  [Core Concepts & Project Structure](#1-core-concepts--project-structure)
    *   1.1 ADK's Foundational Principles
    *   1.2 Essential Primitives
    *   1.3 Standard Project Layout
2.  [Agent Definitions (`LlmAgent`)](#2-agent-definitions-llmagent)
    *   2.1 Basic `LlmAgent` Setup
    *   2.2 Advanced `LlmAgent` Configuration
    *   2.3 LLM Instruction Crafting
3.  [Orchestration with Workflow Agents](#3-orchestration-with-workflow-agents)
    *   3.1 `SequentialAgent`: Linear Execution
    *   3.2 `ParallelAgent`: Concurrent Execution
    *   3.3 `LoopAgent`: Iterative Processes
4.  [Multi-Agent Systems & Communication](#4-multi-agent-systems--communication)
    *   4.1 Agent Hierarchy
    *   4.2 Inter-Agent Communication Mechanisms
    *   4.3 Common Multi-Agent Patterns
5.  [Building Custom Agents (`BaseAgent`)](#5-building-custom-agents-baseagent)
    *   5.1 When to Use Custom Agents
    *   5.2 Implementing `_run_async_impl`
6.  [Models: Gemini, LiteLLM, and Vertex AI](#6-models-gemini-litellm-and-vertex-ai)
    *   6.1 Google Gemini Models (AI Studio & Vertex AI)
    *   6.2 Other Cloud & Proprietary Models via LiteLLM
    *   6.3 Open & Local Models via LiteLLM (Ollama, vLLM)
    *   6.4 Customizing LLM API Clients
7.  [Tools: The Agent's Capabilities](#7-tools-the-agents-capabilities)
    *   7.1 Defining Function Tools: Principles & Best Practices
    *   7.2 The `ToolContext` Object: Accessing Runtime Information
    *   7.3 All Tool Types & Their Usage
8.  [Context, State, and Memory Management](#8-context-state-and-memory-management)
    *   8.1 The `Session` Object & `SessionService`
    *   8.2 `State`: The Conversational Scratchpad
    *   8.3 `Memory`: Long-Term Knowledge & Retrieval
    *   8.4 `Artifacts`: Binary Data Management
9.  [Runtime, Events, and Execution Flow](#9-runtime-events-and-execution-flow)
    *   9.1 The `Runner`: The Orchestrator
    *   9.2 The Event Loop: Core Execution Flow
    *   9.3 `Event` Object: The Communication Backbone
    *   9.4 Asynchronous Programming (Python Specific)
10. [Control Flow with Callbacks](#10-control-flow-with-callbacks)
    *   10.1 Callback Mechanism: Interception & Control
    *   10.2 Types of Callbacks
    *   10.3 Callback Best Practices
11. [Authentication for Tools](#11-authentication-for-tools)
    *   11.1 Core Concepts: `AuthScheme` & `AuthCredential`
    *   11.2 Interactive OAuth/OIDC Flows
    *   11.3 Custom Tool Authentication
12. [Deployment Strategies](#12-deployment-strategies)
    *   12.1 Local Development & Testing (`adk web`, `adk run`, `adk api_server`)
    *   12.2 Vertex AI Agent Engine
    *   12.3 Cloud Run
    *   12.4 Google Kubernetes Engine (GKE)
    *   12.5 CI/CD Integration
13. [Evaluation and Safety](#13-evaluation-and-safety)
    *   13.1 Agent Evaluation (`adk eval`)
    *   13.2 Safety & Guardrails
14. [Debugging, Logging & Observability](#14-debugging-logging--observability)
15. [Advanced I/O Modalities](#15-advanced-io-modalities)
16. [Performance Optimization](#16-performance-optimization)
17. [General Best Practices & Common Pitfalls](#17-general-best-practices--common-pitfalls)

---

## 1. Core Concepts & Project Structure

### 1.1 ADK's Foundational Principles

*   **Modularity**: Break down complex problems into smaller, manageable agents and tools.
*   **Composability**: Combine simple agents and tools to build sophisticated systems.
*   **Observability**: Detailed event logging and tracing capabilities to understand agent behavior.
*   **Extensibility**: Easily integrate with external services, models, and frameworks.
*   **Deployment-Agnostic**: Design agents once, deploy anywhere.

### 1.2 Essential Primitives

*   **`Agent`**: The core intelligent unit. Can be `LlmAgent` (LLM-driven) or `BaseAgent` (custom/workflow).
*   **`Tool`**: Callable function/class providing external capabilities (`FunctionTool`, `OpenAPIToolset`, etc.).
*   **`Session`**: A unique, stateful conversation thread with history (`events`) and short-term memory (`state`).
*   **`State`**: Key-value dictionary within a `Session` for transient conversation data.
*   **`Memory`**: Long-term, searchable knowledge base beyond a single session (`MemoryService`).
*   **`Artifact`**: Named, versioned binary data (files, images) associated with a session or user.
*   **`Runner`**: The execution engine; orchestrates agent activity and event flow.
*   **`Event`**: Atomic unit of communication and history; carries content and side-effect `actions`.
*   **`InvocationContext`**: The comprehensive root context object holding all runtime information for a single `run_async` call.

### 1.3 Standard Project Layout

A well-structured ADK project is crucial for maintainability and leveraging `adk` CLI tools.

```
your_project_root/
├── my_first_agent/             # Each folder is a distinct agent app
│   ├── __init__.py             # Makes `my_first_agent` a Python package (`from . import agent`)
│   ├── agent.py                # Contains `root_agent` definition and `LlmAgent`/WorkflowAgent instances
│   ├── tools.py                # Custom tool function definitions
│   ├── data/                   # Optional: static data, templates
│   └── .env                    # Environment variables (API keys, project IDs)
├── my_second_agent/
│   ├── __init__.py
│   └── agent.py
├── requirements.txt            # Project's Python dependencies (e.g., google-adk, litellm)
├── tests/                      # Unit and integration tests
│   ├── unit/
│   │   └── test_tools.py
│   └── integration/
│       └── test_my_first_agent.py
│       └── my_first_agent.evalset.json # Evaluation dataset for `adk eval`
└── main.py                     # Optional: Entry point for custom FastAPI server deployment
```
*   `adk web` and `adk run` automatically discover agents in subdirectories with `__init__.py` and `agent.py`.
*   `.env` files are automatically loaded by `adk` tools when run from the root or agent directory.

---

## 2. Agent Definitions (`LlmAgent`)

The `LlmAgent` is the cornerstone of intelligent behavior, leveraging an LLM for reasoning and decision-making.

### 2.1 Basic `LlmAgent` Setup

```python
from google.adk.agents import Agent

def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city."""
    # Mock implementation
    if city.lower() == "new york":
        return {"status": "success", "time": "10:30 AM EST"}
    return {"status": "error", "message": f"Time for {city} not available."}

my_first_llm_agent = Agent(
    name="time_teller_agent",
    model="gemini-2.0-flash", # Essential: The LLM powering the agent
    instruction="You are a helpful assistant that tells the current time in cities. Use the 'get_current_time' tool for this purpose.",
    description="Tells the current time in a specified city.", # Crucial for multi-agent delegation
    tools=[get_current_time] # List of callable functions/tool instances
)
```

### 2.2 Advanced `LlmAgent` Configuration

*   **`generate_content_config`**: Controls LLM generation parameters (temperature, token limits, safety).
    ```python
    from google.genai import types as genai_types
    from google.adk.agents import Agent

    gen_config = genai_types.GenerateContentConfig(
        temperature=0.2,            # Controls randomness (0.0-1.0), lower for more deterministic.
        top_p=0.9,                  # Nucleus sampling: sample from top_p probability mass.
        top_k=40,                   # Top-k sampling: sample from top_k most likely tokens.
        max_output_tokens=1024,     # Max tokens in LLM's response.
        stop_sequences=["## END"]   # LLM will stop generating if these sequences appear.
    )
    agent = Agent(
        # ... basic config ...
        generate_content_config=gen_config
    )
    ```

*   **`output_key`**: Automatically saves the agent's final text or structured (if `output_schema` is used) response to the `session.state` under this key. Facilitates data flow between agents.
    ```python
    agent = Agent(
        # ... basic config ...
        output_key="llm_final_response_text"
    )
    # After agent runs, session.state['llm_final_response_text'] will contain its output.
    ```

*   **`input_schema` & `output_schema`**: Define strict JSON input/output formats using Pydantic models.
    > **Warning**: Using `output_schema` forces the LLM to generate JSON and **disables** its ability to use tools or delegate to other agents.

    ```python
    from pydantic import BaseModel, Field
    from google.adk.agents import Agent

    class UserQuerySchema(BaseModel):
        city: str = Field(description="The city for which weather is requested.")
        unit: str = Field(default="celsius", description="Preferred temperature unit ('celsius' or 'fahrenheit').")

    class WeatherReportSchema(BaseModel):
        city: str
        temperature: float
        unit: str
        condition: str
        # Note: LLM will need to *invent* this data if it doesn't use tools.

    # Agent expecting structured JSON input
    input_schema_agent = Agent(
        name="structured_input_agent",
        model="gemini-2.0-flash",
        instruction="Process the weather request from the provided JSON. Respond with the temperature unit converted.",
        input_schema=UserQuerySchema # LLM will expect JSON like {"city": "London", "unit": "fahrenheit"}
    )

    # Agent generating structured JSON output (cannot use tools)
    output_schema_agent = Agent(
        name="structured_output_agent",
        model="gemini-2.0-flash",
        instruction="For New York, generate a fictional weather report. Respond ONLY with a JSON object matching the WeatherReportSchema.",
        output_schema=WeatherReportSchema, # LLM must output JSON matching this schema
        output_key="weather_json_report"
    )
    ```

*   **`include_contents`**: Controls whether the conversation history is sent to the LLM.
    *   `'default'` (default): Sends relevant history.
    *   `'none'`: Sends no history; agent operates purely on current turn's input and `instruction`. Useful for stateless API wrapper agents.
    ```python
    agent = Agent(..., include_contents='none')
    ```

*   **`planner`**: Assign a `BasePlanner` instance (e.g., `ReActPlanner`) to enable multi-step reasoning and planning. (Advanced, covered in Multi-Agents).

*   **`executor`**: Assign a `BaseCodeExecutor` (e.g., `BuiltInCodeExecutor`) to allow the agent to execute code blocks.
    ```python
    from google.adk.code_executors import BuiltInCodeExecutor
    agent = Agent(
        name="code_agent",
        model="gemini-2.0-flash",
        instruction="Write and execute Python code to solve math problems.",
        executor=[BuiltInCodeExecutor] # Allows agent to run Python code
    )
    ```

*   **Callbacks**: Hooks for observing and modifying agent behavior at key lifecycle points (`before_model_callback`, `after_tool_callback`, etc.). (Covered in Callbacks).

### 2.3 LLM Instruction Crafting (`instruction`)

The `instruction` (or `system_instruction`) is critical. It guides the LLM's behavior, persona, and tool usage.

*   **Purpose**: Define the agent's role, constraints, and how to interact with users and tools.
*   **Format**: Can be a simple string or a multi-line markdown string for complex instructions.
*   **Dynamic Injection**: Use `{state_key}` or `{artifact.filename}` placeholders for dynamic values from session state or artifacts. Use `{state_key?}` for optional keys that won't raise an error if missing.

    ```python
    my_agent = Agent(
        # ...
        instruction="""You are a professional customer support agent.
        Your name is {agent_name} and you speak {user:preferred_language}.
        Always refer to the user by their preferred name: {user:first_name?}.
        
        When the user asks for a refund, use the 'process_refund' tool.
        Refund policy is in artifact: {artifact.refund_policy_doc}.
        
        If the 'process_refund' tool returns an error, retrieve additional context from:
        {temp:last_error_details?}
        
        ONLY respond with text. Do NOT generate code.
        """,
        # ...
    )
    ```

*   **Best Practices for Instructions:**
    *   **Be Specific & Concise**: Avoid ambiguity.
    *   **Define Persona**: Give the LLM a clear role.
    *   **Tool Usage**: Explain *when* to use each tool, its parameters, and expected results.
    *   **Constraints**: Explicitly state what the LLM should *not* do.
    *   **Examples (Few-Shot)**: Provide input/output examples within the instruction for complex formats or tricky edge cases.
    *   **Error Handling**: Guide the LLM on how to interpret and respond to tool errors or unexpected situations.
    *   **Iteration**: LLM instructions are an iterative process. Test, observe, and refine.

---

## 3. Orchestration with Workflow Agents

Workflow agents (`SequentialAgent`, `ParallelAgent`, `LoopAgent`) provide deterministic control flow, combining LLM capabilities with structured execution. They do **not** use an LLM for their own orchestration logic.

### 3.1 `SequentialAgent`: Linear Execution

Executes `sub_agents` one after another in the order defined. The `InvocationContext` is passed along, allowing state changes to be visible to subsequent agents.

```python
from google.adk.agents import SequentialAgent, Agent

# Agent 1: Summarizes a document and saves to state
summarizer = Agent(
    name="DocumentSummarizer",
    model="gemini-2.0-flash",
    instruction="Summarize the provided document in 3 sentences.",
    output_key="document_summary" # Output saved to session.state['document_summary']
)

# Agent 2: Generates questions based on the summary from state
question_generator = Agent(
    name="QuestionGenerator",
    model="gemini-2.0-flash",
    instruction="Generate 3 comprehension questions based on this summary: {document_summary}",
    # 'document_summary' is dynamically injected from session.state
)

document_pipeline = SequentialAgent(
    name="SummaryQuestionPipeline",
    sub_agents=[summarizer, question_generator], # Order matters!
    description="Summarizes a document then generates questions."
)
```

### 3.2 `ParallelAgent`: Concurrent Execution

Executes `sub_agents` simultaneously. Useful for independent tasks to reduce overall latency. All sub-agents share the same `session.state`.

```python
from google.adk.agents import ParallelAgent, Agent

# Agents to fetch data concurrently
fetch_stock_price = Agent(name="StockPriceFetcher", ..., output_key="stock_data")
fetch_news_headlines = Agent(name="NewsFetcher", ..., output_key="news_data")
fetch_social_sentiment = Agent(name="SentimentAnalyzer", ..., output_key="sentiment_data")

# Agent to merge results (runs after ParallelAgent, usually in a SequentialAgent)
merger_agent = Agent(
    name="ReportGenerator",
    model="gemini-2.0-flash",
    instruction="Combine stock data: {stock_data}, news: {news_data}, and sentiment: {sentiment_data} into a market report."
)

# Pipeline to run parallel fetching then sequential merging
market_analysis_pipeline = SequentialAgent(
    name="MarketAnalyzer",
    sub_agents=[
        ParallelAgent(
            name="ConcurrentFetch",
            sub_agents=[fetch_stock_price, fetch_news_headlines, fetch_social_sentiment]
        ),
        merger_agent # Runs after all parallel agents complete
    ]
)
```
*   **Concurrency Caution**: When parallel agents write to the same `state` key, race conditions can occur. Always use distinct `output_key`s or manage concurrent writes explicitly (e.g., using `threading.Lock` if outside ADK's `EventActions` atomic updates).

### 3.3 `LoopAgent`: Iterative Processes

Repeatedly executes its `sub_agents` (sequentially within each loop iteration) until a condition is met or `max_iterations` is reached.

```python
from google.adk.agents import LoopAgent, Agent, BaseAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from typing import AsyncGenerator

# Agent to generate or refine content
draft_writer = Agent(
    name="DraftWriter",
    model="gemini-2.0-flash",
    instruction="Expand on the current draft: {current_draft?}. Target length is 200 words. Output ONLY the updated draft.",
    output_key="current_draft"
)

# Custom agent to check a condition and escalate
class WordCountChecker(BaseAgent):
    def __init__(self, name: str, min_words: int):
        super().__init__(name=name)
        self.min_words = min_words
    
    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        current_draft = ctx.session.state.get("current_draft", "")
        word_count = len(current_draft.split())
        
        if word_count >= self.min_words:
            print(f"[{self.name}] Target word count ({self.min_words}) reached ({word_count} words). Escalating to stop loop.")
            # Set escalate=True to signal LoopAgent to stop
            yield Event(author=self.name, actions=EventActions(escalate=True))
        else:
            print(f"[{self.name}] Word count: {word_count}. Still needs more work.")
            # No escalate action, loop continues
        # No final response needed from this checker agent for the user

# Define the loop
document_refiner = LoopAgent(
    name="DocumentRefiner",
    sub_agents=[draft_writer, WordCountChecker(name="WordCountChecker", min_words=200)],
    max_iterations=5, # Fallback to prevent infinite loops
    description="Iteratively refines a document until a target word count is reached or max iterations."
)
```
*   **Termination**: A `LoopAgent` terminates when:
    *   `max_iterations` is reached.
    *   Any `Event` yielded by a sub-agent (or a tool within it) sets `actions.escalate = True`. This provides dynamic, content-driven loop termination.

---

## 4. Multi-Agent Systems & Communication

Building complex applications by composing multiple, specialized agents.

### 4.1 Agent Hierarchy

A hierarchical (tree-like) structure of parent-child relationships defined by the `sub_agents` parameter during `BaseAgent` initialization. An agent can only have one parent.

```python
# Conceptual Hierarchy
# Root
# └── Coordinator (LlmAgent)
#     ├── SalesAgent (LlmAgent)
#     └── SupportAgent (LlmAgent)
#     └── DataPipeline (SequentialAgent)
#         ├── DataFetcher (LlmAgent)
#         └── DataProcessor (LlmAgent)
```

### 4.2 Inter-Agent Communication Mechanisms

1.  **Shared Session State (`session.state`)**: The most common and robust method. Agents read from and write to the same mutable dictionary.
    *   **Mechanism**: Agent A sets `ctx.session.state['key'] = value`. Agent B later reads `ctx.session.state.get('key')`. `output_key` on `LlmAgent` is a convenient auto-setter.
    *   **Best for**: Passing intermediate results, shared configurations, and flags in pipelines (Sequential, Loop agents).

2.  **LLM-Driven Delegation (`transfer_to_agent`)**: A `LlmAgent` can dynamically hand over control to another agent based on its reasoning.
    *   **Mechanism**: The LLM generates a special `transfer_to_agent` function call. The ADK framework intercepts this, routes the next turn to the target agent.
    *   **Prerequisites**:
        *   The initiating `LlmAgent` needs `instruction` to guide delegation and `description` of the target agent(s).
        *   Target agents need clear `description`s to help the LLM decide.
        *   Target agent must be discoverable within the current agent's hierarchy (direct `sub_agent` or a descendant).
    *   **Configuration**: Can be enabled/disabled via `disallow_transfer_to_parent` and `disallow_transfer_to_peers` on `LlmAgent`.

    ```python
    sales_agent = Agent(name="SalesAgent", description="Handles new sales inquiries and product information.")
    support_agent = Agent(name="SupportAgent", description="Assists with post-purchase support and technical issues.")

    customer_router = Agent(
        name="CustomerRouter",
        model="gemini-2.0-flash",
        instruction="You are a customer service router. If the user asks about purchasing, new products, or pricing, delegate to 'SalesAgent'. If they ask for help with an existing order, a technical problem, or a refund, delegate to 'SupportAgent'. For anything else, say you cannot help.",
        sub_agents=[sales_agent, support_agent] # Establishes hierarchy for delegation
    )
    ```

3.  **Explicit Invocation (`AgentTool`)**: An `LlmAgent` can treat another `BaseAgent` instance as a callable tool.
    *   **Mechanism**: Wrap the target agent (`target_agent`) in `AgentTool(agent=target_agent)` and add it to the calling `LlmAgent`'s `tools` list. The `AgentTool` generates a `FunctionDeclaration` for the LLM. When called, `AgentTool` runs the target agent and returns its final response as the tool result.
    *   **Best for**: Hierarchical task decomposition, where a higher-level agent needs a specific output from a lower-level agent.

    ```python
    from google.adk.tools.agent_tool import AgentTool

    # Lower-level agent for content generation
    creative_writer_agent = Agent(
        name="CreativeWriter",
        model="gemini-2.0-flash",
        instruction="Write a short, engaging story about the topic provided. Output ONLY the story text."
    )

    # Higher-level agent uses the writer as a tool
    blog_post_generator = Agent(
        name="BlogPostGenerator",
        model="gemini-2.0-flash",
        instruction="Generate a blog post. First, draft a story using the 'CreativeWriter' tool with the user's topic. Then, add an introduction and conclusion to the story to make it a full blog post.",
        tools=[AgentTool(agent=creative_writer_agent)] # CreativeWriter is now a callable tool
    )
    ```

### 4.3 Common Multi-Agent Patterns

*   **Coordinator/Dispatcher**: A central agent routes requests to specialized sub-agents (often via LLM-driven delegation).
*   **Sequential Pipeline**: `SequentialAgent` orchestrates a fixed sequence of tasks, passing data via shared state.
*   **Parallel Fan-Out/Gather**: `ParallelAgent` runs concurrent tasks, followed by a final agent that synthesizes results from state.
*   **Hierarchical Task Decomposition**: High-level agents break down complex problems, delegating sub-tasks to lower-level agents (often via `AgentTool`).
*   **Review/Critique (Generator-Critic)**: `SequentialAgent` with a generator followed by a critic, often in a `LoopAgent` for iterative refinement.
*   **Iterative Refinement**: `LoopAgent` with agents that refine state until a condition is met (using `escalate`).
*   **Human-in-the-Loop**: Integrates human approval or intervention via custom tools that interact with external systems.

---

## 5. Building Custom Agents (`BaseAgent`)

For unique orchestration logic that doesn't fit standard workflow agents, inherit directly from `BaseAgent`.

### 5.1 When to Use Custom Agents

*   **Complex Conditional Logic**: `if/else` branching based on multiple state variables or complex external checks.
*   **Dynamic Agent Selection**: Choosing which sub-agent to run based on runtime evaluation or external data.
*   **Direct External Integrations**: Calling external APIs, databases, or custom libraries directly within the orchestration flow.
*   **Custom Loop/Retry Logic**: More sophisticated iteration patterns than `LoopAgent`.
*   **Interleaving Logic**: Mixing LLM calls, tool calls, and custom Python logic in a tightly controlled sequence.

### 5.2 Implementing `_run_async_impl`

This is the core asynchronous method you must override.

```python
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from typing import AsyncGenerator
import asyncio

class DynamicWorkflowAgent(BaseAgent):
    def __init__(self, name: str, data_extractor: LlmAgent, validator: LlmAgent, processor: LlmAgent):
        super().__init__(name=name, sub_agents=[data_extractor, validator, processor])
        self.data_extractor = data_extractor
        self.validator = validator
        self.processor = processor

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        print(f"[{self.name}] Starting dynamic workflow.")
        
        # Initial data extraction
        print(f"[{self.name}] Running data extractor...")
        async for event in self.data_extractor.run_async(ctx):
            yield event
        
        extracted_data = ctx.session.state.get("extracted_data")
        if not extracted_data:
            yield Event(author=self.name, content={"parts": [{"text": "Extraction failed, workflow aborted."}]})
            return

        # Loop for validation and re-extraction if needed
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"[{self.name}] Running validator (attempt {attempt + 1})...")
            async for event in self.validator.run_async(ctx):
                yield event
            
            validation_status = ctx.session.state.get("validation_status")
            if validation_status == "valid":
                print(f"[{self.name}] Data validated. Proceeding to processor.")
                break
            else:
                print(f"[{self.name}] Data invalid. Attempting re-extraction.")
                if attempt < max_attempts - 1:
                    async for event in self.data_extractor.run_async(ctx): # Re-run extractor
                        yield event
                    extracted_data = ctx.session.state.get("extracted_data") # Update data
                    if not extracted_data:
                        yield Event(author=self.name, content={"parts": [{"text": "Re-extraction failed, workflow aborted."}]})
                        return
                else:
                    yield Event(author=self.name, content={"parts": [{"text": "Max validation attempts reached. Workflow aborted."}]})
                    return

        # Final processing
        print(f"[{self.name}] Running processor...")
        async for event in self.processor.run_async(ctx):
            yield event
        
        print(f"[{self.name}] Workflow completed.")
        # Final response is likely yielded by the last sub-agent or a final summary by this agent
```
*   **Asynchronous Generator**: `async def ... yield Event`. This allows pausing and resuming execution, handing control back to the `Runner` to commit events.
*   **`ctx: InvocationContext`**: Provides access to all session state (`ctx.session.state`), services (`ctx.artifact_service`), and control flags (`ctx.end_invocation`).
*   **Calling Sub-Agents**: Use `async for event in self.sub_agent_instance.run_async(ctx): yield event`.
*   **State Management**: `ctx.session.state` is read/writeable.
*   **Control Flow**: Use standard Python `if/else`, `for/while` loops, `try/except` for complex logic.

---

## 6. Models: Gemini, LiteLLM, and Vertex AI

ADK's model flexibility allows integrating various LLMs for different needs.

### 6.1 Google Gemini Models (AI Studio & Vertex AI)

*   **Default Integration**: Native support via `google-genai` library.
*   **AI Studio (Easy Start)**:
    *   Set `GOOGLE_API_KEY="YOUR_API_KEY"` (environment variable).
    *   Set `GOOGLE_GENAI_USE_VERTEXAI="False"`.
    *   Model strings: `"gemini-2.0-flash"`, `"gemini-1.5-pro-latest"`, etc.
*   **Vertex AI (Production)**:
    *   Authenticate via `gcloud auth application-default login` (recommended).
    *   Set `GOOGLE_CLOUD_PROJECT="YOUR_PROJECT_ID"`, `GOOGLE_CLOUD_LOCATION="your-region"` (environment variables).
    *   Set `GOOGLE_GENAI_USE_VERTEXAI="True"`.
    *   Model strings: `"gemini-2.0-flash"`, `"gemini-1.5-pro-latest"`, or full Vertex AI endpoint resource names for specific deployments.

### 6.2 Other Cloud & Proprietary Models via LiteLLM

`LiteLlm` provides a unified interface to 100+ LLMs (OpenAI, Anthropic, Cohere, etc.).

*   **Installation**: `pip install litellm`
*   **API Keys**: Set environment variables as required by LiteLLM (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
*   **Usage**:
    ```python
    from google.adk.models.lite_llm import LiteLlm
    agent_openai = Agent(model=LiteLlm(model="openai/gpt-4o"), ...)
    agent_claude = Agent(model=LiteLlm(model="anthropic/claude-3-haiku-20240307"), ...)
    ```

### 6.3 Open & Local Models via LiteLLM (Ollama, vLLM)

For self-hosting, cost savings, privacy, or offline use.

*   **Ollama Integration**: Run Ollama locally (`ollama run <model>`).
    ```bash
    export OLLAMA_API_BASE="http://localhost:11434" # Ensure Ollama server is running
    ```
    ```python
    from google.adk.models.lite_llm import LiteLlm
    # Use 'ollama_chat' provider for tool-calling capabilities with Ollama models
    agent_ollama = Agent(model=LiteLlm(model="ollama_chat/llama3:instruct"), ...)
    ```
    *   **Model Selection**: Ensure the Ollama model supports `tools` in its capabilities.
    *   **Prompt Templates**: May need to fine-tune Ollama's model templates for optimal tool calling.

*   **Self-Hosted Endpoint (e.g., vLLM)**:
    ```python
    from google.adk.models.lite_llm import LiteLlm
    api_base_url = "https://your-vllm-endpoint.example.com/v1"
    agent_vllm = Agent(
        model=LiteLlm(
            model="your-model-name-on-vllm",
            api_base=api_base_url,
            # Add authentication if your endpoint requires it
            extra_headers={"Authorization": "Bearer YOUR_TOKEN"},
            # api_key="YOUR_API_KEY"
        ),
        ...
    )
    ```

### 6.4 Customizing LLM API Clients

For `google-genai` (used by Gemini models), you can configure the underlying client.

```python
import os
from google.genai import configure as genai_configure
from google.adk.agents import Agent

# Example: Configure client for higher timeouts/retries (global for google-genai)
genai_configure.use_defaults(
    timeout=60, # seconds
    client_options={"api_key": os.getenv("GOOGLE_API_KEY")}, # If not using ADC
    # transport='grpc', # Optional: for gRPC transport
    # client_class=MyCustomClient # Advanced: provide custom client class
)

# Your agent will now use this configured client
agent = Agent(model="gemini-2.0-flash", ...)
```

---

## 7. Tools: The Agent's Capabilities

Tools extend an agent's abilities beyond text generation.

### 7.1 Defining Function Tools: Principles & Best Practices

*   **Signature**: `def my_tool(param1: Type, param2: Type, tool_context: ToolContext) -> dict:`
    *   Last parameter `tool_context: ToolContext` is optional but highly recommended.
*   **Function Name**: Descriptive verb-noun (e.g., `schedule_meeting`, `get_weather_report`). This becomes the name the LLM uses to call the tool.
*   **Parameters**:
    *   Clear, descriptive names (e.g., `city` not `c`).
    *   **Required**: Type hints (`str`, `int`, `list`, `dict`, `bool`).
    *   **Crucial**: **NO DEFAULT VALUES** for parameters that the LLM needs to provide. All inputs must be explicitly decided by the LLM or pre-filled.
    *   All types must be JSON-serializable.
*   **Return Type**: **Must** be a `dict` (JSON-serializable).
    *   **Best Practice**: Include a `'status'` key (`'success'`, `'error'`, `'pending'`) for clarity to the LLM.
    *   Provide rich, structured data that the LLM can easily consume and understand.
*   **Docstring**: **THE MOST CRITICAL COMPONENT**. The LLM relies heavily on this.
    *   **Purpose**: What does the tool *do*?
    *   **When to Use**: Guide the LLM on appropriate scenarios.
    *   **Arguments**: Describe *each* parameter clearly (type, what it represents).
    *   **Return Value**: Describe the structure of the returned `dict` (keys, possible values, error formats).
    *   **AVOID**: Mentioning `tool_context` in the docstring. ADK injects it; the LLM doesn't need to know about it.

    ```python
    def calculate_compound_interest(
        principal: float,
        rate: float,
        years: int,
        compounding_frequency: int,
        tool_context: ToolContext # Important: last parameter, optional
    ) -> dict:
        """Calculates the future value of an investment with compound interest.

        Use this tool when the user asks to calculate the future value of an
        investment given a principal amount, interest rate, number of years,
        and how often the interest is compounded per year.

        Args:
            principal (float): The initial amount of money invested.
            rate (float): The annual interest rate (e.g., 0.05 for 5%).
            years (int): The number of years the money is invested.
            compounding_frequency (int): The number of times the interest is
                                         compounded per year (e.g., 1 for annually,
                                         4 for quarterly, 12 for monthly).
            
        Returns:
            dict: A dictionary containing the calculation result.
                  - 'status' (str): "success" or "error".
                  - 'future_value' (float, optional): The calculated future value.
                  - 'error_message' (str, optional): Description of error, if any.
        """
        try:
            # Perform calculation
            amount = principal * (1 + rate / compounding_frequency)**(compounding_frequency * years)
            tool_context.state['last_calculation_result'] = amount # Example: write to state
            return {"status": "success", "future_value": round(amount, 2)}
        except Exception as e:
            return {"status": "error", "error_message": str(e)}
    ```

### 7.2 The `ToolContext` Object: Accessing Runtime Information

`ToolContext` (passed as the last argument to tool functions) is the gateway for tools to interact with the broader ADK runtime.

*   `tool_context.state`: Read and write to the current `Session`'s `state` dictionary. Changes are automatically tracked for persistence.
    ```python
    current_setting = tool_context.state.get("user:theme", "light")
    tool_context.state["last_activity_timestamp"] = time.time()
    ```
*   `tool_context.actions`: Modify the `EventActions` object associated with the tool's result event.
    *   `tool_context.actions.skip_summarization = True`: Instructs ADK to pass the tool's raw `dict` output directly to the LLM (or UI) without LLM-based summarization.
    *   `tool_context.actions.transfer_to_agent = "TargetAgentName"`: Forces a transfer of control to another agent.
    *   `tool_context.actions.escalate = True`: Signals a `LoopAgent` to terminate its loop.
    *   `tool_context.actions.requested_auth_configs`: Used internally for auth flows.
*   `tool_context.list_artifacts()`: Returns a list of filenames available in the current session/user scope.
*   `tool_context.load_artifact(filename: str, version: Optional[int] = None)`: Retrieves a specific artifact (or latest version) as a `genai_types.Part`.
*   `tool_context.save_artifact(filename: str, artifact: genai_types.Part)`: Saves a new version of an artifact.
*   `tool_context.search_memory(query: str)`: Queries the configured `MemoryService`.
*   `tool_context.request_credential(auth_config: AuthConfig)`: Initiates an authentication flow (for custom tools).
*   `tool_context.get_auth_response(auth_config: AuthConfig)`: Retrieves authentication credentials after a flow is completed.
*   `tool_context.function_call_id`: The unique ID of the specific LLM function call that triggered this tool. Useful for linking auth.

### 7.3 All Tool Types & Their Usage

ADK supports a diverse ecosystem of tools.

1.  **`FunctionTool`**: Wraps any Python callable (function or method).
    *   **Usage**: `FunctionTool(func=my_python_function)` or simply `tools=[my_python_function]` (ADK auto-wraps).
    *   **Core**: The most common way to extend capabilities.

2.  **`LongRunningFunctionTool`**: For `async` functions that `yield` intermediate results (useful for streaming progress or human-in-the-loop).
    *   **Signature**: `async def my_streaming_tool(...) -> AsyncGenerator[Dict, None]:`
    *   **Usage**: `LongRunningFunctionTool(func=my_async_generator_function)`.
    *   **Note**: Requires `streaming_mode` to be enabled (`SSE` or `BIDI`) in `RunConfig`.

3.  **`AgentTool`**: Wraps another `BaseAgent` instance, allowing it to be called as a tool.
    *   **Usage**: `AgentTool(agent=another_agent_instance)`.
    *   **Core**: Enables hierarchical task decomposition and complex nested workflows.

4.  **`OpenAPIToolset`**: Automatically generates `RestApiTool` instances from an OpenAPI (Swagger) v3 specification.
    *   **Usage**: `OpenAPIToolset(spec_str=json_or_yaml_string, spec_str_type='json', auth_scheme=..., auth_credential=...)`.
    *   **Core**: Integrates with REST APIs without manual tool definition. Handles request construction and response parsing.

5.  **`MCPToolset`**: Connects to an external Model Context Protocol (MCP) server to consume its exposed tools.
    *   **Usage**: `MCPToolset(connection_params=StdioServerParameters(...) | SseServerParams(...), tool_filter=...)`.
    *   **Core**: Interoperability with MCP-compliant services (e.g., file systems, Google Maps, databases via MCP Toolbox).

6.  **Built-in Tools**: Provided by ADK for common functionalities.
    *   `google_search`: Performs Google searches.
    *   `BuiltInCodeExecutor`: Allows the agent to write and execute Python code in a sandboxed environment.
    *   `VertexAiSearchTool`: Integrates with Vertex AI Search datastores for RAG.
    *   **Usage**: `tools=[google_search]`.

7.  **Third-Party Tool Wrappers**: Seamlessly integrate tools from other agent frameworks.
    *   `LangchainTool`: Wraps a LangChain tool.
        *   **Usage**: `LangchainTool(tool=LangchainToolInstance)`.
    *   `CrewaiTool`: Wraps a CrewAI tool.
        *   **Usage**: `CrewaiTool(name="MyToolName", description="MyToolDescription", tool=CrewaiToolInstance)`. (Requires explicit `name` and `description` for LLM).

8.  **Google Cloud Tools**: Specialized integrations for Google Cloud services.
    *   `ApiHubToolset`: Exposes APIs from Apigee API Hub as tools.
    *   `ApplicationIntegrationToolset`: Integrates with Application Integration workflows and connectors.
    *   `toolbox_core.ToolboxSyncClient`: For MCP Toolbox for Databases.
    *   **Usage**: Similar to `OpenAPIToolset` or direct instantiation, with specific GCP resource names.

---

## 8. Context, State, and Memory Management

Effective context management is crucial for coherent, multi-turn conversations.

### 8.1 The `Session` Object & `SessionService`

*   **`Session`**: The container for a single, ongoing conversation.
    *   Properties: `id` (unique session ID), `app_name`, `user_id`, `state` (mutable dictionary), `events` (chronological list of `Event` objects), `last_update_time`.
*   **`SessionService`**: Manages the lifecycle of `Session` objects.
    *   `create_session(app_name, user_id, session_id, state={})`: Initializes a new session.
    *   `get_session(app_name, user_id, session_id)`: Retrieves an existing session.
    *   `append_event(session, event)`: **The ONLY way to reliably update session state and history.** This method processes `event.actions.state_delta` and `event.actions.artifact_delta`.
    *   `list_sessions(app_name, user_id)`: Lists sessions for a user.
    *   `delete_session(app_name, user_id, session_id)`: Deletes a session.
*   **Implementations**:
    *   `InMemorySessionService`: Default, ephemeral (lost on restart). For local dev.
    *   `VertexAiSessionService`: Persistent, scalable, managed by Vertex AI Agent Engine. For production.
    *   `DatabaseSessionService`: Persistent storage in a relational database (SQLite, PostgreSQL, MySQL). For self-managed persistence.

### 8.2 `State`: The Conversational Scratchpad

A mutable dictionary within `session.state` for short-term, dynamic data.

*   **Values**: Must be JSON-serializable (strings, numbers, booleans, lists/dicts of these). Avoid complex Python objects.
*   **Update Mechanism**: Always update via `context.state` (in callbacks/tools) or `LlmAgent.output_key`. These changes are batched into `Event.actions.state_delta` and atomically committed by `session_service.append_event()`.
    *   **DO NOT** directly modify `session.state` obtained via `session_service.get_session()` outside the `Runner`'s event loop (e.g., `session.state['key'] = value` won't persist).
*   **Prefixes for Scope**:
    *   **(No prefix)**: Session-specific (e.g., `session.state['booking_step']`).
    *   `user:`: Persistent for a `user_id` across *all* their sessions (e.g., `session.state['user:preferred_currency']`). Requires persistent `SessionService`.
    *   `app:`: Persistent for `app_name` across *all* users and sessions (e.g., `session.state['app:global_config_version']`). Requires persistent `SessionService`.
    *   `temp:`: Volatile, *only* for the current `Invocation` turn. Guaranteed to be discarded after the `Runner` processes the event. Useful for intermediate data not meant for persistence.

    ```python
    # Example state usage in a tool (via ToolContext)
    def update_settings(key: str, value: Any, tool_context: ToolContext) -> dict:
        """Updates a user or session setting."""
        if key.startswith("user:"):
            tool_context.state[key] = value # Stores as user-scoped state
            print(f"Set user-scoped state: {key}={value}")
            return {"status": "success", "message": f"User setting '{key}' updated."}
        else:
            tool_context.state[key] = value # Stores as session-scoped state
            print(f"Set session-scoped state: {key}={value}")
            return {"status": "success", "message": f"Session setting '{key}' updated."}
    ```

### 8.3 `Memory`: Long-Term Knowledge & Retrieval

For knowledge beyond a single conversation, spanning many past interactions or external data.

*   **`BaseMemoryService`**: Defines the interface for long-term knowledge.
    *   `add_session_to_memory(session: Session)`: Ingests a session's content into memory.
    *   `search_memory(app_name, user_id, query, limit=...)`: Retrieves relevant snippets.
*   **Implementations**:
    *   `InMemoryMemoryService`: Ephemeral, simple keyword search. For dev/testing.
    *   `VertexAiRagMemoryService`: Persistent, scalable, semantic search via Vertex AI RAG Corpus. For production.
*   **Usage**: Agents interact via tools. The built-in `load_memory` tool or a custom tool using `tool_context.search_memory()`.

    ```python
    from google.adk.memory import InMemoryMemoryService
    from google.adk.tools import load_memory # Built-in tool

    memory_service = InMemoryMemoryService()
    my_agent = Agent(
        name="KnowledgeAgent",
        model="gemini-2.0-flash",
        instruction="Use the 'load_memory' tool to recall past discussions.",
        tools=[load_memory]
    )
    # Runner needs to be configured with memory_service:
    # runner = Runner(..., memory_service=memory_service)
    # To add memory after a session:
    # await memory_service.add_session_to_memory(completed_session)
    ```

### 8.4 `Artifacts`: Binary Data Management

For named, versioned binary data (files, images, audio, PDFs).

*   **Representation**: `google.genai.types.Part` (specifically its `inline_data` field, a `Blob` with `data: bytes` and `mime_type: str`).
*   **`BaseArtifactService`**: Manages storage and retrieval.
    *   `save_artifact(app_name, user_id, session_id, filename, artifact_part)`: Stores, auto-versions.
    *   `load_artifact(app_name, user_id, session_id, filename, version=None)`: Retrieves.
    *   `list_artifact_keys(app_name, user_id, session_id)`: Lists filenames.
*   **Implementations**:
    *   `InMemoryArtifactService`: Ephemeral. For local dev.
    *   `GcsArtifactService`: Persistent via Google Cloud Storage. For production.
*   **Usage**: Primarily via `CallbackContext` or `ToolContext` methods.
    *   `context.save_artifact(filename, artifact_part)`: Saves.
    *   `context.load_artifact(filename, version=None)`: Loads.
    *   `context.list_artifacts()`: Lists.
*   **Versioning**: `save_artifact` returns the new version number. `load_artifact` without `version` gets the latest.
*   **Namespacing**: `filename` can be prefixed with `"user:"` (e.g., `"user:profile.jpg"`) to scope to a user across sessions. Otherwise, it's session-specific.

    ```python
    from google.adk.artifacts import InMemoryArtifactService
    from google.genai import types as genai_types

    # Assume runner is configured with InMemoryArtifactService
    # In a tool or callback:
    async def process_image(image_bytes: bytes, tool_context: ToolContext) -> dict:
        image_part = genai_types.Part(
            inline_data=genai_types.Blob(mime_type="image/png", data=image_bytes)
        )
        filename = "user_uploaded_photo.png"
        version = await tool_context.save_artifact(filename, image_part)
        print(f"Image saved as {filename}, version {version}.")

        # Later, to load:
        loaded_image_part = await tool_context.load_artifact(filename)
        # process loaded_image_part.inline_data.data
        
        # To list available images:
        available_images = await tool_context.list_artifacts()
        return {"status": "success", "message": f"Processed image. Available: {available_images}"}
    ```

---

## 9. Runtime, Events, and Execution Flow

The `Runner` is the central orchestrator of an ADK application.

### 9.1 The `Runner`: The Orchestrator

*   **Role**: Manages the agent's lifecycle, the event loop, and coordinates with services.
*   **Entry Point**: `runner.run_async(user_id, session_id, new_message)` for asynchronous execution. `runner.run()` is a synchronous wrapper.
*   **Configuration**: Initialized with a `BaseAgent`, `app_name`, `SessionService`, and optional `ArtifactService`/`MemoryService`.

    ```python
    from google.adk.runners import Runner
    from google.adk.agents import Agent
    from google.adk.sessions import InMemorySessionService
    from google.adk.artifacts import InMemoryArtifactService
    from google.adk.memory import InMemoryMemoryService

    # Assuming 'my_root_agent' is defined
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()
    memory_service = InMemoryMemoryService()

    my_runner = Runner(
        agent=my_root_agent,
        app_name="my_application_name",
        session_service=session_service,
        artifact_service=artifact_service, # Optional
        memory_service=memory_service,     # Optional
    )
    ```

### 9.2 The Event Loop: Core Execution Flow

The `Runner` operates on an `Event` loop:

1.  **User Input**: `Runner` receives `new_message` (a `genai_types.Content` object), packages it as a `user` `Event`, and appends to session history.
2.  **Agent Execution**: `Runner` calls `agent.run_async(invocation_context)`.
3.  **Agent Yields Event**: Your agent/tool logic executes, and when it needs to communicate (e.g., provide text, request a tool call, signal a state change), it `yield`s an `Event`.
4.  **Execution Pauses**: Agent/tool code pauses immediately after `yield`.
5.  **Runner Processes Event**: `Runner` receives the `Event`.
    *   It applies `Event.actions.state_delta` to `session.state` (via `SessionService.append_event`).
    *   It processes `Event.actions.artifact_delta`.
    *   It then `yield`s the processed `Event` to the client application.
6.  **Execution Resumes**: `Runner` signals the agent/tool to resume. The agent/tool's code now sees the state reflecting changes from the *last yielded event*.
7.  **Repeat**: This cycle continues until the agent's `run_async` generator is exhausted.

### 9.3 `Event` Object: The Communication Backbone

`Event` objects carry all information and signals.

*   `Event.author`: Source of the event (`'user'`, agent name, `'system'`).
*   `Event.invocation_id`: Unique ID for the entire `run_async` call.
*   `Event.id`: Unique ID for *this specific* event instance.
*   `Event.timestamp`: When the event was created.
*   `Event.content`: The primary payload (text, function calls, function responses, blobs).
    *   `event.content.parts[0].text`: For text messages.
    *   `event.get_function_calls()`: List of `FunctionCall` objects (`.name`, `.args`).
    *   `event.get_function_responses()`: List of `FunctionResponse` objects (`.name`, `.response`).
    *   `event.content.parts[0].inline_data`: For binary data (`Blob`).
*   `Event.actions`: Signals side effects or control flow changes.
    *   `event.actions.state_delta`: `{key: value}` changes applied to `session.state`.
    *   `event.actions.artifact_delta`: `{filename: version}` info for saved artifacts.
    *   `event.actions.transfer_to_agent`: Name of agent to transfer to.
    *   `event.actions.escalate`: `bool` flag for `LoopAgent` termination.
    *   `event.actions.skip_summarization`: `bool` flag for tool results.
*   `Event.partial`: `bool` for streaming text (more chunks coming).
*   `Event.is_final_response()`: Helper to identify the complete, displayable message for the turn.
*   `Event.error_code`, `Event.error_message`: For errors.

### 9.4 Asynchronous Programming (Python Specific)

ADK is built on `asyncio`.

*   **`async def` / `await`**: All I/O-bound operations (LLM calls, tool executions, service interactions) within ADK are asynchronous.
*   **`async for`**: Used to consume event streams from `runner.run_async()` or `LongRunningFunctionTool`s.
*   **`asyncio.run()`**: Used to run a top-level `async def` function in a synchronous context (e.g., main script, testing).
*   **`asyncio.create_task()`**: For concurrent execution of independent `async` coroutines.

    ```python
    import asyncio
    from google.adk.runners import Runner
    # ... setup runner ...

    async def main_interaction():
        print("Starting interaction...")
        user_query = "What's the weather in London?"
        
        # Consuming the async generator from runner.run_async
        async for event in my_runner.run_async(user_id="user1", session_id="session1", new_message={"parts": [{"text": user_query}]}):
            if event.is_final_response() and event.content and event.content.parts:
                print(f"Agent's final response: {event.content.parts[0].text}")
                break # Stop after final response
        print("Interaction complete.")

    if __name__ == "__main__":
        try:
            asyncio.run(main_interaction())
        except RuntimeError as e:
            if "cannot run from a running event loop" in str(e):
                print("Already in an async loop (e.g., Jupyter). Run with `await main_interaction()`.")
            else:
                raise
    ```

---

## 10. Control Flow with Callbacks

Callbacks are user-defined functions that intercept and control agent execution at specific points.

### 10.1 Callback Mechanism: Interception & Control

*   **Definition**: A standard Python function assigned to an agent's `callback` parameter (e.g., `before_model_callback=my_func`).
*   **Context**: Receives a `CallbackContext` (or `ToolContext` for tool-related callbacks) providing runtime information.
*   **Return Value**: **Crucially determines flow.**
    *   `return None`: Allow the default action to proceed (or use the original result for `after_*` callbacks).
    *   `return <Specific Object>`: **Override** the default action/result. ADK uses the returned object and skips the next step or replaces the previous result.

### 10.2 Types of Callbacks

1.  **Agent Lifecycle Callbacks (`CallbackContext`)**:
    *   `before_agent_callback(context: CallbackContext)`:
        *   When: Just before `agent._run_async_impl` starts.
        *   Purpose: Pre-run validation, setup, logging.
        *   Return `genai_types.Content`: Skips agent's execution, returns this content as final response.
    *   `after_agent_callback(context: CallbackContext)`:
        *   When: Just after `agent._run_async_impl` completes.
        *   Purpose: Post-run processing, logging, modifying final output.
        *   Return `genai_types.Content`: Replaces agent's original final response.

2.  **LLM Interaction Callbacks (`CallbackContext`)**:
    *   `before_model_callback(context: CallbackContext, llm_request: LlmRequest)`:
        *   When: Just before `LlmRequest` is sent to LLM.
        *   Purpose: Input validation/guardrails, dynamic prompt injection, caching.
        *   Return `LlmResponse`: Skips LLM call, uses this response.
    *   `after_model_callback(context: CallbackContext, llm_response: LlmResponse)`:
        *   When: Just after `LlmResponse` is received from LLM.
        *   Purpose: Output sanitization, reformatting, logging.
        *   Return `LlmResponse`: Replaces LLM's original response.

3.  **Tool Execution Callbacks (`ToolContext`)**:
    *   `before_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext)`:
        *   When: Just before `tool.run_async` is invoked.
        *   Purpose: Tool argument validation, authorization, caching.
        *   Return `dict`: Skips tool execution, uses this dict as tool result.
    *   `after_tool_callback(tool: BaseTool, args: dict, tool_context: ToolContext, tool_response: dict)`:
        *   When: Just after `tool.run_async` completes.
        *   Purpose: Post-processing tool results, logging, modifying data for LLM.
        *   Return `dict`: Replaces tool's original result.

### 10.3 Callback Best Practices

*   **Keep Focused**: Each callback for a single purpose.
*   **Performance**: Avoid blocking I/O or heavy computation. Callbacks run synchronously within the event loop.
*   **Error Handling**: Use `try...except` in callbacks to prevent crashes.
*   **State Management**: Use `context.state` (or `output_key`). Be deliberate about state keys.
*   **Clarity**: Descriptive names, clear docstrings (excluding `context` param).
*   **Test Thoroughly**: Unit test callbacks with mock contexts.

---

## 11. Authentication for Tools

Enabling agents to securely access protected external resources.

### 11.1 Core Concepts: `AuthScheme` & `AuthCredential`

*   **`AuthScheme`**: Defines *how* an API expects authentication (e.g., `APIKey`, `HTTPBearer`, `OAuth2`, `OpenIdConnectWithConfig`).
*   **`AuthCredential`**: Holds *initial* information to *start* the auth process (e.g., API key value, OAuth client ID/secret).

### 11.2 Interactive OAuth/OIDC Flows

When a tool requires user interaction (OAuth consent), ADK pauses and signals your `Agent Client` application.

1.  **Detect Auth Request**: `runner.run_async()` yields an event with a special `adk_request_credential` function call.
2.  **Redirect User**: Extract `auth_uri` from `auth_config` in the event. Your client app redirects the user's browser to this `auth_uri` (appending `redirect_uri`).
3.  **Handle Callback**: Your client app has a pre-registered `redirect_uri` to receive the user after authorization. It captures the full callback URL (containing `authorization_code`).
4.  **Send Auth Result to ADK**: Your client prepares a `FunctionResponse` for `adk_request_credential`, setting `auth_config.exchanged_auth_credential.oauth2.auth_response_uri` to the captured callback URL.
5.  **Resume Execution**: `runner.run_async()` is called again with this `FunctionResponse`. ADK performs the token exchange, stores the access token, and retries the original tool call.

### 11.3 Custom Tool Authentication

If building a `FunctionTool` that needs authentication:

1.  **Check for Cached Creds**: `tool_context.state.get("my_token_cache_key")`.
2.  **Check for Auth Response**: `tool_context.get_auth_response(my_auth_config)`.
3.  **Initiate Auth**: If no creds, call `tool_context.request_credential(my_auth_config)` and return a pending status. This triggers the external flow.
4.  **Cache Credentials**: After obtaining, store in `tool_context.state`.
5.  **Make API Call**: Use the valid credentials (e.g., `google.oauth2.credentials.Credentials`).

---

## 12. Deployment Strategies

From local dev to production.

### 12.1 Local Development & Testing (`adk web`, `adk run`, `adk api_server`)

*   **`adk web`**: Launches a local web UI for interactive chat, session inspection, and visual tracing.
    ```bash
    adk web /path/to/your/project_root
    ```
*   **`adk run`**: Command-line interactive chat.
    ```bash
    adk run /path/to/your/agent_folder
    ```
*   **`adk api_server`**: Launches a local FastAPI server exposing `/run`, `/run_sse`, `/list-apps`, etc., for API testing with `curl` or client libraries.
    ```bash
    adk api_server /path/to/your/project_root
    ```

### 12.2 Vertex AI Agent Engine

Fully managed, scalable service for ADK agents on Google Cloud.

*   **Features**: Auto-scaling, session management, observability integration.
*   **Deployment**: Use `vertexai.agent_engines.create()`.
    ```python
    from vertexai.preview import reasoning_engines # or agent_engines directly in later versions
    
    # Wrap your root_agent for deployment
    app_for_engine = reasoning_engines.AdkApp(agent=root_agent, enable_tracing=True)
    
    # Deploy
    remote_app = agent_engines.create(
        agent_engine=app_for_engine,
        requirements=["google-cloud-aiplatform[adk,agent_engines]"],
        display_name="My Production Agent"
    )
    print(remote_app.resource_name) # projects/PROJECT_NUM/locations/REGION/reasoningEngines/ID
    ```
*   **Interaction**: Use `remote_app.stream_query()`, `create_session()`, etc.

### 12.3 Cloud Run

Serverless container platform for custom web applications.

*   **Deployment**:
    1.  Create a `Dockerfile` for your FastAPI app (using `google.adk.cli.fast_api.get_fast_api_app`).
    2.  Use `gcloud run deploy --source .`.
    3.  Alternatively, `adk deploy cloud_run` (simpler, opinionated).
*   **Example `main.py`**:
    ```python
    import os
    from fastapi import FastAPI
    from google.adk.cli.fast_api import get_fast_api_app

    # Ensure your agent_folder (e.g., 'my_first_agent') is in the same directory as main.py
    app: FastAPI = get_fast_api_app(
        agents_dir=os.path.dirname(os.path.abspath(__file__)),
        session_db_url="sqlite:///./sessions.db", # In-container SQLite, for simple cases
        # For production: use a persistent DB (Cloud SQL) or VertexAiSessionService
        allow_origins=["*"],
        web=True # Serve ADK UI
    )
    # uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080))) # If running directly
    ```

### 12.4 Google Kubernetes Engine (GKE)

For maximum control, run your containerized agent in a Kubernetes cluster.

*   **Deployment**:
    1.  Build Docker image (`gcloud builds submit`).
    2.  Create Kubernetes Deployment and Service YAMLs.
    3.  Apply with `kubectl apply -f deployment.yaml`.
    4.  Configure Workload Identity for GCP permissions.

### 12.5 CI/CD Integration

*   Automate testing (`pytest`, `adk eval`) in CI.
*   Automate container builds and deployments (e.g., Cloud Build, GitHub Actions).
*   Use environment variables for secrets.

---

## 13. Evaluation and Safety

Critical for robust, production-ready agents.

### 13.1 Agent Evaluation (`adk eval`)

Systematically assess agent performance using predefined test cases.

*   **Evalset File (`.evalset.json`)**: Contains `eval_cases`, each with a `conversation` (user queries, expected tool calls, expected intermediate/final responses) and `session_input` (initial state).
    ```json
    {
      "eval_set_id": "weather_bot_eval",
      "eval_cases": [
        {
          "eval_id": "london_weather_query",
          "conversation": [
            {
              "user_content": {"parts": [{"text": "What's the weather in London?"}]},
              "final_response": {"parts": [{"text": "The weather in London is cloudy..."}]},
              "intermediate_data": {
                "tool_uses": [{"name": "get_weather", "args": {"city": "London"}}]
              }
            }
          ],
          "session_input": {"app_name": "weather_app", "user_id": "test_user", "state": {}}
        }
      ]
    }
    ```
*   **Running Evaluation**:
    *   `adk web`: Interactive UI for creating/running eval cases.
    *   `adk eval /path/to/agent_folder /path/to/evalset.json`: CLI execution.
    *   `pytest`: Integrate `AgentEvaluator.evaluate()` into unit/integration tests.
*   **Metrics**: `tool_trajectory_avg_score` (tool calls match expected), `response_match_score` (final response similarity using ROUGE). Configurable via `test_config.json`.

### 13.2 Safety & Guardrails

Multi-layered defense against harmful content, misalignment, and unsafe actions.

1.  **Identity and Authorization**:
    *   **Agent-Auth**: Tool acts with the agent's service account (e.g., `Vertex AI User` role). Simple, but all users share access level. Logs needed for attribution.
    *   **User-Auth**: Tool acts with the end-user's identity (via OAuth tokens). Reduces risk of abuse.
2.  **In-Tool Guardrails**: Design tools defensively. Tools can read policies from `tool_context.state` (set deterministically by developer) and validate model-provided arguments before execution.
    ```python
    def execute_sql(query: str, tool_context: ToolContext) -> dict:
        policy = tool_context.state.get("user:sql_policy", {})
        if not policy.get("allow_writes", False) and ("INSERT" in query.upper() or "DELETE" in query.upper()):
            return {"status": "error", "message": "Policy: Write operations are not allowed."}
        # ... execute query ...
    ```
3.  **Built-in Gemini Safety Features**:
    *   **Content Safety Filters**: Automatically block harmful content (CSAM, PII, hate speech, etc.). Configurable thresholds.
    *   **System Instructions**: Guide model behavior, define prohibited topics, brand tone, disclaimers.
4.  **Model and Tool Callbacks (LLM as a Guardrail)**: Use callbacks to inspect inputs/outputs.
    *   `before_model_callback`: Intercept `LlmRequest` before it hits the LLM. Block (return `LlmResponse`) or modify.
    *   `before_tool_callback`: Intercept tool calls (name, args) before execution. Block (return `dict`) or modify.
    *   **LLM-based Safety**: Use a cheap/fast LLM (e.g., Gemini Flash) in a callback to classify input/output safety.
        ```python
        def safety_checker_callback(context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
            # Use a separate, small LLM to classify safety
            safety_llm_agent = Agent(name="SafetyChecker", model="gemini-2.0-flash-001", instruction="Classify input as 'safe' or 'unsafe'. Output ONLY the word.")
            # Run the safety agent (might need a new runner instance or direct model call)
            # For simplicity, a mock:
            user_input = llm_request.contents[-1].parts[0].text
            if "dangerous_phrase" in user_input.lower():
                context.state["safety_violation"] = True
                return LlmResponse(content=genai_types.Content(parts=[genai_types.Part(text="I cannot process this request due to safety concerns.")]))
            return None
        ```
5.  **Sandboxed Code Execution**:
    *   `BuiltInCodeExecutor`: Uses secure, sandboxed execution environments.
    *   Vertex AI Code Interpreter Extension.
    *   If custom, ensure hermetic environments (no network, isolated).
6.  **Network Controls & VPC-SC**: Confine agent activity within secure perimeters (VPC Service Controls) to prevent data exfiltration.
7.  **Output Escaping in UIs**: Always properly escape LLM-generated content in web UIs to prevent XSS attacks and indirect prompt injections.

---

## 14. Debugging, Logging & Observability

*   **`adk web` UI**: Best first step. Provides visual trace, session history, and state inspection.
*   **Event Stream Logging**: Iterate `runner.run_async()` events and print relevant fields.
    ```python
    async for event in runner.run_async(...):
        print(f"[{event.author}] Event ID: {event.id}, Invocation: {event.invocation_id}")
        if event.content and event.content.parts:
            if event.content.parts[0].text:
                print(f"  Text: {event.content.parts[0].text[:100]}...")
            if event.get_function_calls():
                print(f"  Tool Call: {event.get_function_calls()[0].name} with {event.get_function_calls()[0].args}")
            if event.get_function_responses():
                print(f"  Tool Response: {event.get_function_responses()[0].response}")
        if event.actions:
            if event.actions.state_delta:
                print(f"  State Delta: {event.actions.state_delta}")
            if event.actions.transfer_to_agent:
                print(f"  TRANSFER TO: {event.actions.transfer_to_agent}")
        if event.error_message:
            print(f"  ERROR: {event.error_message}")
    ```
*   **Tool/Callback `print` statements**: Simple logging directly within your functions.
*   **Python `logging` module**: Integrate with standard logging frameworks.
*   **Tracing Integrations**: ADK supports OpenTelemetry (e.g., via Comet Opik) for distributed tracing.
    ```python
    # Example using Comet Opik integration (conceptual)
    # pip install comet_opik_adk
    # from comet_opik_adk import enable_opik_tracing
    # enable_opik_tracing() # Call at app startup
    # Then run your ADK app, traces appear in Comet workspace.
    ```
*   **Session History (`session.events`)**: Persisted for detailed post-mortem analysis.

---

## 15. Advanced I/O Modalities

ADK (especially with Gemini Live API models) supports richer interactions.

*   **Audio**: Input via `Blob(mime_type="audio/pcm", data=bytes)`, Output via `genai_types.SpeechConfig` in `RunConfig`.
*   **Vision (Images/Video)**: Input via `Blob(mime_type="image/jpeg", data=bytes)` or `Blob(mime_type="video/mp4", data=bytes)`. Models like `gemini-2.0-flash-exp` can process these.
*   **Multimodal Input in `Content`**:
    ```python
    multimodal_content = genai_types.Content(
        parts=[
            genai_types.Part(text="Describe this image:"),
            genai_types.Part(inline_data=genai_types.Blob(mime_type="image/jpeg", data=image_bytes))
        ]
    )
    ```
*   **Streaming Modalities**: `RunConfig.response_modalities=['TEXT', 'AUDIO']`.

---

## 16. Performance Optimization

*   **Model Selection**: Choose the smallest model that meets requirements (e.g., `gemini-2.0-flash` for simple tasks).
*   **Instruction Prompt Engineering**: Concise, clear instructions reduce tokens and improve accuracy.
*   **Tool Use Optimization**:
    *   Design efficient tools (fast API calls, optimize database queries).
    *   Cache tool results (e.g., using `before_tool_callback` or `tool_context.state`).
*   **State Management**: Store only necessary data in state to avoid large context windows.
*   **`include_contents='none'`**: For stateless utility agents, saves LLM context window.
*   **Parallelization**: Use `ParallelAgent` for independent tasks.
*   **Streaming**: Use `StreamingMode.SSE` or `BIDI` for perceived latency reduction.
*   **`max_llm_calls`**: Limit LLM calls to prevent runaway agents and control costs.

---

## 17. General Best Practices & Common Pitfalls

*   **Start Simple**: Begin with `LlmAgent`, mock tools, and `InMemorySessionService`. Gradually add complexity.
*   **Iterative Development**: Build small features, test, debug, refine.
*   **Modular Design**: Use agents and tools to encapsulate logic.
*   **Clear Naming**: Descriptive names for agents, tools, state keys.
*   **Error Handling**: Implement robust `try...except` blocks in tools and callbacks. Guide LLMs on how to handle tool errors.
*   **Testing**: Write unit tests for tools/callbacks, integration tests for agent flows (`pytest`, `adk eval`).
*   **Dependency Management**: Use virtual environments (`venv`) and `requirements.txt`.
*   **Secrets Management**: Never hardcode API keys. Use `.env` for local dev, environment variables or secret managers (Google Cloud Secret Manager) for production.
*   **Avoid Infinite Loops**: Especially with `LoopAgent` or complex LLM tool-calling chains. Use `max_iterations`, `max_llm_calls`, and strong instructions.
*   **Handle `None` & `Optional`**: Always check for `None` or `Optional` values when accessing nested properties (e.g., `event.content and event.content.parts and event.content.parts[0].text`).
*   **Immutability of Events**: Events are immutable records. If you need to change something *before* it's processed, do so in a `before_*` callback and return a *new* modified object.
*   **Understand `output_key` vs. direct `state` writes**: `output_key` is for the agent's *final conversational* output. Direct `tool_context.state['key'] = value` is for *any other* data you want to save.

This expanded cheatsheet provides a comprehensive overview of ADK's capabilities, design patterns, and best practices. It should serve as a powerful reference for building sophisticated and reliable AI agents.
