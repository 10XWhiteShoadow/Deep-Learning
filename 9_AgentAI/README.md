# 9. Agentic AI (Tool-Using LLM Agent)

## Model Overview
An **Agentic AI** is a system where a Large Language Model (LLM) acts as a reasoning engine that can autonomously plan, use tools, observe results, and iterate to complete complex goals — going beyond a simple prompt-response cycle.

## How Agentic AI Works
```
User Goal
    │
    ▼
┌─────────────┐
│    Agent    │ ←── LLM (Reasoning Engine)
│  (Planner)  │
└──────┬──────┘
       │ decides which tool to call
       ▼
┌─────────────┐
│    Tools    │  → Web Search, Calculator, Code Executor, APIs, DB...
└──────┬──────┘
       │ returns observation
       ▼
┌─────────────┐
│  Observation│ ←── Agent processes result
└──────┬──────┘
       │ repeat until task done
       ▼
   Final Answer
```

## Core Concepts

### ReAct Framework (Reason + Act)
The most popular paradigm for agents:
1. **Thought** — Agent reasons about what to do next
2. **Action** — Agent calls a tool with arguments
3. **Observation** — Tool returns a result
4. **Repeat** — Until goal is achieved

### Agent Memory Types
| Memory Type | Description | Example |
|-------------|-------------|---------|
| Short-term | In-context conversation history | Chat messages |
| Long-term | Vector DB / external storage | User preferences |
| Episodic | Past task traces & outcomes | Previous searches |
| Semantic | World knowledge | LLM's training data |

### Tool Types Agents Can Use
| Tool | Description |
|------|-------------|
| Search | Web search (SerpAPI, DuckDuckGo) |
| Calculator | Math operations |
| Code Interpreter | Run Python/JavaScript |
| Database | Query SQL/NoSQL |
| API Calls | Weather, stocks, news |
| File System | Read/write files |
| LLM | Call another AI model |

## Sample Project: Research Assistant Agent
The notebook implements a **Research & Math Agent** using:
- **Google Gemini / OpenAI GPT** as the LLM backbone
- **LangChain** agent framework
- **Tools**: Wikipedia search + Calculator + Python REPL
- **Memory**: Conversation buffer memory

## Architecture
```
User Query
    │
    ▼
LangChain Agent Executor
    ├── Tool: Wikipedia Search
    ├── Tool: Calculator (numexpr)
    ├── Tool: Python REPL
    └── Memory: ConversationBufferMemory
    │
    ▼
Structured Final Answer
```

## Frameworks & Libraries
| Framework | Purpose |
|-----------|---------|
| LangChain | Agent orchestration, tool use |
| LangGraph | Stateful multi-step agent graphs |
| AutoGen | Multi-agent conversations (Microsoft) |
| CrewAI | Role-based multi-agent teams |
| Semantic Kernel | Microsoft's agent SDK |
| OpenAI Assistants | Hosted agent API |

## Key Agentic AI Patterns
| Pattern | Description |
|---------|-------------|
| ReAct | Reason + Act loop |
| Chain-of-Thought | Step-by-step reasoning |
| Reflection | Agent critiques its own outputs |
| Plan & Execute | Plan all steps, then execute |
| Multi-Agent | Multiple specialized agents collaborate |
| RAG | Retrieval-Augmented Generation |

## Applications
- 🔬 **Research Agents** — Autonomously search, summarize, cite papers
- 💻 **Code Agents** — Write, debug, and execute code
- 📊 **Data Agents** — Query databases, generate reports
- 🛒 **E-commerce** — Product recommendation + order management
- 🏥 **Healthcare** — Patient triage + medical record analysis
- 🤖 **Personal AI** — Calendar management, email drafting

## How to Run on Google Colab
1. Upload `agent_ai.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Install dependencies (first cell will do this automatically)
3. **Add your API key** in the API key cell (OpenAI or Google Gemini)
4. Runtime → **Run All**

## Expected Output
- Agent reasoning traces (Thought → Action → Observation)
- Answers to multi-step questions using tools
- Demonstration of math + web search tool use
- Final structured response with citations

## Getting API Keys (Free Tiers Available)
- **Google Gemini**: https://aistudio.google.com/apikey (Free tier: 15 RPM)
- **OpenAI GPT**: https://platform.openai.com/api-keys (Paid, ~$5 credit to start)
- **Groq (Fast + Free)**: https://console.groq.com (Free tier with Llama 3)
