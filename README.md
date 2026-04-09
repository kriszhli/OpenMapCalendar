# Open Map Calendar

Plan schedules and movement together: calendar timeline + live map + local AI agent.

![Demo GIF](/docs/screenshots/DEMO.gif)

## What This Is

Open Map Calendar is a local-first planning app for trips, multi-stop days, and travel-heavy schedules.
It combines:
- Time planning with `Day`, `Row`, and `Grid` calendar views
- Map-aware events with origin and destination support
- Route visualization for travel-aware scheduling
- A local AI agent that drafts, reviews, and confirms plans instead of just answering chat prompts

## Why It Matters

Most calendar apps ignore location context and most AI assistants stop at chat.
This project is designed to turn a messy itinerary into something you can review, edit, and commit with control.

## Product Tour

### 1. Start with Calendar Manager

First-time users land in `Calendar Manager`. Create a calendar, rename it, switch between calendars, or delete one when it is no longer needed.

### 2. Set the planning window

Use the top controls to choose:
- Start date
- Number of days
- Working-hour range
- View mode: `Day`, `Row`, or `Grid`

### 3. Build events directly on the timeline

- Drag in a day column to create an event block.
- Resize the block to refine start and end times.
- Add a title, notes, and optional origin/destination details.

### 4. Visualize movement

- `simple` route mode draws a lightweight route line.
- `precise` route mode fetches driving geometry and caches it.
- `hidden` mode keeps the locations but hides the line.

### 5. Use the AI agent

The floating AI Planner is not a basic chatbot. It drafts a schedule, asks for clarification when needed, and stages proposals for review before commit.

## AI Agent Highlights

### Agentic Workflow

The planner uses a LangGraph-based workflow in `planner_service/graph.py` to break a request into structured planning steps instead of one-shot generation.

### Multi-Level Memory System

- Short-term session state tracks the current conversation, staged proposal, and confirmation status.
- Long-term memory is stored in Chroma and populated from confirmed plans and manual corrections.
- Memory retrieval is calendar-specific, so preferences stay isolated per user or calendar.

### MCP Protocol Support

The planner can discover and call read-only MCP tools for external context such as weather, route estimates, and place normalization.
When MCP is unavailable, it falls back to local deterministic helpers so the planner still works.

### Planning Safety

- Strict JSON output contract for planner responses
- Input sanitization on all planner context fields
- Clarification mode when date, time, or location details are ambiguous
- Overlap checks before events are committed
- Rollback support for the last AI-generated batch

### Human-in-the-Loop Control

- AI proposals are staged before they are applied to the calendar
- Users can edit the proposal before commit
- Confirmed corrections are distilled back into memory as preference updates

## Tech Stack

- Frontend: React + TypeScript + Vite
- Backend: Express
- Agent runtime: Python + LangGraph
- Memory: ChromaDB
- MCP: read-only tool registry and protocol client
- Map: Leaflet
- Routing: OSRM
- Geocoding: Nominatim
- Local LLM: Ollama

## AI Engineering Notes

The planner service is intentionally built like an engineering system, not a prompt demo:
- `planner_service/graph.py` orchestrates the agent flow, replanning, clarification, and validation
- `planner_service/memory.py` provides long-term memory and retrieval
- `planner_service/mcp.py` discovers and filters MCP tools to read-only capabilities
- `planner_service/distillation.py` and `planner_service/export.py` turn confirmed interactions into training data and memory facts
- `planner_service/service.py` exposes the local HTTP API and proxies requests from the Node server

## Quick Start

### Requirements

- Node.js 18+
- npm
- Python 3.11+ for the planner service
- Optional: Ollama for local model execution

### Run locally

```bash
npm install
npm run dev
```

- App: `http://localhost:5173`
- API: `http://localhost:3000`

### Run the planner service directly

```bash
python3 -m planner_service --serve
```

### Optional AI setup

```bash
ollama serve
ollama pull gemma4:e2b
```

Optional env vars:
- `OLLAMA_URL` defaults to `http://127.0.0.1:11434`
- `OLLAMA_MODEL` defaults to `gemma4:e2b`
- `OLLAMA_TIMEOUT_MS` defaults to `45000`
- `PLANNER_SERVICE_URL` defaults to `http://127.0.0.1:8001`
- `PLANNER_MCP_CONFIG` or `PLANNER_MCP_SERVERS` configure MCP servers

## How It Runs

- `npm run dev` starts the Vite client, Node API server, and planner service together
- `npm run host` builds the frontend and starts the production Node server
- `server.js` proxies AI requests to the local planner service
- Calendar state is stored in JSON under `calendars/`

## Deployment

This project is designed to run as a single local Node server with a separate Python planner process.

```bash
npm install
npm run host
```

If you want the AI agent enabled in production, also start:

```bash
python3 -m planner_service --serve
```

## Scripts

- `npm run dev` - API + Vite + planner service
- `npm run dev:client` - Vite only
- `npm run dev:planner` - planner service only
- `npm run build` - type-check + production build
- `npm run start` - run Node server
- `npm run host` - build + host
- `npm run lint` - lint checks
