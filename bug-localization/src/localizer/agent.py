# src/localizer/agent.py
#
# TAJ owns this loop; it calls FARHAN's tools.
# Upgraded to use LangChain and LangGraph for an Autonomous Iterative Agent Loop.
import json
import re
import uuid
from collections import Counter, defaultdict
from typing import Annotated, Callable, TypedDict, Sequence, operator

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama

from src.graph import tools as farhan_tools
from src.llm import current_provider, make_chat_groq, groq_key_failed
from src.schemas import Budget, Instance, Prediction
from src.service.errors import JobCancelled

_TREE_CHARS_PER_HOP = 3_000

_RE_ENTITY = re.compile(r"([\w./\-]+\.py)\s*:\s*([A-Za-z_][\w.]*)")

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    tokens: int
    usd: float

def _parse_entities(text: str, limit: int) -> list[str]:
    out = []
    seen = set()
    for line in (text or "").splitlines():
        line = line.strip().strip("`")
        if not line or line.startswith(("```", "~~~")): continue
        m = _RE_ENTITY.search(line)
        if not m: continue
        path, name = m.group(1), m.group(2)
        ents = [f"{path}:{name}"]
        if "." in name:
            bare = name.rsplit(".", 1)[-1]
            if bare: ents.append(f"{path}:{bare}")
        for ent in ents:
            if ent not in seen:
                seen.add(ent)
                out.append(ent)
        if len(out) >= limit: break
    return out[:limit]

_SYSTEM_PROMPT = """You are an autonomous AI software engineer localizing a bug in a codebase.
Given the GitHub problem description, your objective is to localize the specific files, classes or functions that need modification.

Follow these steps:
1. Extract Keywords: Think of relevant code keywords from the problem statement.
2. Search & Traverse: Use `search_entity` and `traverse_graph` tools to explore the codebase.
3. Locate Target: Identify the exact entities requiring changes.

WARNING: You have a strict budget of {max_hops} tool actions. You MUST provide your final answer before exceeding this limit!

CRITICAL: When calling a tool, you MUST output ONLY the tool call JSON. Do NOT output any conversational preamble, reasoning, or thoughts before the tool call, otherwise the system will crash.

Final Answer Format:
Once you are confident you found the buggy file(s), output your final answer wrapped in triple backticks.
Each location must be formatted as: path/file.py:function
List them most-suspicious first. Do NOT use any tools after providing your final answer.
"""

_RE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.M)


def _extract_fallback_tool_call(content: str) -> dict | None:
    """Recover a tool call from Qwen-via-Ollama's raw-text tool-call dumps.

    Ollama's native tool-calling with this model is unreliable: instead of populating
    `AIMessage.tool_calls`, it sometimes writes the call as a single JSON object in
    `content` (`{"name": ..., "arguments": {...}}`), and sometimes writes a *burst* of
    several such objects, one per line, occasionally wrapped in a ```json fence —
    apparently one line per tool it considered calling.

    Without this recovery, `json.loads(content)` on the whole string raises (it isn't
    valid JSON — several objects back to back), the exception is swallowed, no
    tool_calls ends up set, and should_continue reads that as "no tools requested" and
    routes straight to END. The result: zero tools ever ran, and the burst of JSON is
    fed to `_parse_entities` as if it were the final answer — either matching nothing
    (empty prediction) or, worse, regex-matching a placeholder path/name the model used
    as an example argument for a call it never made (e.g. "path/to/your/model.py").

    Only the first valid call is taken — if the model wants to search further, the
    next round-trip through this loop is exactly what `max_hops` budgets for; taking
    the whole burst at once would let one malformed turn consume the entire effort
    budget regardless of the adaptive tier.
    """
    text = _RE_FENCE.sub("", (content or "").strip()).strip()
    if not text:
        return None

    def to_call(obj) -> dict | None:
        if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
            return {"name": obj["name"], "args": obj["arguments"], "id": str(uuid.uuid4())}
        return None

    try:
        call = to_call(json.loads(text))
        if call:
            return call
    except (json.JSONDecodeError, TypeError):
        pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            call = to_call(json.loads(line))
        except (json.JSONDecodeError, TypeError):
            continue
        if call:
            return call
    return None


def _invoke_with_groq_rotation(build_model: Callable[[], object], messages):
    """Invoke a Groq-backed chat model, rotating to the next configured API key
    and retrying on a 429 (rate/day limit) instead of failing the whole sample.

    ChatGroq bakes the API key into the client at construction time, so a plain
    retry of `.invoke()` on the same instance would just hit the same exhausted
    key again — the client has to be rebuilt (via `build_model`) after rotating.
    `_call_groq` in src/llm.py has its own copy of this loop for the non-agent
    call path; this one exists because the LangGraph nodes below construct and
    invoke ChatGroq directly rather than going through src.llm.call().
    """
    import groq

    model = build_model()
    while True:
        try:
            return model.invoke(messages)
        except (groq.RateLimitError, groq.APIStatusError) as e:
            is_rate_limit = (isinstance(e, groq.RateLimitError)
                             or getattr(e, "status_code", None) == 429)
            if not is_rate_limit:
                raise
            if not groq_key_failed():
                raise
            model = build_model()


def build_langgraph_app(graph, budget: Budget):
    @tool
    def search_entity(keyword: str) -> str:
        """Search the codebase using a keyword to find related files, classes, or functions. Returns a list of entities."""
        try:
            hits = farhan_tools.search_entity(graph, keyword, detail="preview")
            return str(hits[:budget.max_candidates])
        except Exception as e:
            return f"Error: {e}"

    @tool
    def traverse_graph(seeds: list[str], direction: str = "both") -> str:
        """Traverse the codebase graph starting from seed entity IDs (e.g. 'path/file.py:func') to find related entities.

        direction: 'out' = what these entities use/contain, 'in' = what uses/calls/imports
        these entities (reverse edges, shown with a '-by' suffix e.g. 'invoke-by'),
        'both' = both directions at once. Use 'in' or 'both' when you need to find every
        caller/importer of a suspicious entity, not just what it depends on.
        """
        try:
            tree = farhan_tools.traverse_graph(
                graph, seeds[:budget.max_candidates],
                hops=budget.max_hops,
                edge_types=["contain", "invoke", "import", "inherit"],
                direction=direction,
            )
            cap = _TREE_CHARS_PER_HOP * max(1, budget.max_hops)
            if len(tree or "") <= cap:
                return tree or ""
            kept = (tree or "")[:cap].rsplit("\n", 1)[0]
            return f"{kept}\n... truncated"
        except Exception as e:
            return f"Error: {e}"

    @tool
    def retrieve_entity(entity_id: str) -> str:
        """Retrieve the full code content of a specific entity ID."""
        try:
            return str(farhan_tools.retrieve_entity(graph, entity_id))
        except Exception as e:
            return f"Error: {e}"

    tools_list = [search_entity, traverse_graph, retrieve_entity]
    tool_node = ToolNode(tools_list)

    is_groq = current_provider() == "groq"
    if not is_groq:
        chat_model = ChatOllama(model=budget.model, temperature=0.0).bind_tools(tools_list)

    def agent_node(state: AgentState):
        if is_groq:
            response = _invoke_with_groq_rotation(
                lambda: make_chat_groq(budget.model, temperature=0.0).bind_tools(tools_list),
                state["messages"])
        else:
            response = chat_model.invoke(state["messages"])
        meta = response.response_metadata or {}
        usage = getattr(response, "usage_metadata", {}) or {}
        tin = meta.get("prompt_eval_count") or usage.get("input_tokens", 0)
        tout = meta.get("eval_count") or usage.get("output_tokens", 0)

        if not getattr(response, "tool_calls", None) and getattr(response, "content", None):
            call = _extract_fallback_tool_call(response.content)
            if call:
                response.tool_calls = [call]

        return {
            "messages": [response],
            "tokens": state.get("tokens", 0) + tin + tout,
            "usd": state.get("usd", 0.0)
        }

    def force_answer_node(state: AgentState):
        """Forces the agent to provide a final answer without tools when budget runs out."""
        reminder = SystemMessage(content="BUDGET EXHAUSTED. You must provide your final prediction wrapped in triple backticks NOW. Format: path/file.py:function")

        # Flatten tool history to avoid strict tool schema validation errors on Groq
        clean_messages = []
        for m in state["messages"]:
            if m.type == "ai" and getattr(m, "tool_calls", None):
                c = m.content or ""
                names = [tc["name"] for tc in getattr(m, "tool_calls", [])]
                clean_messages.append(AIMessage(content=f"{c}\n[Used tools: {names}]"))
            elif m.type == "tool":
                clean_messages.append(HumanMessage(content=f"[Tool Result]\n{m.content}"))
            else:
                clean_messages.append(m)

        if is_groq:
            response = _invoke_with_groq_rotation(
                lambda: make_chat_groq(budget.model, temperature=0.0),
                clean_messages + [reminder])
        else:
            llm_no_tools = ChatOllama(model=budget.model, temperature=0.0)
            response = llm_no_tools.invoke(clean_messages + [reminder])
        meta = response.response_metadata or {}
        usage = getattr(response, "usage_metadata", {}) or {}
        tin = meta.get("prompt_eval_count") or usage.get("input_tokens", 0)
        tout = meta.get("eval_count") or usage.get("output_tokens", 0)
        return {
            "messages": [response],
            "tokens": state.get("tokens", 0) + tin + tout,
            "usd": state.get("usd", 0.0)
        }

    def should_continue(state: AgentState):
        last_msg = state["messages"][-1]
        
        # Count how many tool calls have been completed
        tool_count = sum(1 for m in state["messages"] if m.type == "tool")
        
        if getattr(last_msg, "tool_calls", None):
            if tool_count >= budget.max_hops:
                return "force_answer"
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("force_answer", force_answer_node)
    
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", "force_answer", END])
    workflow.add_edge("tools", "agent")
    workflow.add_edge("force_answer", END)
    
    return workflow.compile()

def _fake_localize(inst: Instance, budget: Budget) -> Prediction:
    """Mock localization for fake/offline provider to not break tests."""
    return Prediction(
        instance_id=inst.instance_id,
        ranked_files=["src/cache.py"],
        ranked_functions=["src/cache.py:Cache.get"],
        tokens=100,
        usd=0.0
    )

def _message_to_trace(msg) -> dict | None:
    """Turn one new LangGraph message into a small, UI-renderable trace entry."""
    kind = getattr(msg, "type", "")
    if kind == "ai":
        calls = getattr(msg, "tool_calls", None)
        if calls:
            return {"type": "tool_call",
                    "calls": [{"tool": c.get("name"), "args": c.get("args")}
                             for c in calls]}
        if msg.content:
            return {"type": "final_answer", "text": msg.content}
        return None
    if kind == "tool":
        return {"type": "tool_result", "tool": getattr(msg, "name", "?"),
                "output": str(msg.content)[:2000]}
    return None


def _run_sample(app, state: dict, sample_idx: int,
                progress_cb: Callable[[int, dict], None] | None) -> dict:
    """Run one agent sample, streaming trace entries out via progress_cb as they occur.

    Returns the same shape app.invoke(state) would have returned.
    """
    limit = 50  # generous: max_hops is enforced explicitly via force_answer, not this
    if progress_cb is None:
        return app.invoke(state, config={"recursion_limit": limit})

    final_state = state
    seen = 0
    for step_state in app.stream(state, config={"recursion_limit": limit},
                                 stream_mode="values"):
        final_state = step_state
        msgs = final_state.get("messages", [])
        for msg in msgs[seen:]:
            entry = _message_to_trace(msg)
            if entry:
                progress_cb(sample_idx, entry)
        seen = len(msgs)
    return final_state


def localize(inst: Instance, budget: Budget, graph,
             progress_cb: Callable[[int, dict], None] | None = None) -> Prediction:
    """Run the agent loop and return the aggregated Prediction.

    progress_cb(sample_index, trace_entry), if given, is called once per new agent
    message (tool call / tool result / final answer) as the loop runs — used by the
    ad-hoc web UI (src/service/adhoc.py) to show a live trace. Not used by the offline
    evaluation pipeline (src/localizer/run.py), which leaves it at the default None.
    """
    if current_provider() == "fake":
        return _fake_localize(inst, budget)

    fn_votes = Counter()
    file_votes = Counter()
    T, U = 0, 0.0

    app = build_langgraph_app(graph, budget)

    for sample_idx in range(max(1, budget.max_samples)):
        try:
            state = {
                "messages": [
                    SystemMessage(content=_SYSTEM_PROMPT.format(max_hops=budget.max_hops)),
                    HumanMessage(content=f"Report:\n{inst.problem_statement[:12000]}")
                ],
                "tokens": 0,
                "usd": 0.0
            }
            final_state = _run_sample(app, state, sample_idx, progress_cb)

            T += final_state.get("tokens", 0)
            U += final_state.get("usd", 0.0)

            last_msg = final_state["messages"][-1]
            ents = _parse_entities(last_msg.content, budget.max_candidates)
            
            best_per_file = defaultdict(float)
            for rank, e in enumerate(ents):
                rr = 1.0 / (rank + 1)
                fn_votes[e] += rr
                f = e.split(":")[0]
                best_per_file[f] = max(best_per_file[f], rr)
            for f, rr in best_per_file.items():
                file_votes[f] += rr

        except JobCancelled:
            raise   # do not swallow: the job must stop, not skip to the next sample
        except Exception as e:
            print(f"    [warn] sample failed for {inst.instance_id}: {e}")
            continue

    return Prediction(
        instance_id=inst.instance_id,
        ranked_files=[f for f, _ in file_votes.most_common()],
        ranked_functions=[e for e, _ in fn_votes.most_common()],
        tokens=T,
        usd=U,
    )
