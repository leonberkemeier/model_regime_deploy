import logging
import json
import asyncio
from contextlib import AsyncExitStack

# mcp client imports
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

from connectors.webserver_client import WebserverClient
from connectors.ollama_client import OllamaClient
from config.settings import OLLAMA_MODEL, WEBSERVER_URL

logger = logging.getLogger(__name__)

async def _agent_score_asset(ticker: str, regime: str, ollama_client: OllamaClient, mcp_session: ClientSession) -> dict:
    """Run the autonomous agent loop for a single asset."""
    # 1. Get MCP Tools and format for Ollama
    mcp_tools_res = await mcp_session.list_tools()
    ollama_tools = []
    
    for t in mcp_tools_res.tools:
        ollama_tools.append({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.inputSchema
            }
        })
        
    system_msg = (
        f"You are an Autonomous Financial Agent. The current market regime is '{regime}'. "
        f"Your task is to analyze the asset '{ticker}'. "
        "You MUST use your tools (like get_quantitative_risk, search_filings) to gather risk data and unstructured sentiment. "
        "After gathering data, you MUST formulate a final answer containing ONLY a JSON object: "
        '{"score": <float between 0.0 and 1.0>, "reasoning": "<short explanation>"}'
    )
    
    messages = [{"role": "system", "content": system_msg}, {"role": "user", "content": f"Score {ticker} now."}]
    
    # 2. Agent Run Loop
    for step in range(5): # Max 5 turns
        logger.info(f"Agent Loop Step {step+1} for {ticker}...")
        
        # Query Ollama
        response_msg = ollama_client.chat(messages=messages, tools=ollama_tools, model=OLLAMA_MODEL)
        messages.append(response_msg)
        
        tool_calls = response_msg.get("tool_calls")
        if not tool_calls:
            logger.info("Agent halted tool use. Returning final structured JSON.")
            break
            
        # 3. Call remote Webserver MCP Tools
        for tc in tool_calls:
            func_name = tc["function"]["name"]
            args = tc["function"]["arguments"]
            logger.info(f"-> Agent called MCP tool '{func_name}' with {args}")
            
            try:
                tool_result = await mcp_session.call_tool(func_name, arguments=args)
                # Parse result safe-string (MCP returns a list of content blocks)
                try:
                    content_str = tool_result.content[0].text if tool_result.content else str(tool_result.content)
                except AttributeError:
                    content_str = str(tool_result.content)
                    
                messages.append({
                    "role": "tool",
                    "content": content_str,
                    "name": func_name
                })
            except Exception as e:
                logger.error(f"MCP Tool call remote execution failed: {e}")
                messages.append({
                    "role": "tool",
                    "content": f"Error calling tool on remote webserver: {e}",
                    "name": func_name
                })

    # 4. Parse Final JSON Response
    final_text = messages[-1].get("content", "")
    if "{" in final_text and "}" in final_text:
        start = final_text.find("{")
        end = final_text.rfind("}") + 1
        try:
            parsed = json.loads(final_text[start:end])
            return {
                "score": float(parsed.get("score", 0.5)),
                "reasoning": parsed.get("reasoning", "Parsed from agent memory")
            }
        except json.JSONDecodeError:
            pass
            
    return {"score": 0.5, "reasoning": "Could not parse valid JSON from agent response."}


async def _run_llm_pipeline_async(api_client: WebserverClient, ollama_client: OllamaClient):
    """Async wrapper connecting the MCP SSE transport."""
    logger.info("=== Starting Phase 3: Agentic LLM Asset Scoring via MCP ===")
    
    try:
        regime = api_client.get_markov_state()
        mc_results_list = api_client.get_monte_carlo_results()
        
        if not mc_results_list:
            logger.warning("No assets to score. Exiting.")
            return

        sse_url = f"{WEBSERVER_URL}/sse"
        logger.info(f"Connecting to Webserver MCP Server at {sse_url}")

        llm_scores = {}
        
        async with AsyncExitStack() as stack:
            # Connect over HTTP SSE to Webserver
            transport = await stack.enter_async_context(sse_client(sse_url))
            mcp_session = await stack.enter_async_context(ClientSession(transport))
            await mcp_session.initialize()
            logger.info("MCP Session Initialized successfully.")
            
            for asset in mc_results_list:
                ticker = asset.get('ticker')
                score_data = await _agent_score_asset(
                    ticker, 
                    regime.get('current_regime', 'unknown'), 
                    ollama_client, 
                    mcp_session
                )
                llm_scores[ticker] = score_data
                
        # Post LLM metrics back
        payload = {
            "execution_date": regime.get('execution_date'),
            "scores": llm_scores
        }
        api_client.post_llm_scores(payload)
        logger.info(f"✅ Success! Autonomous Agent processed {len(mc_results_list)} assets.")
        
    except Exception as e:
        logger.error(f"❌ LLM Task Failed: {e}", exc_info=True)


def run_llm_task(api_client: WebserverClient, ollama_client: OllamaClient):
    """Synchronous interface called by the daemon scheduler."""
    asyncio.run(_run_llm_pipeline_async(api_client, ollama_client))

