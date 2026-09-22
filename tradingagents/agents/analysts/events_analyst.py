from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_corporate_actions,
    get_earnings_calendar,
    get_earnings_history,
    get_instrument_context_from_state,
    get_language_instruction,
    strip_final_proposal_prefix,
)

def create_events_analyst(llm):
    def events_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_earnings_calendar,
            get_earnings_history,
            get_corporate_actions,
        ]

        system_message = (
            f"You are an events researcher tasked with analyzing SCHEDULED, forward-looking corporate events for the Chinese A-share company behind the ticker: the upcoming periodic-report disclosure date, the settlement pattern of recent quarterly reports, and dividend / share-conversion (corporate action) records. Your question is the one traders ask before every session: what is on the calendar in the coming weeks, and how have past events played out for this company? Use the available tools: get_earnings_calendar(ticker, curr_date) for the next report disclosure date (the scheduled date with days-until, any reschedule history, and the statutory deadline when the schedule is not yet published), get_earnings_history(ticker, curr_date) for the last settled quarterly reports (EPS, revenue and net profit with YoY growth, filtered to what was publicly announced by the analysis date), and get_corporate_actions(ticker, curr_date, lookback_years) for dividend and share-conversion plans (plan terms, progress, record and ex-dividend dates). Always pass curr_date exactly as given: the tools are point-in-time and withhold anything not yet public at that date. Ground every statement in retrieved tool output — if a tool reports data unavailable or a schedule not yet published, say so explicitly and reason from the statutory deadline or the settled history instead; never invent dates, figures, or events. Interpretation guidance: a disclosure days away is a volatility event, so flag the position-timing implication; a consistent beat pattern (net profit YoY positive across settled quarters) supports confidence in the next report; a plan approaching its ex-dividend date matters for entry timing; an overdue disclosure is a red flag worth stating plainly. Provide specific, actionable insights that help traders decide whether to act before or after the event."
            + """ Make sure to append a Markdown table at the end of the report to organize the event calendar and key takeaways in the report, organized and easy to read."""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)
        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = strip_final_proposal_prefix(result.content)

        return {
            "messages": [result],
            "events_report": report,
        }

    return events_analyst_node
