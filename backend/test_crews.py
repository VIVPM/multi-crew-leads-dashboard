# test_crews.py — manual smoke test + CrewAI evaluation, built on the real
# pipeline.build_crews() so this can never drift out of sync with the actual
# agent/task structure (which is exactly what happened to the old hand-rolled
# copy of this file).
import os
import sys
import warnings

# Windows' console defaults to cp1252, which can't print the emoji CrewAI's
# internal event-bus logging emits (✨ etc.) — reconfigure before anything
# else touches stdout/stderr. os.environ["PYTHONIOENCODING"] alone doesn't
# retroactively affect an already-open stream in this process.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings('ignore')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/ — pipeline.py lives right here

try:
    from dotenv import load_dotenv
    from crewai import LLM
except ModuleNotFoundError as e:
    venv_python = os.path.join(BASE_DIR, '.venv', 'Scripts', 'python.exe')
    sys.exit(
        f"{e}\n\n"
        f"This script needs the backend virtualenv, not the system Python "
        f"({sys.executable}).\nRun it with:\n\n"
        f'    "{venv_python}" "{__file__}"\n\n'
        f"(or activate backend/.venv first: backend\\.venv\\Scripts\\activate)"
    )

load_dotenv(dotenv_path=os.path.join(BASE_DIR, '.env'))

from pipeline import build_crews, format_company_summary

PROVIDER = "gemini"
gemini_key = os.getenv("GEMINI_API_KEY") or input("Enter your Gemini API Key: ")
tavily_key = os.getenv("TAVILY_API_KEY") or input("Enter your Tavily API Key: ")

# A stronger model as the LLM-as-judge for .test() evaluation, independent
# of whichever (cheaper) models power the crews being evaluated.
eval_llm = LLM(model="gemini/gemini-2.5-pro", api_key=gemini_key, temperature=0.7)

crews = build_crews(gemini_key, tavily_key)
company_crew = crews["company"]
personal_scoring_crew = crews["personal_scoring"]
email_crew = crews["email"]

our_company_context = (
    "Company Name: CrewAI\n"
    "Product: Multi-Agent Orchestration Platform\n"
    "ICP: Enterprise companies looking into Agentic automation.\n"
    "Pitch: We are a platform that allows you to orchestrate AI Agents for "
    "automations to any vertical."
)

lead_data = {
    "name": "Brian Chesky",
    "job_title": "CEO & Co-Founder",
    "company": "Airbnb",
    "email": "brian.chesky@airbnb.com",
    "use_case": "Customer experience automation",
    "industry": "Technology & Software",
    "location": "San Francisco, USA",
    "source": "Website",
}

if __name__ == "__main__":
    print("=" * 60)
    print(f"Testing with Provider: {PROVIDER}")
    print("=" * 60)

    # --- Step 1: company research (independently kickoff-able for caching) ---
    print("\n=== Running Company Research Crew (kickoff) ===")
    company_result = company_crew.kickoff(inputs={
        "lead_data": lead_data,
        "our_company_context": our_company_context,
    })
    print(company_result.pydantic)
    company_summary = format_company_summary(company_result.pydantic.dict())

    # --- Step 2: personal research + scoring/validation ---
    print("\n=== Running Personal Research + Scoring Crew (kickoff) ===")
    score_result = personal_scoring_crew.kickoff(inputs={
        "lead_data": lead_data,
        "company_research_summary": company_summary,
    })
    print(score_result.pydantic)

    # --- Step 3: email (merged draft + optimize) ---
    print("\n=== Running Email Crew (kickoff) ===")
    email_inputs = {**score_result.to_dict(), "our_company_context": our_company_context}
    email_result = email_crew.kickoff(inputs=email_inputs)
    print(email_result.raw)

    # --- Step 4: CrewAI evaluation tables ---
    # verbose is switched off here on purpose: the crews were built with
    # verbose=True for the kickoff steps above (useful to watch), but that
    # same verbose printing collides with .test()'s own rich-rendered table
    # on Windows consoles, garbling both into unreadable interleaved output.
    for crew in (company_crew, personal_scoring_crew, email_crew):
        crew.verbose = False
        for agent in crew.agents:
            agent.verbose = False

    print("\n=== Testing Company Research Crew (eval table) ===")
    company_crew.test(n_iterations=2, eval_llm=eval_llm, inputs={
        "lead_data": lead_data, "our_company_context": our_company_context,
    })

    print("\n=== Testing Personal Research + Scoring Crew (eval table) ===")
    personal_scoring_crew.test(n_iterations=2, eval_llm=eval_llm, inputs={
        "lead_data": lead_data, "company_research_summary": company_summary,
    })

    print("\n=== Testing Email Crew (eval table) ===")
    email_crew.test(n_iterations=2, eval_llm=eval_llm, inputs=email_inputs)
