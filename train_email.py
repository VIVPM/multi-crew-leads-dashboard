import os
import yaml
from crewai import LLM, Agent, Task, Crew
from dotenv import load_dotenv

load_dotenv(dotenv_path='../.env')

files = {
    'lead_agents': 'config/lead_qualification_agents.yaml',
    'lead_tasks': 'config/lead_qualification_tasks.yaml',
    'email_agents': 'config/email_engagement_agents.yaml',
    'email_tasks': 'config/email_engagement_tasks.yaml'
}

configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r', encoding='utf-8') as file:
        configs[config_type] = yaml.safe_load(file)
        
email_agents_config = configs['email_agents']
email_tasks_config = configs['email_tasks']

inputs = {
        "lead_score":{
            "score":95,
            "scoring_criteria":["Role Relevance","Company Size","Market Presence","Cultural Fit"],
            "validation_notes":"""The cultural values of Amazon align with CrewAI's product and pitch.
                                However, potential differences in company culture and values should be considered to ensure a successful partnership.
                                CrewAI should be prepared to adapt to Amazon's fast-paced and competitive environment, while Amazon should be open to CrewAI's innovative approach to AI orchestration.
                            """
        },
        "company_info":{
            "revenue":513,
            "industry":"E-commerce, Technology, Retail",
            "company_name":"Amazon",
            "company_size":1500000,
            "market_presence":10
        },
        "personal_info":{
            "name":"Jeff Bezos",
            "job_title":"Founder & Executive Chairman",
            "role_relevance":10,
            "professional_background":"Jeff Bezos founded Amazon in 1994 and has been instrumental in its growth and innovation, including the development of Amazon Web Services (AWS)."
        }
}

llm3 = LLM(model="sambanova/Meta-Llama-3.3-70B-Instruct", api_key=os.getenv('SAMBANAVA_API_KEY'))
# Creating Email Writing Agents
email_content_specialist = Agent(
        config=email_agents_config['email_content_specialist'],
        llm=llm3
)

engagement_strategist = Agent(
        config=email_agents_config['engagement_strategist'],
        llm=llm3
)

    # Creating Email Writing Tasks
email_drafting = Task(
        config=email_tasks_config['email_drafting'],
        agent=email_content_specialist
)

engagement_optimization = Task(
        config=email_tasks_config['engagement_optimization'],
        context=[email_drafting],
        agent=engagement_strategist
)

    # Creating Email Writing Crew
email_writing_crew = Crew(
        agents=[email_content_specialist, engagement_strategist],
        tasks=[email_drafting, engagement_optimization],
        verbose=True
)

email_writing_crew.train(n_iterations=1, filename="training_data.pkl",inputs = inputs)
print("Training completed!")
email_writing_crew.test(n_iterations=2,eval_llm=llm3,inputs = inputs)
print("Testing completed!")