ZS_COT_SUFFIX = """
Additional requirement:
Think step by step to ensure all constraints are satisfied and all items come from Given information.
Do not output your reasoning steps.
Only output the final Travel Plan in the exact format of the example.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output:
"""

PLANNER_INSTRUCTION_ZS_COT_OG = PLANNER_INSTRUCTION_OG.replace(
    "Given information: {text}\nQuery: {query}\nTraveler Persona:\n{persona}\nOutput: ",
    ZS_COT_SUFFIX
)

PLANNER_INSTRUCTION_ZS_COT_PARAM = PLANNER_INSTRUCTION_PARAMETER_INFO.replace(
    "Given information: {text}\nQuery: {query}\nTraveler Persona:\n{persona}\nOutput: ",
    ZS_COT_SUFFIX
)

planner_agent_prompt_zs_cot_og = PromptTemplate(
    input_variables=["text", "query", "persona"],
    template=PLANNER_INSTRUCTION_ZS_COT_OG,
)

planner_agent_prompt_zs_cot_param = PromptTemplate(
    input_variables=["text", "query", "persona"],
    template=PLANNER_INSTRUCTION_ZS_COT_PARAM,
)
