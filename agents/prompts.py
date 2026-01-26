from langchain.prompts import PromptTemplate

PLANNER_INSTRUCTION_OG = """You are a proficient planner. Based on the provided information, query and persona, please give a detailed travel plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plans should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the "Current City" section as in the example (i.e., from A to B). Include events happening on that day, if any. Provide a Point of Interest List, which is an ordered list of places visited throughout the day. This list should include only accommodations, attractions, or restaurants and their starting and ending timestamps. Each day must start and end with the accommodation where the traveler is staying.
 

****** Example ******  

Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?  
Traveler Persona:
Traveler Type: Laidback Traveler;
Purpose of Travel: Relaxation;
Spending Preference: Economical Traveler;
Location Preference: Beaches
  
Travel Plan:  
Day 1:  
Current City: from Ithaca to Charlotte  
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:15, Arrival Time: 07:28  
Breakfast: Nagaland's Kitchen, Charlotte  
Attraction: The Charlotte Museum of History, Charlotte  
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte
Event: -  
Point of Interest List: Affordable Spacious Refurbished Room in Bushwick!, stay from 08:00 to 08:30, nearest transit: Bushwick Stop, 100m away; Nagaland's Kitchen, visit from 09:00 to 09:45, nearest transit: Uptown Station, 200m away; The Charlotte Museum of History, visit from 10:30 to 13:30, nearest transit: Museum Station, 300m away; Cafe Maple Street, visit from 14:00 to 15:00, nearest transit: Maple Avenue Stop, 100m away; Bombay Vada Pav, visit from 19:00 to 20:00, nearest transit: Bombay Stop, 150m away; Affordable Spacious Refurbished Room in Bushwick!, stay from 21:00 to 07:00, nearest transit: Bushwick Stop, 100m away.  

Day 2:  
Current City: Charlotte  
Transportation: -  
Breakfast: Olive Tree Cafe, Charlotte  
Attraction: The Mint Museum, Charlotte; Romare Bearden Park, Charlotte  
Lunch: Birbal Ji Dhaba, Charlotte  
Dinner: Pind Balluchi, Charlotte  
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte  
Event: -  
Point of Interest List: Affordable Spacious Refurbished Room in Bushwick!, stay from 07:00 to 08:30, nearest transit: Bushwick Stop, 100m away; Olive Tree Cafe, visit from 09:00 to 09:45, nearest transit: Cafe Station, 250m away; The Mint Museum, visit from 10:30 to 13:00, nearest transit: Mint Stop, 200m away; Birbal Ji Dhaba, visit from 14:00 to 15:30, nearest transit: Dhaba Stop, 120m away; Romare Bearden Park, visit from 16:00 to 18:00, nearest transit: Park Stop, 150m away; Pind Balluchi, visit from 19:30 to 21:00, nearest transit: Pind Stop, 150m away; Affordable Spacious Refurbished Room in Bushwick!, stay from 21:30 to 07:00, nearest transit: Bushwick Stop, 100m away.  

Day 3:  
Current City: from Charlotte to Ithaca  
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26  
Breakfast: Subway, Charlotte  
Attraction: Books Monument, Charlotte  
Lunch: Olive Tree Cafe, Charlotte  
Dinner: Kylin Skybar, Charlotte  
Accommodation: -  
Event: -  
Point of Interest List: Affordable Spacious Refurbished Room in Bushwick!, stay from 07:00 to 08:30, nearest transit: Bushwick Stop, 100m away; Subway, visit from 09:00 to 10:00, nearest transit: Subway Station, 150m away; Books Monument, visit from 10:30 to 13:30, nearest transit: Central Library Stop, 200m away; Olive Tree Cafe, visit from 14:00 to 15:00, nearest transit: Cafe Station, 250m away; Kylin Skybar, visit from 19:00 to 20:00, nearest transit: Skybar Stop, 180m away.  

****** Example Ends ******

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """

PLANNER_INSTRUCTION_PARAMETER_INFO = """You are a proficient planner. Based on the provided information, query and persona, please give a detailed travel plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plans should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the "Current City" section as in the example (i.e., from A to B). Include events happening on that day, if any. Provide a Point of Interest List, which is an ordered list of places visited throughout the day. This list should include accommodations, attractions, or restaurants and their starting and ending timestamps. Each day must start and end with the accommodation where the traveler is staying. Breakfast is ideally scheduled at 9:40 AM and lasts about 50 minutes. Lunch is best planned for 2:20 PM, with a duration of around an hour. Dinner should take place at 8:45 PM, lasting approximately 1 hour and 15 minutes. Laidback Travelers typically explore one attraction per day and sometimes opt for more, while Adventure Seekers often visit 2 or 3 attractions, occasionally exceeding that number.
 

****** Example ******  

Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?  
Traveler Persona:
Traveler Type: Laidback Traveler;
Purpose of Travel: Relaxation;
Spending Preference: Economical Traveler;
Location Preference: Beaches
  
Travel Plan:  
Day 1:  
Current City: from Ithaca to Charlotte  
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:15, Arrival Time: 07:28  
Breakfast: Nagaland's Kitchen, Charlotte  
Attraction: The Charlotte Museum of History, Charlotte  
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte
Event: -  
Point of Interest List: Affordable Spacious Refurbished Room in Bushwick!, stay from 08:00 to 08:30, nearest transit: Bushwick Stop, 100m away; Nagaland's Kitchen, visit from 09:00 to 09:45, nearest transit: Uptown Station, 200m away; The Charlotte Museum of History, visit from 10:30 to 13:30, nearest transit: Museum Station, 300m away; Cafe Maple Street, visit from 14:00 to 15:00, nearest transit: Maple Avenue Stop, 100m away; Bombay Vada Pav, visit from 19:00 to 20:00, nearest transit: Bombay Stop, 150m away; Affordable Spacious Refurbished Room in Bushwick!, stay from 21:00 to 07:00, nearest transit: Bushwick Stop, 100m away.  

Day 2:  
Current City: Charlotte  
Transportation: -  
Breakfast: Olive Tree Cafe, Charlotte  
Attraction: The Mint Museum, Charlotte; Romare Bearden Park, Charlotte  
Lunch: Birbal Ji Dhaba, Charlotte  
Dinner: Pind Balluchi, Charlotte  
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte  
Event: -  
Point of Interest List: Affordable Spacious Refurbished Room in Bushwick!, stay from 07:00 to 08:30, nearest transit: Bushwick Stop, 100m away; Olive Tree Cafe, visit from 09:00 to 09:45, nearest transit: Cafe Station, 250m away; The Mint Museum, visit from 10:30 to 13:00, nearest transit: Mint Stop, 200m away; Birbal Ji Dhaba, visit from 14:00 to 15:30, nearest transit: Dhaba Stop, 120m away; Romare Bearden Park, visit from 16:00 to 18:00, nearest transit: Park Stop, 150m away; Pind Balluchi, visit from 19:30 to 21:00, nearest transit: Pind Stop, 150m away; Affordable Spacious Refurbished Room in Bushwick!, stay from 21:30 to 07:00, nearest transit: Bushwick Stop, 100m away.  

Day 3:  
Current City: from Charlotte to Ithaca  
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26  
Breakfast: Subway, Charlotte  
Attraction: Books Monument, Charlotte  
Lunch: Olive Tree Cafe, Charlotte  
Dinner: Kylin Skybar, Charlotte  
Accommodation: -  
Event: -  
Point of Interest List: Affordable Spacious Refurbished Room in Bushwick!, stay from 07:00 to 08:30, nearest transit: Bushwick Stop, 100m away; Subway, visit from 09:00 to 10:00, nearest transit: Subway Station, 150m away; Books Monument, visit from 10:30 to 13:30, nearest transit: Central Library Stop, 200m away; Olive Tree Cafe, visit from 14:00 to 15:00, nearest transit: Cafe Station, 250m away; Kylin Skybar, visit from 19:00 to 20:00, nearest transit: Skybar Stop, 180m away.  

****** Example Ends ******

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """



planner_agent_prompt_direct_og = PromptTemplate(
                        input_variables=["text","query","persona"],
                        template = PLANNER_INSTRUCTION_OG,
                        )

planner_agent_prompt_direct_param = PromptTemplate(
                        input_variables=["text","query","persona"],
                        template = PLANNER_INSTRUCTION_PARAMETER_INFO,
                        )

# ReAct-style planning prompt.
REACT_PLANNER_INSTRUCTION = """You are a proficient planner. Based on the provided information, query and persona, please give a detailed travel plan, including specifics such as flight numbers (e.g., F0123456), restaurant names, and accommodation names. Note that all the information in your plans should be derived from the provided data. You must adhere to the format given in the example. Additionally, all details should align with common sense. The symbol '-' indicates that information is unnecessary. For example, in the provided sample, you do not need to plan after returning to the departure city. When you travel to two cities in one day, you should note it in the "Current City" section as in the example (i.e., from A to B). Include events happening on that day, if any. Provide a Point of Interest List, which is an ordered list of places visited throughout the day. This list should include only accommodations, attractions, or restaurants and their starting and ending timestamps. Each day must start and end with the accommodation where the traveler is staying.

You can use the following tool when needed:
CostEnquiry[<json>]: returns the cost for a single-day plan. The JSON must be a dict for one day only.

Use Thought/Action/Observation steps. When you are ready to answer, output:
Finish[<final Travel Plan>]

Only complete the current line after the last label in the scratchpad. Do not output multiple steps at once.

****** Example ******

Query: Could you create a travel plan for 7 people from Ithaca to Charlotte spanning 3 days, from March 8th to March 14th, 2022, with a budget of $30,200?
Traveler Persona:
Traveler Type: Laidback Traveler;
Purpose of Travel: Relaxation;
Spending Preference: Economical Traveler;
Location Preference: Beaches

Travel Plan:
Day 1:
Current City: from Ithaca to Charlotte
Transportation: Flight Number: F3633413, from Ithaca to Charlotte, Departure Time: 05:15, Arrival Time: 07:28
Breakfast: Nagaland's Kitchen, Charlotte
Attraction: The Charlotte Museum of History, Charlotte
Lunch: Cafe Maple Street, Charlotte
Dinner: Bombay Vada Pav, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte
Event: -
Point of Interest List: Affordable Spacious Refurbished Room in Bushwick!, stay from 08:00 to 08:30, nearest transit: Bushwick Stop, 100m away; Nagaland's Kitchen, visit from 09:00 to 09:45, nearest transit: Uptown Station, 200m away; The Charlotte Museum of History, visit from 10:30 to 13:30, nearest transit: Museum Station, 300m away; Cafe Maple Street, visit from 14:00 to 15:00, nearest transit: Maple Avenue Stop, 100m away; Bombay Vada Pav, visit from 19:00 to 20:00, nearest transit: Bombay Stop, 150m away; Affordable Spacious Refurbished Room in Bushwick!, stay from 21:00 to 07:00, nearest transit: Bushwick Stop, 100m away.

Day 2:
Current City: Charlotte
Transportation: -
Breakfast: Olive Tree Cafe, Charlotte
Attraction: The Mint Museum, Charlotte; Romare Bearden Park, Charlotte
Lunch: Birbal Ji Dhaba, Charlotte
Dinner: Pind Balluchi, Charlotte
Accommodation: Affordable Spacious Refurbished Room in Bushwick!, Charlotte
Event: -
Point of Interest List: Affordable Spacious Refurbished Room in Bushwick!, stay from 07:00 to 08:30, nearest transit: Bushwick Stop, 100m away; Olive Tree Cafe, visit from 09:00 to 09:45, nearest transit: Cafe Station, 250m away; The Mint Museum, visit from 10:30 to 13:00, nearest transit: Mint Stop, 200m away; Birbal Ji Dhaba, visit from 14:00 to 15:30, nearest transit: Dhaba Stop, 120m away; Romare Bearden Park, visit from 16:00 to 18:00, nearest transit: Park Stop, 150m away; Pind Balluchi, visit from 19:30 to 21:00, nearest transit: Pind Stop, 150m away; Affordable Spacious Refurbished Room in Bushwick!, stay from 21:30 to 07:00, nearest transit: Bushwick Stop, 100m away.

Day 3:
Current City: from Charlotte to Ithaca
Transportation: Flight Number: F3786167, from Charlotte to Ithaca, Departure Time: 21:42, Arrival Time: 23:26
Breakfast: Subway, Charlotte
Attraction: Books Monument, Charlotte
Lunch: Olive Tree Cafe, Charlotte
Dinner: Kylin Skybar, Charlotte
Accommodation: -
Event: -
Point of Interest List: Affordable Spacious Refurbished Room in Bushwick!, stay from 07:00 to 08:30, nearest transit: Bushwick Stop, 100m away; Subway, visit from 09:00 to 10:00, nearest transit: Subway Station, 150m away; Books Monument, visit from 10:30 to 13:30, nearest transit: Central Library Stop, 200m away; Olive Tree Cafe, visit from 14:00 to 15:00, nearest transit: Cafe Station, 250m away; Kylin Skybar, visit from 19:00 to 20:00, nearest transit: Skybar Stop, 180m away.

****** Example Ends ******

Given information: {text}
Query: {query}
Traveler Persona:
{persona}

{scratchpad}
"""

react_planner_agent_prompt = PromptTemplate(
                        input_variables=["text","query","persona","scratchpad"],
                        template = REACT_PLANNER_INSTRUCTION,
                        )

# cot_planner_agent_prompt = PromptTemplate(
#                         input_variables=["text","query"],
#                         template = COT_PLANNER_INSTRUCTION,
#                         )

# react_planner_agent_prompt = PromptTemplate(
#                         input_variables=["text","query", "scratchpad"],
#                         template = REACT_PLANNER_INSTRUCTION,
#                         )

# reflect_prompt = PromptTemplate(
#                         input_variables=["text", "query", "scratchpad"],
#                         template = REFLECT_INSTRUCTION,
#                         )

# react_reflect_planner_agent_prompt = PromptTemplate(
#                         input_variables=["text", "query", "reflections", "scratchpad"],
#                         template = REACT_REFLECT_PLANNER_INSTRUCTION,
                        # )

# ZS-CoT augmentation
ZS_COT_ADDON = """
Additional requirement:
Think step by step internally to ensure all constraints are satisfied and all items come from the provided data.
Do not reveal your reasoning steps.
Only output the final Travel Plan in the exact format of the example.
"""

def _inject_zs_cot(instruction: str) -> str:
    """
    Insert ZS-CoT addon right before the "Given information" block.
    Keeps the rest of the dataset instruction unchanged.
    """
    marker = "\nGiven information: {text}\nQuery: {query}\nTraveler Persona:\n{persona}\nOutput: "
    if marker in instruction:
        return instruction.replace(marker, "\n" + ZS_COT_ADDON + "\n" + marker)

    # fallback: if the marker string differs slightly
    return instruction + "\n" + ZS_COT_ADDON + "\nGiven information: {text}\nQuery: {query}\nTraveler Persona:\n{persona}\nOutput: "


# You already have:
# PLANNER_INSTRUCTION_OG
# PLANNER_INSTRUCTION_PARAMETER_INFO

PLANNER_INSTRUCTION_ZS_COT_OG = _inject_zs_cot(PLANNER_INSTRUCTION_OG)
PLANNER_INSTRUCTION_ZS_COT_PARAM = _inject_zs_cot(PLANNER_INSTRUCTION_PARAMETER_INFO)

planner_agent_prompt_zs_cot_og = PromptTemplate(
    input_variables=["text", "query", "persona"],
    template=PLANNER_INSTRUCTION_ZS_COT_OG,
)

planner_agent_prompt_zs_cot_param = PromptTemplate(
    input_variables=["text", "query", "persona"],
    template=PLANNER_INSTRUCTION_ZS_COT_PARAM,
)

# Plan-and-execute: first produce a JSON skeleton with placeholders,
# then fill in concrete values using the provided data.
PLAN_SKELETON_INSTRUCTION = """You are a proficient planner. Based on the provided information, query, and persona, produce a JSON skeleton of the travel plan.

IMPORTANT OUTPUT FORMAT (STRICT JSON ONLY):
- Return ONLY a JSON array. Do not include any other text or Markdown.
- Each array item must include exactly these keys:
  "days", "current_city", "transportation", "breakfast", "attraction",
  "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
- Use "-" if any field is missing. Use ";" to separate multiple items.
- "days" is an integer day number starting from 1.
- When traveling between cities that day, set "current_city" to "from A to B".
- Use the placeholder "TBD" for any specific entity names or times that must be filled later.
- Keep the correct number of days and correct city transitions based on the query.
- Do NOT invent concrete restaurant/attraction/flight/accommodation names here.
- Ensure valid JSON with double quotes, no trailing commas.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """

PLAN_EXECUTE_INSTRUCTION = """You are a proficient planner. Based on the provided information, query, persona, and the JSON skeleton, fill in all placeholders with concrete details drawn ONLY from the provided data.

IMPORTANT OUTPUT FORMAT (STRICT JSON ONLY):
- Return ONLY a JSON array. Do not include any other text or Markdown.
- Each array item must include exactly these keys:
  "days", "current_city", "transportation", "breakfast", "attraction",
  "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
- Use "-" if any field is missing. Use ";" to separate multiple items.
- "days" is an integer day number starting from 1.
- When traveling between cities that day, set "current_city" to "from A to B".
- Ensure valid JSON with double quotes, no trailing commas.

JSON skeleton:
{plan_json}

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """

# Parameter-aware plan-and-execute prompts.
PLAN_SKELETON_INSTRUCTION_PARAM = """You are a proficient planner. Based on the provided information, query, and persona, produce a JSON skeleton of the travel plan. Breakfast is ideally scheduled at 9:40 AM and lasts about 50 minutes. Lunch is best planned for 2:20 PM, with a duration of around an hour. Dinner should take place at 8:45 PM, lasting approximately 1 hour and 15 minutes. Laidback Travelers typically explore one attraction per day and sometimes opt for more, while Adventure Seekers often visit 2 or 3 attractions, occasionally exceeding that number.

IMPORTANT OUTPUT FORMAT (STRICT JSON ONLY):
- Return ONLY a JSON array. Do not include any other text or Markdown.
- Each array item must include exactly these keys:
  "days", "current_city", "transportation", "breakfast", "attraction",
  "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
- Use "-" if any field is missing. Use ";" to separate multiple items.
- "days" is an integer day number starting from 1.
- When traveling between cities that day, set "current_city" to "from A to B".
- Use the placeholder "TBD" for any specific entity names or times that must be filled later.
- Keep the correct number of days and correct city transitions based on the query.
- Do NOT invent concrete restaurant/attraction/flight/accommodation names here.
- Ensure valid JSON with double quotes, no trailing commas.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """

PLAN_EXECUTE_INSTRUCTION_PARAM = """You are a proficient planner. Based on the provided information, query, persona, and the JSON skeleton, fill in all placeholders with concrete details drawn ONLY from the provided data. Breakfast is ideally scheduled at 9:40 AM and lasts about 50 minutes. Lunch is best planned for 2:20 PM, with a duration of around an hour. Dinner should take place at 8:45 PM, lasting approximately 1 hour and 15 minutes. Laidback Travelers typically explore one attraction per day and sometimes opt for more, while Adventure Seekers often visit 2 or 3 attractions, occasionally exceeding that number.

IMPORTANT OUTPUT FORMAT (STRICT JSON ONLY):
- Return ONLY a JSON array. Do not include any other text or Markdown.
- Each array item must include exactly these keys:
  "days", "current_city", "transportation", "breakfast", "attraction",
  "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
- Use "-" if any field is missing. Use ";" to separate multiple items.
- "days" is an integer day number starting from 1.
- When traveling between cities that day, set "current_city" to "from A to B".
- Ensure valid JSON with double quotes, no trailing commas.

JSON skeleton:
{plan_json}

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """
# Verifier-guided repair (hard + commonsense constraints)
VERIFIER_REPAIR_INSTRUCTION = """You are a meticulous planner. You are given a travel plan JSON and a list of constraint failures.
Fix the plan to satisfy all constraints. Use ONLY the provided data. Preserve the required JSON format.

Constraints failed (must fix all):
{failures}

Current plan JSON:
{plan_json}

IMPORTANT OUTPUT FORMAT (STRICT JSON ONLY):
- Return ONLY a JSON array. Do not include any other text or Markdown.
- Each array item must include exactly these keys:
  "days", "current_city", "transportation", "breakfast", "attraction",
  "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
- Use "-" if any field is missing. Use ";" to separate multiple items.
- "days" is an integer day number starting from 1.
- When traveling between cities that day, set "current_city" to "from A to B".
- Ensure valid JSON with double quotes, no trailing commas.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """

# Parameter-aware verifier repair prompt.
VERIFIER_REPAIR_INSTRUCTION_PARAM = """You are a meticulous planner. You are given a travel plan JSON and a list of constraint failures.
Fix the plan to satisfy all constraints. Use ONLY the provided data. Preserve the required JSON format. Breakfast is ideally scheduled at 9:40 AM and lasts about 50 minutes. Lunch is best planned for 2:20 PM, with a duration of around an hour. Dinner should take place at 8:45 PM, lasting approximately 1 hour and 15 minutes. Laidback Travelers typically explore one attraction per day and sometimes opt for more, while Adventure Seekers often visit 2 or 3 attractions, occasionally exceeding that number.

Constraints failed (must fix all):
{failures}

Current plan JSON:
{plan_json}

IMPORTANT OUTPUT FORMAT (STRICT JSON ONLY):
- Return ONLY a JSON array. Do not include any other text or Markdown.
- Each array item must include exactly these keys:
  "days", "current_city", "transportation", "breakfast", "attraction",
  "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
- Use "-" if any field is missing. Use ";" to separate multiple items.
- "days" is an integer day number starting from 1.
- When traveling between cities that day, set "current_city" to "from A to B".
- Ensure valid JSON with double quotes, no trailing commas.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """

# Reflexion: reflect silently, then repair the plan JSON.
REFLEXION_REPAIR_INSTRUCTION = """You are a meticulous planner. Reflect on why the current plan violates constraints, then repair it.
Do not output your reflection. Only output the corrected JSON plan.

Constraints failed (must fix all):
{failures}

Current plan JSON:
{plan_json}

IMPORTANT OUTPUT FORMAT (STRICT JSON ONLY):
- Return ONLY a JSON array. Do not include any other text or Markdown.
- Each array item must include exactly these keys:
  "days", "current_city", "transportation", "breakfast", "attraction",
  "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
- Use "-" if any field is missing. Use ";" to separate multiple items.
- "days" is an integer day number starting from 1.
- When traveling between cities that day, set "current_city" to "from A to B".
- Ensure valid JSON with double quotes, no trailing commas.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """

REFLEXION_REPAIR_INSTRUCTION_PARAM = """You are a meticulous planner. Reflect on why the current plan violates constraints, then repair it. Breakfast is ideally scheduled at 9:40 AM and lasts about 50 minutes. Lunch is best planned for 2:20 PM, with a duration of around an hour. Dinner should take place at 8:45 PM, lasting approximately 1 hour and 15 minutes. Laidback Travelers typically explore one attraction per day and sometimes opt for more, while Adventure Seekers often visit 2 or 3 attractions, occasionally exceeding that number.

Constraints failed (must fix all):
{failures}

Current plan JSON:
{plan_json}

IMPORTANT OUTPUT FORMAT (STRICT JSON ONLY):
- Return ONLY a JSON array. Do not include any other text or Markdown.
- Each array item must include exactly these keys:
  "days", "current_city", "transportation", "breakfast", "attraction",
  "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
- Use "-" if any field is missing. Use ";" to separate multiple items.
- "days" is an integer day number starting from 1.
- When traveling between cities that day, set "current_city" to "from A to B".
- Ensure valid JSON with double quotes, no trailing commas.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """

plan_skeleton_prompt = PromptTemplate(
    input_variables=["text", "query", "persona"],
    template=PLAN_SKELETON_INSTRUCTION,
)

plan_skeleton_prompt_param = PromptTemplate(
    input_variables=["text", "query", "persona"],
    template=PLAN_SKELETON_INSTRUCTION_PARAM,
)

plan_execute_prompt = PromptTemplate(
    input_variables=["text", "query", "persona", "plan_json"],
    template=PLAN_EXECUTE_INSTRUCTION,
)

plan_execute_prompt_param = PromptTemplate(
    input_variables=["text", "query", "persona", "plan_json"],
    template=PLAN_EXECUTE_INSTRUCTION_PARAM,
)

# LLM-guided template + action selection (MCTS-style filling).
TEMPLATE_GUIDANCE_INSTRUCTION_PARAM = """You are a proficient planner. Based on the provided information, query, and persona, produce a high-level JSON template for the travel plan. The template should guide action selection later and must NOT include specific POI names, flight numbers, or exact accommodations/restaurants. Use abstract hints instead (e.g., "museum", "park", "local cuisine", "budget hotel").

IMPORTANT OUTPUT FORMAT (STRICT JSON ONLY):
- Return ONLY a JSON array. Do not include any other text or Markdown.
- Each array item must include exactly these keys:
  "days", "current_city", "transportation", "breakfast", "attraction",
  "lunch", "dinner", "accommodation", "event", "point_of_interest_list".
- Use "-" if any field is intentionally skipped.
- Use "TBD" when a concrete entity is required but unspecified.
- For "attraction", list desired types or themes separated by ";" (e.g., "Museum; Park") or "-" if none.
- "days" is an integer day number starting from 1.
- When traveling between cities that day, set "current_city" to "from A to B".
- Ensure valid JSON with double quotes, no trailing commas.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Output: """

ACTION_SELECT_INSTRUCTION = """You are selecting exactly ONE action from candidates to fill the template.
Return ONLY a JSON object: {{"choice": <index>}} where <index> is the 0-based index of the chosen candidate.
If the template indicates skipping this slot (value "-" or "none"), choose a skip_* action if present.

Context:
Day: {day}
Slot: {slot}
Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Template Day JSON:
{template_day}

Candidates:
{candidates}

Output: """

ACTION_SELECT_REACT_INSTRUCTION = """You are selecting exactly ONE action from candidates to fill the template.
Think step by step to align with the template and persona, but ONLY output the final JSON object.
Return ONLY a JSON object: {{"choice": <index>}} where <index> is the 0-based index of the chosen candidate.
If the template indicates skipping this slot (value "-" or "none"), choose a skip_* action if present.

Context:
Day: {day}
Slot: {slot}
Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Template Day JSON:
{template_day}

Candidates:
{candidates}

Output: """

ACTION_SELECT_REFLEXION_INSTRUCTION = """You previously chose an action index. Reflect on whether it matches the template and persona.
If it is suboptimal, change it. Return ONLY a JSON object: {{"choice": <index>}} where <index> is the 0-based index.

Previous choice: {initial_choice}

Context:
Day: {day}
Slot: {slot}
Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Template Day JSON:
{template_day}

Candidates:
{candidates}

Output: """

ACTION_SELECT_BATCH_INSTRUCTION = """You are selecting actions for multiple slots to fill the template.
Return ONLY a JSON array with length equal to the number of steps.
Each item can be either an integer index or an object {{"choice": <index>}}.
Indices are 0-based and correspond to the candidates list for that step.
If the template indicates skipping this slot (value "-" or "none"), choose a skip_* action if present.

Given information: {text}
Query: {query}
Traveler Persona:
{persona}
Template JSON:
{template}

Steps (ordered):
{steps}

Output: """

template_guidance_prompt_param = PromptTemplate(
    input_variables=["text", "query", "persona"],
    template=TEMPLATE_GUIDANCE_INSTRUCTION_PARAM,
)

action_select_prompt = PromptTemplate(
    input_variables=["day", "slot", "text", "persona", "query", "template_day", "candidates"],
    template=ACTION_SELECT_INSTRUCTION,
)

action_select_prompt_react = PromptTemplate(
    input_variables=["day", "slot", "text", "persona", "query", "template_day", "candidates"],
    template=ACTION_SELECT_REACT_INSTRUCTION,
)

action_select_prompt_reflexion = PromptTemplate(
    input_variables=["day", "slot", "text", "persona", "query", "template_day", "candidates", "initial_choice"],
    template=ACTION_SELECT_REFLEXION_INSTRUCTION,
)

action_select_prompt_batch = PromptTemplate(
    input_variables=["text", "query", "persona", "template", "steps"],
    template=ACTION_SELECT_BATCH_INSTRUCTION,
)

verifier_repair_prompt = PromptTemplate(
    input_variables=["text", "query", "persona", "plan_json", "failures"],
    template=VERIFIER_REPAIR_INSTRUCTION,
)

verifier_repair_prompt_param = PromptTemplate(
    input_variables=["text", "query", "persona", "plan_json", "failures"],
    template=VERIFIER_REPAIR_INSTRUCTION_PARAM,
)

reflexion_repair_prompt = PromptTemplate(
    input_variables=["text", "query", "persona", "plan_json", "failures"],
    template=REFLEXION_REPAIR_INSTRUCTION,
)

reflexion_repair_prompt_param = PromptTemplate(
    input_variables=["text", "query", "persona", "plan_json", "failures"],
    template=REFLEXION_REPAIR_INSTRUCTION_PARAM,
)
