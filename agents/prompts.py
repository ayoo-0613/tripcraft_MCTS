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

PLANNER_INSTRUCTION_LLM_DIRECT = """You are a planner. Based ONLY on the given information, choose actions by index for each day to fill the template.
Output ONLY valid JSON. Do NOT include explanations.

Rules:
- Choose indices that exist in each candidate list.
- Use at most 2 attractions per day.
- If you want to skip a slot, choose the index of "-" in that list.
- Output must exactly match the schema below.

Schema:
{{
  "days": [
    {{
      "day": 1,
      "transportation_idx": 0,
      "accommodation_idx": 0,
      "breakfast_idx": 0,
      "lunch_idx": 0,
      "dinner_idx": 0,
      "attraction_idxs": [0, 1],
      "event_idx": 0
    }}
  ]
}}

Template JSON:
{template}

Candidates JSON (each slot list contains objects with "idx" and "value"):
{candidates}

Given information:
{text}
Query: {query}
Traveler Persona:
{persona}
"""

PLANNER_INSTRUCTION_LLM_DIRECT_REPAIR = """You must output ONLY valid JSON. Do NOT include explanations.

Task: Convert the draft output into the required JSON schema by choosing indices from the Candidates JSON.
Rules:
- Choose indices that exist in each candidate list.
- Use at most 2 attractions per day.
- If you want to skip a slot, choose the index of "-" in that list.
- Output must exactly match the schema below.

Schema:
{{
  "days": [
    {{
      "day": 1,
      "transportation_idx": 0,
      "accommodation_idx": 0,
      "breakfast_idx": 0,
      "lunch_idx": 0,
      "dinner_idx": 0,
      "attraction_idxs": [0, 1],
      "event_idx": 0
    }}
  ]
}}

Candidates JSON:
{candidates}

Draft output:
{text}
"""

PLANNER_INSTRUCTION_LLM_COT_PLAN = """You are a planner. Produce a slot-level allocation plan for each day using the given information.
Output only plain text (no JSON). Keep it concise and structured by day.

Template JSON:
{template}

Candidates JSON (each slot list contains objects with "idx" and "value"):
{candidates}

Given information:
{text}
Query: {query}
Traveler Persona:
{persona}
"""

PLANNER_INSTRUCTION_LLM_COT_FILL = """You must output ONLY valid JSON. Do NOT include explanations.

Task: Convert the draft allocation plan into the required JSON schema by choosing indices from the Candidates JSON.
Rules:
- Choose indices that exist in each candidate list.
- Use at most 2 attractions per day.
- If you want to skip a slot, choose the index of "-" in that list.
- Output must exactly match the schema below.

Schema:
{{
  "days": [
    {{
      "day": 1,
      "transportation_idx": 0,
      "accommodation_idx": 0,
      "breakfast_idx": 0,
      "lunch_idx": 0,
      "dinner_idx": 0,
      "attraction_idxs": [0, 1],
      "event_idx": 0
    }}
  ]
}}

Candidates JSON:
{candidates}

Draft allocation plan:
{text}
"""

PLANNER_INSTRUCTION_LLM_REFLEXION_REPAIR = """You must output ONLY valid JSON. Do NOT include explanations.

Task: Repair the slot assignments to fix the issues. Choose indices from the Candidates JSON.
Rules:
- Choose indices that exist in each candidate list.
- Use at most 2 attractions per day.
- If you want to skip a slot, choose the index of "-" in that list.
- Output must exactly match the schema below.

Schema:
{{
  "days": [
    {{
      "day": 1,
      "transportation_idx": 0,
      "accommodation_idx": 0,
      "breakfast_idx": 0,
      "lunch_idx": 0,
      "dinner_idx": 0,
      "attraction_idxs": [0, 1],
      "event_idx": 0
    }}
  ]
}}

Candidates JSON:
{candidates}

Detected issues:
{issues}

Draft output:
{text}
"""



planner_agent_prompt_direct_og = PromptTemplate(
                        input_variables=["text","query","persona"],
                        template = PLANNER_INSTRUCTION_OG,
                        )

planner_agent_prompt_direct_param = PromptTemplate(
                        input_variables=["text","query","persona"],
                        template = PLANNER_INSTRUCTION_PARAMETER_INFO,
                        )

planner_agent_prompt_llm_direct = PromptTemplate(
                        input_variables=["text","query","persona","template","candidates"],
                        template = PLANNER_INSTRUCTION_LLM_DIRECT,
                        )

planner_agent_prompt_llm_direct_repair = PromptTemplate(
                        input_variables=["text","candidates"],
                        template = PLANNER_INSTRUCTION_LLM_DIRECT_REPAIR,
                        )

planner_agent_prompt_llm_cot_plan = PromptTemplate(
                        input_variables=["text","query","persona","template","candidates"],
                        template = PLANNER_INSTRUCTION_LLM_COT_PLAN,
                        )

planner_agent_prompt_llm_cot_fill = PromptTemplate(
                        input_variables=["text","candidates"],
                        template = PLANNER_INSTRUCTION_LLM_COT_FILL,
                        )

planner_agent_prompt_llm_reflexion_repair = PromptTemplate(
                        input_variables=["text","candidates","issues"],
                        template = PLANNER_INSTRUCTION_LLM_REFLEXION_REPAIR,
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
