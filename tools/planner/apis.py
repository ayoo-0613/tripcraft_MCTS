import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import (
    planner_agent_prompt_direct_og,
    planner_agent_prompt_direct_param,
    react_planner_agent_prompt,
    plan_skeleton_prompt,
    plan_execute_prompt,
    verifier_repair_prompt,
    reflexion_repair_prompt,
)
# from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.llms.base import BaseLLM
# from langchain_community.llms import OpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from env import ReactEnv,ReactReflectEnv
import tiktoken
import re
import json
import openai
import time
from enum import Enum
from typing import Any, Dict, List, Union, Literal, Optional
# from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import argparse

_EVAL_IMPORT_CWD = os.getcwd()
try:
    from evaluation.commonsense_constraint import evaluation as commonsense_eval
    from evaluation.hard_constraint import evaluation as hard_eval
finally:
    os.chdir(_EVAL_IMPORT_CWD)


OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
# openai.api_key = OPENAI_API_KEY
# GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

def _resolve_ollama_model(model_name: str) -> Optional[str]:
    if model_name.startswith("ollama:"):
        return model_name.split(":", 1)[1]
    if model_name == "ollama":
        return os.environ.get("OLLAMA_MODEL")
    return None


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout_sec: float = 999.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def chat(self, prompt: str) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout_sec,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content")
        return str(content or "")


class LLMRunner:
    def __init__(self, model_name: str = "gpt-3.5-turbo-1106") -> None:
        self.model_name = model_name
        self.ollama_model = _resolve_ollama_model(model_name)
        self.ollama_client = None
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if model_name.startswith("ollama") and not self.ollama_model:
            raise ValueError("Ollama model name missing. Set OLLAMA_MODEL or use model_name=ollama:<model>.")

        if model_name in ["qwen", "phi4"]:
            model_path = {
                "qwen": "Qwen/Qwen2.5-7B-Instruct",
                "phi4": "microsoft/Phi-4-mini-instruct",
            }[model_name]
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload",
                attn_implementation="flash_attention_2",
            )
            self.llm = None
        elif self.ollama_model:
            self.ollama_client = OllamaChatClient(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=self.ollama_model,
                timeout_sec=float(os.environ.get("OLLAMA_TIMEOUT", "999")),
            )
            self.llm = None
        else:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                max_tokens=4096,
                openai_api_key=OPENAI_API_KEY,
            )

    def chat(self, prompt: str) -> str:
        if len(self.enc.encode(prompt)) > 12000:
            return "Max Token Length Exceeded."
        if self.model_name in ["qwen", "phi4"]:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            output = self.model.generate(**inputs, max_new_tokens=3072)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response_start = generated_text.find(prompt)
            if response_start != -1:
                generated_text = generated_text[response_start + len(prompt):].strip()
            return generated_text
        if self.ollama_client:
            return self.ollama_client.chat(prompt)
        if self.model_name == "gpt-4o":
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4096,
                api_key=OPENAI_API_KEY,
            )
            return response["choices"][0]["message"]["content"]
        return self.llm([HumanMessage(content=prompt)]).content

def catch_openai_api_error():
    error = sys.exc_info()[0]
    if error == openai.error.APIConnectionError:
        print("APIConnectionError")
    elif error == openai.error.RateLimitError:
        print("RateLimitError")
        time.sleep(60)
    elif error == openai.error.APIError:
        print("APIError")
    elif error == openai.error.AuthenticationError:
        print("AuthenticationError")
    else:
        print("API error:", error)


class ReflexionStrategy(Enum):
    """
    REFLEXION: Apply reflexion to the next reasoning trace 
    """
    REFLEXION = 'reflexion'


class Planner:
    def __init__(self,
                 agent_prompt: PromptTemplate = planner_agent_prompt_direct_og,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:
        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
        self.ollama_model = _resolve_ollama_model(model_name)
        self.ollama_client = None
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        if model_name.startswith("ollama") and not self.ollama_model:
            raise ValueError("Ollama model name missing. Set OLLAMA_MODEL or use model_name=ollama:<model>.")

        if model_name in ['qwen','phi4']:
            model_path = {
                'qwen': "Qwen/Qwen2.5-7B-Instruct",
                'phi4': "microsoft/Phi-4-mini-instruct"
            }[model_name]
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                offload_folder="offload",  # Enables CPU offloading
                attn_implementation="flash_attention_2"  # Speeds up inference
            )
        elif self.ollama_model:
            self.ollama_client = OllamaChatClient(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=self.ollama_model,
                timeout_sec=float(os.environ.get("OLLAMA_TIMEOUT", "999")),
            )
        else:
            self.llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=4096, openai_api_key=OPENAI_API_KEY)
        
        print(f"PlannerAgent {model_name} loaded.")

    def run(self, text, query, persona, log_file=None) -> str:
        if log_file:
            log_file.write('\n---------------Planner\n' + self._build_agent_prompt(text, query, persona))
        
        prompt = self._build_agent_prompt(text, query, persona)
        
        if self.model_name in ['qwen','phi4']:
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
            # print(self.model.generation_config)
            output = self.model.generate(**inputs, max_new_tokens=3072) #do_sample=False) # temperature=0.0) #equivalent
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            response_start = generated_text.find(prompt)
            if response_start != -1:
                generated_text = generated_text[response_start + len(prompt):].strip()
            
            return generated_text
        elif self.ollama_client:
            return self.ollama_client.chat(prompt)
        else:
            if len(self.enc.encode(prompt)) > 12000:
                return 'Max Token Length Exceeded.'
            elif self.model_name == 'gpt-4o':
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=4096,
                    api_key=OPENAI_API_KEY
                )
                return response['choices'][0]['message']['content']
            else:
                return self.llm([HumanMessage(content=prompt)]).content

    def _build_agent_prompt(self, text, query, persona) -> str:
        return self.agent_prompt.format(text=text, query=query, persona=persona)


REQUIRED_PLAN_KEYS = [
    "days",
    "current_city",
    "transportation",
    "breakfast",
    "attraction",
    "lunch",
    "dinner",
    "accommodation",
    "event",
    "point_of_interest_list",
]


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\\n", "", text)
        text = re.sub(r"\\n```$", "", text)
    return text.strip()


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    if text is None:
        return None
    text = _strip_code_fence(str(text).strip())
    if text.startswith("[") and text.endswith("]"):
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    m = re.search(r"\\[.*\\]", text, flags=re.DOTALL)
    if m:
        chunk = m.group(0)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        try:
            obj = eval(chunk, {"__builtins__": {}})
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return None


def _normalize_plan(plan: Any) -> Optional[List[Dict[str, Any]]]:
    if not isinstance(plan, list):
        return None
    normalized = []
    for idx, item in enumerate(plan, start=1):
        if not isinstance(item, dict):
            return None
        norm = {}
        for key in REQUIRED_PLAN_KEYS:
            value = item.get(key, "-")
            if key == "days":
                try:
                    value = int(value)
                except Exception:
                    value = idx
            elif isinstance(value, list):
                value = "; ".join([str(v) for v in value])
            elif value is None:
                value = "-"
            else:
                value = str(value)
            norm[key] = value
        normalized.append(norm)
    return normalized


def _coerce_query_data(query_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not query_data:
        return None
    query_data = dict(query_data)
    local_constraint = query_data.get("local_constraint")
    if isinstance(local_constraint, str):
        try:
            query_data["local_constraint"] = eval(local_constraint, {"__builtins__": {}})
        except Exception:
            pass
    return query_data


def _collect_failures(query_data: Optional[Dict[str, Any]], plan: List[Dict[str, Any]]) -> List[str]:
    if not query_data:
        return ["Missing query data for constraint evaluation."]
    failures = []
    commonsense_info = commonsense_eval(query_data, plan)
    hard_info = hard_eval(query_data, plan)
    for info in (commonsense_info, hard_info):
        if not info:
            continue
        for key, value in info.items():
            if value is None:
                continue
            ok = value[0]
            msg = value[1] if len(value) > 1 else ""
            if ok is False:
                failures.append(f"{key}: {msg}")
    return failures


class VerifierRepairPlanner:
    def __init__(
        self,
        model_name: str,
        agent_prompt: PromptTemplate,
        repair_prompt: PromptTemplate,
        max_rounds: int = 2,
    ) -> None:
        self.runner = LLMRunner(model_name=model_name)
        self.agent_prompt = agent_prompt
        self.repair_prompt = repair_prompt
        self.max_rounds = max_rounds

    def run(self, text, query, persona, query_data=None) -> str:
        prompt = self.agent_prompt.format(text=text, query=query, persona=persona)
        plan_text = self.runner.chat(prompt)
        query_data = _coerce_query_data(query_data)
        for _ in range(self.max_rounds):
            parsed = _extract_json_array(plan_text)
            normalized = _normalize_plan(parsed) if parsed is not None else None
            if normalized is None:
                failures = ["Plan is not valid JSON array with required keys."]
            else:
                failures = _collect_failures(query_data, normalized)
            if not failures:
                return plan_text
            plan_json = json.dumps(normalized or plan_text, ensure_ascii=True)
            repair = self.repair_prompt.format(
                text=text,
                query=query,
                persona=persona,
                plan_json=plan_json,
                failures="; ".join(failures),
            )
            plan_text = self.runner.chat(repair)
        return plan_text


class ReflexionPlanner(VerifierRepairPlanner):
    pass


class PlanExecutePlanner:
    def __init__(
        self,
        model_name: str,
        plan_prompt: PromptTemplate,
        execute_prompt: PromptTemplate,
    ) -> None:
        self.runner = LLMRunner(model_name=model_name)
        self.plan_prompt = plan_prompt
        self.execute_prompt = execute_prompt

    def run(self, text, query, persona, query_data=None) -> str:
        skeleton = self.runner.chat(self.plan_prompt.format(text=text, query=query, persona=persona))
        execute = self.execute_prompt.format(
            text=text,
            query=query,
            persona=persona,
            plan_json=skeleton,
        )
        return self.runner.chat(execute)


class ReactPlanner:
    """
    A question answering ReAct Agent.
    """
    def __init__(self,
                 agent_prompt: PromptTemplate = react_planner_agent_prompt,
                 model_name: str = 'gpt-3.5-turbo-1106',
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.model_name = model_name
        self.ollama_model = _resolve_ollama_model(model_name)
        self.ollama_client = None

        if model_name.startswith("ollama") and not self.ollama_model:
            raise ValueError("Ollama model name missing. Set OLLAMA_MODEL or use model_name=ollama:<model>.")

        if self.ollama_model:
            self.ollama_client = OllamaChatClient(
                base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
                model=self.ollama_model,
                timeout_sec=float(os.environ.get("OLLAMA_TIMEOUT", "999")),
            )
            self.react_llm = None
        else:
            self.react_llm = ChatOpenAI(
                model_name=model_name,
                temperature=0,
                max_tokens=1024,
                openai_api_key=OPENAI_API_KEY,
                model_kwargs={"stop": ["Action", "Thought", "Observation"]},
            )

        self.env = ReactEnv()
        self.query = None
        self.persona = None
        self.max_steps = 30
        self.reset()
        self.finished = False
        self.answer = ''
        self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def run(self, text, query, persona, reset = True) -> None:

        self.query = query
        self.text = text
        self.persona = persona

        if reset:
            self.reset()

        while not (self.is_halted() or self.is_finished()):
            self.step()

        if not self.answer:
            force_prompt = (
                self._build_agent_prompt()
                + "\n\nYou must now finish. Output only: Finish[<final Travel Plan>]"
            )
            try:
                forced = format_step(self._call_llm(force_prompt))
            except Exception:
                forced = ""
            action_type, action_arg = parse_action(forced)
            if action_type == "Finish":
                self.answer = action_arg
            elif forced:
                self.answer = forced

        return self.answer, self.scratchpad

    def _call_llm(self, prompt: str) -> str:
        if self.ollama_client:
            return self.ollama_client.chat(prompt)
        return self.react_llm([HumanMessage(content=prompt)]).content

    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.curr_step}:'
        self.scratchpad += ' ' + self.prompt_agent()
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.curr_step}:'
        action = self.prompt_agent()
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.curr_step}: '

        action_type, action_arg = parse_action(action)

        if action_type == 'CostEnquiry':
            try:
                input_arg = eval(action_arg)
                if type(input_arg) != dict:
                    raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
                observation = f'Cost: {self.env.run(input_arg)}'
            except SyntaxError:
                observation = f'The sub plan can not be parsed into json format, please check.'
            except ValueError as e:
                observation = str(e)

        elif action_type == 'Finish':
            self.finished = True
            observation = f'The plan is finished.'
            self.answer = action_arg

        else:
            observation = f'Action {action_type} is not supported.'

        self.curr_step += 1

        self.scratchpad += observation
        print(self.scratchpad.split('\n')[-1])

    def prompt_agent(self) -> str:
        while True:
            try:
                return format_step(self._call_llm(self._build_agent_prompt()))
            except:
                catch_openai_api_error()
                print(self._build_agent_prompt())
                print(len(self.enc.encode(self._build_agent_prompt())))
                time.sleep(5)

    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            query = self.query,
                            text = self.text,
                            persona = self.persona,
                            scratchpad = self.scratchpad)

    def is_finished(self) -> bool:
        return self.finished

    def is_halted(self) -> bool:
        return ((self.curr_step > self.max_steps) or (
                    len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

    def reset(self) -> None:
        self.scratchpad = ''
        self.answer = ''
        self.curr_step = 1
        self.finished = False


# class ReactReflectPlanner:
#     """
#     A question answering Self-Reflecting React Agent.
#     """
#     def __init__(self,
#                  agent_prompt: PromptTemplate = react_reflect_planner_agent_prompt,
#                 reflect_prompt: PromptTemplate = reflect_prompt,
#                  model_name: str = 'gpt-3.5-turbo-1106',
#                  ) -> None:
        
#         self.agent_prompt = agent_prompt
#         self.reflect_prompt = reflect_prompt
#         if model_name in ['gemini']:
#             self.react_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
#             self.reflect_llm = ChatGoogleGenerativeAI(temperature=0,model="gemini-pro",google_api_key=GOOGLE_API_KEY)
#         else:
#             self.react_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
#             self.reflect_llm = ChatOpenAI(model_name=model_name, temperature=0, max_tokens=1024, openai_api_key=OPENAI_API_KEY,model_kwargs={"stop": ["Action","Thought","Observation,'\n"]})
#         self.model_name = model_name
#         self.env = ReactReflectEnv()
#         self.query = None
#         self.max_steps = 30
#         self.reset()
#         self.finished = False
#         self.answer = ''
#         self.reflections: List[str] = []
#         self.reflections_str: str = ''
#         self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

#     def run(self, text, query, reset = True) -> None:

#         self.query = query
#         self.text = text

#         if reset:
#             self.reset()
        

#         while not (self.is_halted() or self.is_finished()):
#             self.step()
#             if self.env.is_terminated and not self.finished:
#                 self.reflect(ReflexionStrategy.REFLEXION)

        
#         return self.answer, self.scratchpad

    
#     def step(self) -> None:
#         # Think
#         self.scratchpad += f'\nThought {self.curr_step}:'
#         self.scratchpad += ' ' + self.prompt_agent()
#         print(self.scratchpad.split('\n')[-1])

#         # Act
#         self.scratchpad += f'\nAction {self.curr_step}:'
#         action = self.prompt_agent()
#         self.scratchpad += ' ' + action
#         print(self.scratchpad.split('\n')[-1])

#         # Observe
#         self.scratchpad += f'\nObservation {self.curr_step}: '

#         action_type, action_arg = parse_action(action)

#         if action_type == 'CostEnquiry':
#             try:
#                 input_arg = eval(action_arg)
#                 if type(input_arg) != dict:
#                     raise ValueError('The sub plan can not be parsed into json format, please check. Only one day plan is supported.')
#                 observation = f'Cost: {self.env.run(input_arg)}'
#             except SyntaxError:
#                 observation = f'The sub plan can not be parsed into json format, please check.'
#             except ValueError as e:
#                 observation = str(e)
        
#         elif action_type == 'Finish':
#             self.finished = True
#             observation = f'The plan is finished.'
#             self.answer = action_arg
        
#         else:
#             observation = f'Action {action_type} is not supported.'
        
#         self.curr_step += 1

#         self.scratchpad += observation
#         print(self.scratchpad.split('\n')[-1])

#     def reflect(self, strategy: ReflexionStrategy) -> None:
#         print('Reflecting...')
#         if strategy == ReflexionStrategy.REFLEXION: 
#             self.reflections += [self.prompt_reflection()]
#             self.reflections_str = format_reflections(self.reflections)
#         else:
#             raise NotImplementedError(f'Unknown reflection strategy: {strategy}')
#         print(self.reflections_str)

#     def prompt_agent(self) -> str:
#         while True:
#             try:
#                 if self.model_name in ['gemini']:
#                     return format_step(self.react_llm.invoke(self._build_agent_prompt()).content)
#                 else:
#                     return format_step(self.react_llm([HumanMessage(content=self._build_agent_prompt())]).content)
#             except:
#                 catch_openai_api_error()
#                 print(self._build_agent_prompt())
#                 print(len(self.enc.encode(self._build_agent_prompt())))
#                 time.sleep(5)
    
#     def prompt_reflection(self) -> str:
#         while True:
#             try:
#                 if self.model_name in ['gemini']:
#                     return format_step(self.reflect_llm.invoke(self._build_reflection_prompt()).content)
#                 else:
#                     return format_step(self.reflect_llm([HumanMessage(content=self._build_reflection_prompt())]).content)
#             except:
#                 catch_openai_api_error()
#                 print(self._build_reflection_prompt())
#                 print(len(self.enc.encode(self._build_reflection_prompt())))
#                 time.sleep(5)
    
#     def _build_agent_prompt(self) -> str:
#         return self.agent_prompt.format(
#                             query = self.query,
#                             text = self.text,
#                             scratchpad = self.scratchpad,
#                             reflections = self.reflections_str)
    
#     def _build_reflection_prompt(self) -> str:
#         return self.reflect_prompt.format(
#                             query = self.query,
#                             text = self.text,
#                             scratchpad = self.scratchpad)
    
#     def is_finished(self) -> bool:
#         return self.finished

#     def is_halted(self) -> bool:
#         return ((self.curr_step > self.max_steps) or (
#                     len(self.enc.encode(self._build_agent_prompt())) > 14000)) and not self.finished

#     def reset(self) -> None:
#         self.scratchpad = ''
#         self.answer = ''
#         self.curr_step = 1
#         self.finished = False
#         self.reflections = []
#         self.reflections_str = ''
#         self.env.reset()

class ReactReflectPlanner(ReactPlanner):
    pass


def format_step(step: str) -> str:
    return step.strip()

def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string, flags=re.DOTALL)

    try:
        if match:
            action_type = match.group(1)
            action_arg = match.group(2)
            return action_type, action_arg
        else:
            return None, None
        
    except:
        return None, None

# def format_reflections(reflections: List[str],
#                         header: str = REFLECTION_HEADER) -> str:
#     if reflections == []:
#         return ''
#     else:
#         return header + 'Reflections:\n- ' + '\n- '.join([r.strip() for r in reflections])

# if __name__ == '__main__':
    
