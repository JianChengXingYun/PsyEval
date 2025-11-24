import asyncio
import copy
import json
import logging
import re
import time
import traceback
from typing import Optional
import ast
from .LLM_Core import LLM_Core
from .Prompter import Prompter
from .Tools import Tools
from .PsyEval_Prompter import Prompter as  Prompter_PsyEval
from .PsyEval_Prompter_E import Prompter as  Prompter_PsyEval_E
from .PsyEval_PrompterV2 import Prompter as  Prompter_PsyEvalV2

class Process_Controller:
    def __init__(self, llm: LLM_Core, tools: Tools):
        """
        Inicializa el controlador de procesos con los componentes necesarios.
        Se han eliminado todos los atributos y componentes no utilizados por LLMExplorer_Socrates.
        """
        self.llm = llm
        self.tools = tools
        self.prompter = Prompter()
        self.logger = logging.getLogger(__name__)
        self.Prompter_PsyEval = Prompter_PsyEval()
        self.Prompter_PsyEval_E = Prompter_PsyEval_E()
        self.Prompter_PsyEvalV2 = Prompter_PsyEvalV2()
        # Plantilla de datos base para las llamadas a la API del LLM.
        self.data_template = {
            "model": self.llm.api_model,
            "messages": [],
            "temperature": 0.95,
            "top_p": 0.9,
            "extra_body": {},
            "stream": False,
        }
        self.scoring_prompts = [
                ("emotion_recognition", self.Prompter_PsyEval.emotion_recognition_scoring_prompt),
                ("analysis_ability", self.Prompter_PsyEval.analysis_ability_scoring_prompt),
                ("semantic_understanding", self.Prompter_PsyEval.semantic_understanding_scoring_prompt),
                ("logical_ability", self.Prompter_PsyEval.logical_ability_scoring_prompt),
                ("open_ended_questioning", self.Prompter_PsyEval.open_ended_questioning_scoring_prompt),
                ("positive_guidance", self.Prompter_PsyEval.positive_guidance_scoring_prompt),
                ("continuous_dialogue", self.Prompter_PsyEval.continuous_dialogue_scoring_prompt),
                ("context_summary", self.Prompter_PsyEval.context_summary_scoring_prompt),
                ("resistance_handling", self.Prompter_PsyEval.resistance_handling_scoring_prompt),
                ("humanized_expression", self.Prompter_PsyEval.humanized_expression_scoring_prompt),
            ]
        self.scoring_promptsV2 = [
                ("Concern", self.Prompter_PsyEvalV2.Concern_scoring_prompt),
                ("Expressiveness", self.Prompter_PsyEvalV2.Expressiveness_scoring_prompt),
                ("Resonate_or_capture_client_feelings", self.Prompter_PsyEvalV2.Resonate_or_capture_client_feelings_scoring_prompt),
                ("Warmth", self.Prompter_PsyEvalV2.Warmth_scoring_prompt),
                ("Attuned_to_clients_inner_world", self.Prompter_PsyEvalV2.Attuned_to_clients_inner_world_scoring_prompt),
                ("Understanding_cognitive_framework", self.Prompter_PsyEvalV2.Understanding_cognitive_framework_scoring_prompt),
                ("Understanding_feelings_or_inner_experience", self.Prompter_PsyEvalV2.Understanding_feelings_or_inner_experience_scoring_prompt),
                ("Acceptance_of_feelings_or_inner_experiences", self.Prompter_PsyEvalV2.Acceptance_of_feelings_or_inner_experiences_scoring_prompt),
                ("Responsiveness", self.Prompter_PsyEvalV2.Responsiveness_scoring_prompt),
                ("Dialogical_Logical_Consistency", self.Prompter_PsyEvalV2.Dialogical_Logical_Consistency_scoring_prompt),
                ("Conversational_Continuity", self.Prompter_PsyEvalV2.Conversational_Continuity_scoring_prompt),
                ("Handling_Resistance", self.Prompter_PsyEvalV2.Handling_Resistance_scoring_prompt),
                ("Summarization_Ability", self.Prompter_PsyEvalV2.Summarization_Ability_scoring_prompt),
                ("Ethics_Avoidance_of_Harmful_Suggestions_and_Positive_Guidance", self.Prompter_PsyEvalV2.Ethics_Avoidance_of_Harmful_Suggestions_and_Positive_Guidance_prompt),
                ("DialogueRhythm_And_ProcessManagement", self.Prompter_PsyEvalV2.DialogueRhythm_And_ProcessManagement_prompt),
                ("Fallacy_avoidance", self.Prompter_PsyEvalV2.Fallacy_avoidance_prompt)
            ]

    async def receive_data(self, llm: LLM_Core, data, max_retries=1, initial_delay=1):
        """
        Recibe datos de forma asíncrona del modelo LLM con reintentos en caso de error.
        """
        data_copy = copy.deepcopy(data)
        output = ""
        attempt = 0
        delay = initial_delay
        
        while attempt < max_retries:
            try:
                async for chunk in llm.async_model(data=data_copy):
                    json_string = chunk
                    chunk_data = json.loads(json_string)
                    if data_copy.get('stream', False):
                        output += chunk_data['choices'][0]['delta']['content']
                    else:
                        output += chunk_data['choices'][0]['message']['content']
                return output
            except Exception as e:
                error_message = traceback.format_exc()
                self.logger.error(f"Error al recibir datos: {error_message}")
                attempt += 1
                if attempt < max_retries:
                    self.logger.info(f"Reintentando la solicitud en {delay}s... ({attempt}/{max_retries})")
                    await asyncio.sleep(delay)
                    delay *= 1.1
                else:
                    self.logger.error("Se ha superado el número máximo de reintentos. La solicitud ha fallado.")
                    break
        return output if output else "<|_error_|>"

    def check_contains_sensitive(self, string_a: str) -> bool:
        """
        Comprueba si una cadena de texto contiene alguna de las palabras sensibles definidas en las herramientas.
        """
        string_list = self.tools.sensitive_words
        for item in string_list:
            if item in string_a:
                return True
        return False

    def _extract_content(self, text: str, pattern: str) -> Optional[str]:
        """
        Extrae contenido de un texto utilizando un patrón de regex.
        """
        if not text:
            return None
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    def _extract_dialog(self, text: str, pattern: str) -> Optional[str]:
        """
        Extrae y formatea un diálogo de múltiples turnos a partir de un texto.
        """
        dialog_text = self._extract_content(text, pattern)
        if not dialog_text:
            return None
            
        dialog_list = self.tools.parse_fields(dialog_text)
        
        first_user_index = next((i for i, item in enumerate(dialog_list) if item["role"] == "user"), None)
        if first_user_index is not None:
            dialog_list = dialog_list[first_user_index:]
        
        formatted_dialog = ""
        for item in dialog_list:
            role_name = "来访者" if item["role"] == "user" else "咨询师"
            formatted_dialog += f"{role_name}：" + item["content"].strip() + "\n"
                
        return formatted_dialog

    def extract_total_score(self, assessment_text: str) -> Optional[float]:
        """
        Extrae la puntuación total del texto de evaluación de forma robusta.
        """
        if not isinstance(assessment_text, str) or not assessment_text:
            return None

        keywords = ['Total Score', 'Final Score', 'Overall Score', '总评分', '综合得分', '最终得分', 'Score', '得分']
        keyword_pattern = '|'.join(keywords)
        number_pattern = r'(-?\d+(?:\.\d+)?)'
        
        flexible_pattern = re.compile(
            f'({keyword_pattern})'
            r'[\s\S]{0,50}?'
            f'({number_pattern})',
            flags=re.IGNORECASE
        )
        
        matches = flexible_pattern.findall(assessment_text)
        
        if matches:
            last_match = matches[-1]
            score_str = last_match[-1]
            try:
                return float(score_str)
            except (ValueError, TypeError):
                pass
        
        return None

    async def Generate_Response(self, choose_llm: LLM_Core, data_template, max_retries=6, pattern=None):
        """
        Genera una respuesta estándar del LLM.
        """
        for attempt in range(int(max_retries)):
            data_template2 = copy.deepcopy(data_template)
            Example_Response = await self.receive_data(choose_llm, data_template2)
            
            if self.check_contains_sensitive(Example_Response):
                self.logger.warning(f"Respuesta contiene palabras sensibles. Reintentando ({attempt + 1}/{max_retries}).")
                await asyncio.sleep(0.1)
                continue

            if pattern:
                Example_Response = self._extract_content(Example_Response, pattern)
                if Example_Response is None:
                    self.logger.warning(f"No se pudo encontrar el patrón en la respuesta. Reintentando ({attempt + 1}/{max_retries}).")
                    continue
            
            return Example_Response, "" # Devuelve tupla para consistencia
        return "<|_error_|>", ""

    async def Generate_EnhanceResponse(self, choose_llm: LLM_Core, data_template, max_retries=6, pattern=None):
        """
        Genera una respuesta mejorada del LLM.
        """
        for attempt in range(max_retries):
            template_copy = copy.deepcopy(data_template)
            template_copy["model"] = choose_llm.api_model
            response = await self.receive_data(choose_llm, template_copy)
            
            if pattern:
                Example_Response = self._extract_content(response, pattern)
                if Example_Response is None:
                    self.logger.warning(f"No se pudo encontrar el patrón en la respuesta mejorada. Reintentando ({attempt + 1}/{max_retries}).")
                    continue
            else:
                Example_Response = response
            
            if self.check_contains_sensitive(Example_Response):
                self.logger.warning(f"Respuesta mejorada contiene palabras sensibles. Reintentando ({attempt + 1}/{max_retries}).")
                await asyncio.sleep(0.1)
                continue

            return Example_Response
        return "<|_error_|>"

    async def Generate_PsyResponse(self, choose_llm: LLM_Core, data_template, max_retries=6, pattern=r'#+\s*多轮对话\s*#+(.*)'):
        """
        Genera una respuesta de diálogo psicológico.
        """
        for attempt in range(int(max_retries)):
            template_copy = copy.deepcopy(data_template)
            template_copy["model"] = choose_llm.api_model
            response = await self.receive_data(choose_llm, template_copy)
            response = response.replace("记住，", "").replace("听起来", "").replace("记得，", "")
            
            formatted_dialogue = self._extract_dialog(response, pattern)
            if not formatted_dialogue:
                self.logger.warning(f"No se pudo extraer el diálogo psicológico. Reintentando ({attempt + 1}/{max_retries}).")
                continue
            
            token_count = len(self.llm.tokenizer.encode(formatted_dialogue))
            if token_count < 800 or self.check_contains_sensitive(formatted_dialogue):
                self.logger.warning(f"Respuesta Psy no válida (longitud: {token_count} o sensible). Reintentando ({attempt + 1}/{max_retries}).")
                await asyncio.sleep(0.1)
                continue
            
            return formatted_dialogue
        return "<|_error_|>"

    async def Generate_EnhancePsyResponse(self, choose_llm: LLM_Core, data_template, max_retries=6, pattern=r'#+\s*强对话\s*#+(.*)'):
        """
        Genera una respuesta de diálogo psicológico mejorada.
        """
        for attempt in range(int(max_retries)):
            template_copy = copy.deepcopy(data_template)
            template_copy["model"] = choose_llm.api_model
            response = await self.receive_data(choose_llm, template_copy)
            response = response.replace("记住，", "").replace("听起来", "").replace("记得，", "")

            formatted_dialogue = self._extract_dialog(response, pattern)
            if not formatted_dialogue:
                self.logger.warning(f"No se pudo extraer el diálogo Psy mejorado. Reintentando ({attempt + 1}/{max_retries}).")
                continue

            token_count = len(self.llm.tokenizer.encode(formatted_dialogue))
            if token_count < 800 or self.check_contains_sensitive(formatted_dialogue):
                self.logger.warning(f"Respuesta Psy mejorada no válida (longitud: {token_count} o sensible). Reintentando ({attempt + 1}/{max_retries}).")
                await asyncio.sleep(0.1)
                continue
            
            return formatted_dialogue
        return "<|_error_|>"
        
    async def Judge_Quantity(self, choose_llm: LLM_Core, data_template, max_retry=3, refine_pattern=r'#+\s*改进建议\s*#*(.*)'):
        """
        Evalúa una respuesta, extrayendo puntuación, justificación y sugerencias de mejora.
        """
        data_template2 = copy.deepcopy(data_template)
        data_template2["model"] = choose_llm.api_model
        
        for attempt in range(max_retry):
            judge_text = await self.receive_data(choose_llm, data_template2)
            
            score = self.extract_total_score(judge_text)
            refine = self._extract_content(judge_text, refine_pattern) if refine_pattern else "wu"
            
            if score is None or refine is None or not (0 <= score <= 10):
                self.logger.warning(f"Fallo en la evaluación (score: {score}). Reintentando ({attempt + 1}/{max_retry}).\nRespuesta: {judge_text}")
                continue
                
            return score, judge_text
            
        return 3.0, ""
    
    
    async def process_stage_dialog_generate(self,input):
        input_copy = copy.deepcopy(input)
        if "Counseling_Report" in input_copy:
            Counseling_Report = input_copy["Counseling_Report"]
        else:
            Counseling_Report = input_copy["prompt"][-1]["content"]
        # if "PsyLLM" in self.llm.api_model:
        #     prompt = self.prompter.sentiment_prompt_Impedance_speical2.replace("Counseling_Report", Counseling_Report)
        # else:
        prompt = self.prompter.sentiment_prompt_Impedance_normal.replace("Counseling_Report", Counseling_Report)
       
        #result = await self.receive_data(self.llm,self.data_copy)
        if "chosen" in input_copy:
            del input_copy["chosen"]
        if "rejected" in input_copy:
            del input_copy["rejected"]
        attempt = 0
        max_retries = 12
        # print(self.llm.tokenizer.pad_token)
        # print(self.llm.tokenizer.pad_token_id)
        while attempt < max_retries:
            self.data_copy = copy.deepcopy(self.data_template)
            self.data_copy["messages"] = [
                #{"role": "system", "content": "你叫南希，来自健成星云的心理咨询师。"},#你的任务是与来访者模拟一段不少于30轮次的心理咨询的对话
                {"role": "user", "content": prompt}#
            ]
            # if "Qwen3" in self.llm.api_model:
            enable_thinking = False
            self.data_copy["extra_body"]={"chat_template_kwargs": {"enable_thinking": enable_thinking}}
            # if "simpsybot_D" in self.llm.api_model:
            #     self.data_copy["extra_body"] = {
            #             # 'repetition_penalty': 1.0,
            #             # 'top_k': 40,
            #             # 'min_p': 0.1,
            #             "ignore_eos": True,
            #             'stop_token_ids': [100001],
            #         }
            # if "SoulChat2___0-Qwen2-7B" in self.llm.api_model:
            #     self.data_copy["extra_body"] = {
            #         "ignore_eos": True,
            #         'stop_token_ids': [self.llm.tokenizer.pad_token_id],
            #         'stop': [self.llm.tokenizer.pad_token],
            #     }
            # if "CPsyCounX" in self.llm.api_model or "EmoLLMV3" in self.llm.api_model:
            #     self.data_copy["extra_body"] = {
            #         "ignore_eos": True,
            #         'stop_token_ids': [2],
            #     }
            response = await self.receive_data(self.llm,self.data_copy)
            if "Qwen3" in self.llm.api_model or "PsyLLM" in self.llm.api_model and enable_thinking==True:
                response = response.split("</think>")[-1].strip()
            #print(response)
            dialog_list = self.tools.parse_fields(response)
            # 检测连续的角色
            consecutive_issue = False
            for i in range(1, len(dialog_list)):
                if dialog_list[i]['role'] == dialog_list[i-1]['role']:
                    print(f"检测到连续的角色: {dialog_list[i]['role']} 在行 {i} 和 {i+1}")
                    consecutive_issue = True
                    break
            if consecutive_issue==False:
                response = ""
                for item in dialog_list:
                    if item["role"] == "user":
                        response += "来访者：" + item["content"].strip() + "\n"
                    elif item["role"] == "assistant":
                        response += "南希：" + item["content"].strip() + "\n"
                #sum += self.llm.tokenizer.pad_token
                print("弱答案生成成功")
            token_count = len(self.llm.tokenizer.encode(response))
            if token_count < 800:
                print(f"生成的结果 token 数量 {token_count} 少于 800，重新生成")
                if attempt == max_retries:
                    return "无法生成有效的增强答案。"
                await asyncio.sleep(0.1)
                continue
            else:
                break
        input_copy["chosen"] = [{"role": "assistant", "content": response}]
        return dict(input_copy)
    


    async def process_stage_reward_therapy_quality(self,input, update_prompt=None, use_intervent=False):
        input_copy = copy.deepcopy(input)
        multi_dialog=""
        total_score = 0 
        MAX_RETRIES=12
        if isinstance(input_copy["chosen"],list):
            for i in input_copy["chosen"]:
                if i["role"] == "user":
                    multi_dialog += "来访者: "+i["content"]+"\n"
                else:
                    multi_dialog += "心理咨询师: "+i["content"]+"\n"
        else:
            multi_dialog = input_copy["chosen"]

        def fix_unescaped_quotes(json_str):
            result = ''
            in_string = False
            escape = False
            for c in json_str:
                if escape:
                    result += c
                    escape = False
                elif c == '\\':
                    result += c
                    escape = True
                elif c == '"':
                    result += c
                    if not in_string:
                        in_string = True
                    else:
                        in_string = False
                elif c == '\n':
                    if in_string:
                        result += '\\n'
                    else:
                        result += c
                elif c == '“' or c == '”':
                    # 将中文双引号替换为转义的英文双引号
                    result += '\\"'
                elif c == '"' and in_string:
                    result += '\\"'
                else:
                    result += c
            return result
        
        async def evaluate_prompt(name, prompt, ans):
            """
            对单个提示进行评估。

            :param name: 评估名称。
            :param prompt: 评估的提示文本。
            :param ans: 要评估的答案。
            :param evaluation_index: 评估索引。
            :return: 单个评估的得分和评估结果。
            """
         
            a = 1.0
            TIMEOUT = 120  # 30秒超时时间
            score = 0
            ans = re.sub(r'[\(（].*?[\)）]', '', ans)
            query = prompt.replace("multi_dialog", ans)
            sum_judge = ""
            response = ""
            attempt = 0 
            #print(f"开始评估: {name} (索引: {evaluation_index})")
            while (score == 0 or score > 10) and attempt < MAX_RETRIES:
                
                try:
                    # Make request with timeout
                    start_time = time.time()
                    # 深拷贝数据模板并设置必要的字段
                    data = copy.deepcopy(self.data_template)
                    data["messages"] = [{"role": "user", "content": query}]
                    data["model"] = self.llm.api_model
                
                    # 发送请求并获取响应
                    #response = await self.llm.receive_data_GPT(data)
                    # 添加超时机制
                    response = await asyncio.wait_for(
                        self.llm.receive_data_GPT(data),
                        timeout=TIMEOUT
                    )
                    # 提取 JSON 字符串部分
                    start_idx = response.find('{')
                    end_idx = response.rfind('}') + 1
                    if start_idx == -1 or end_idx == 0:
                        raise ValueError("响应中未找到有效的 JSON 部分")
                    judge_string = response[start_idx:end_idx]
                    judge_string = fix_unescaped_quotes(judge_string)
                    try:
                        # 尝试解析 JSON
                        response_json = ast.literal_eval(judge_string)
                        score = float(response_json.get("得分", 0)) * a
                        #print(f"{name} 评估完成, 得分: {score:.2f} (通过 JSON 解析)")
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        # Fallback to regex if JSON parsing fails
                        #match = re.search(r'得分\s*[:：]\s*(\d+(\.\d+)?)', response)
                        pattern = r"\{\'得分\'\:\s*(\d+(\.\d+)?)\}"
                        match = re.search(pattern, response)
                        if match:
                            score = float(match.group(1)) * a
                            return name, score, response
                        # If no score found, raise to trigger retry
                        else:
                            attempt+=1
                            print(f"响应中未找到有效的得分 Attempt {attempt + 1}/{MAX_RETRIES}")
                            #await asyncio.sleep(0.1)  # Exponential backoff
                            continue

                except (asyncio.TimeoutError, Exception) as e:
                    elapsed = time.time() - start_time
                    print(f"Attempt {attempt + 1}/{MAX_RETRIES} failed for {name}: {str(e)} {response} "
                        f"(elapsed: {elapsed:.2f}s)")
                    # if attempt < MAX_RETRIES - 1:
                    #     await asyncio.sleep(0.1)  # Exponential backoff
                    attempt+=1
                    continue

            sum_judge = response  # 收集评估的响应
            return name, score, sum_judge
        J = self.scoring_promptsV2
        # 创建所有评估任务
        tasks = []
        for name, prompt in J:
            if update_prompt:
                if name in update_prompt:
                    print(f"正在评估 {name}...")
                    task = evaluate_prompt(name, prompt, multi_dialog)
                    tasks.append(task)
            else:
                task = evaluate_prompt(name, prompt, multi_dialog)
                tasks.append(task)
        # 并行执行所有评估任务
        results = await asyncio.gather(*tasks)
        if update_prompt:
            # 累计总得分和评估结果
            total_score = 0
            sum_judge = ""

            for name, score, judge in results:
                input_copy[name] = judge
                input_copy[name+"_score"] = score
            # 累计总得分和评估结果
            total_score = 0
            sum_judge = ""
            for name, prompt in J:
                total_score += input_copy[name+"_score"]
                sum_judge += input_copy[name]
            input_copy["Reward_label"] = 10*(total_score / len(J))
            score = input_copy["Reward_label"]
            print(f"总得分: {score}")
            input_copy["Reward_reasoning"] = sum_judge
            return dict(input_copy)
        else:
            # 累计总得分和评估结果
            total_score = 0
            sum_judge = ""
            for name, score, judge in results:
                total_score += score
                sum_judge += judge
                input_copy[name] = judge
                input_copy[name+"_score"] = score
            input_copy["Reward_label"] = 10*(total_score / len(J))
            #input_copy["Reward_label"] = total_score
            score = input_copy["Reward_label"]
            print(f"总得分: {score}")
            input_copy["Reward_reasoning"] = sum_judge
            return dict(input_copy)
    
