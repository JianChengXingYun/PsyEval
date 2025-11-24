# -*- encoding:utf-8 -*- 
import argparse
import asyncio
import os
from pathlib import Path
import random
import time
import json
import concurrent.futures
from tqdm import tqdm
from Swimming_Pool_Async.Duplicater import Duplicater
from Swimming_Pool_Async.Process_Controller import Process_Controller
from Swimming_Pool_Async.LLM_Core import LLM_Core
from Swimming_Pool_Async.Tools import Tools
from Swimming_Pool_Async.Duplicater import ProcessStage
import pandas as pd
# Define the list of models to process
MODELS = [
    "PsyLLMV3-Large-250519",
    "PsyLLMV3-Mini-250519",
    "claude-3-7-sonnet-20250219",
    "gemini-2.5-pro-exp-03-25",
    "gpt-4o-2024-11-20",
    "qwen-max-2025-01-25",
    "gpt-4o-mini-2024-07-18",
    "doubao-1-5-pro-32k-250115",
    "Qwen2.5-72B-Instruct",
    "doubao-1-5-lite-32k-250115",
    "Qwen2.5-32B-Instruct",
    "gemini-2.5-flash-preview-04-17",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "SoulChat2___0-Qwen2-7B",
    "Xinyuan-LLM-14B-0428",
    "simpsybot_D",
    "Qwen3-32B_NoThink",
    "GLM-4-9B-0414",
    "GLM-4-32B-0414",
    "CPsyCounX",
    #"PsyLLMV3-Large-250607",
]

parser = argparse.ArgumentParser()
parser.add_argument("--inputmodel", default="PsyLLMV4_5-Kairosa-251121-Instruct-SFT-GLM", type=str, help="Specific model to process or 'all' for all models")
parser.add_argument("--base_url", default="http://localhost:6012/v1", type=str)
parser.add_argument("--filename", default="sampled_2000_Counseling_Report_DPO_Tag.jsonl", type=str)
args = parser.parse_args()

# If a specific model is provided, process only that model; otherwise, process all models
models_to_process = [args.inputmodel] if args.inputmodel and args.inputmodel != "all" else MODELS
filename = args.filename
base_url = args.base_url
# Define scoring prompts and weights
scoring_prompts = [
    "Concern",
    "Expressiveness",
    "Resonate_or_capture_client_feelings",
    "Warmth",
    "Attuned_to_clients_inner_world",
    "Understanding_cognitive_framework",
    "Understanding_feelings_or_inner_experience",
    "Acceptance_of_feelings_or_inner_experiences",
    "Responsiveness",
    "Dialogical_Logical_Consistency",
    "Conversational_Continuity",
    "Handling_Resistance",
    "Summarization_Ability",
    "Ethics_Avoidance_of_Harmful_Suggestions_and_Positive_Guidance",
    "DialogueRhythm_And_ProcessManagement",
    "Fallacy_avoidance",
]
scoring_weights = {
    'Concern': 0.5,
    'Expressiveness': 1.0,
    'Resonate_or_capture_client_feelings': 0.5,
    'Warmth': 1.0,
    'Attuned_to_clients_inner_world': 0.5,
    'Understanding_cognitive_framework': 1.0,
    'Understanding_feelings_or_inner_experience': 0.5,
    'Acceptance_of_feelings_or_inner_experiences': 0.5,
    'Responsiveness': 0.5,
    'Dialogical_Logical_Consistency': 1.0,
    'Conversational_Continuity': 0.5,
    'Handling_Resistance': 0.5,
    'Summarization_Ability': 0.5,
    'Ethics_Avoidance_of_Harmful_Suggestions_and_Positive_Guidance': 0.5,
    "DialogueRhythm_And_ProcessManagement": 0.5,
    "Fallacy_avoidance": 0.5
}

# Define leaderboard paths
LEADERBOARD_DIR = "result_new"
LEADERBOARD_TXT = os.path.join(LEADERBOARD_DIR, "PsyEval3-plus_leaderboard.txt")
LEADERBOARD_XLSX = os.path.join(LEADERBOARD_DIR, "PsyEval3-plus_leaderboard.xlsx")

# Ensure the leaderboard directory exists
os.makedirs(LEADERBOARD_DIR, exist_ok=True)

def delete_score_files():
    """Delete all dsv3_score_psyeval3-plus.jsonl files in subdirectories of LEADERBOARD_DIR"""
    try:
        count = 0
        # Walk through all subdirectories in the result directory
        for root, dirs, files in os.walk(LEADERBOARD_DIR):
            for file in files:
                if file == "dsv3_score_psyeval3-plus.jsonl":
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                    count += 1
                    print(f"Deleted: {file_path}")
        
        print(f"Total {count} dsv3_score_psyeval3-plus.jsonl files deleted.")
    except Exception as e:
        print(f"Error deleting files: {e}")

async def process_model(model_name, base_url):
    print(f"\n{'='*50}")
    print(f"Processing model: {model_name}")
    print(f"{'='*50}\n")
    model_name = model_name
    
    #model_name = model_name+ "_NoThink"
    # Create directories and set filenames
    transcription_cache_dir = Path("result_new/" + model_name)
    transcription_cache_dir.mkdir(parents=True, exist_ok=True)
    filename2 = "result_new/" + model_name + '/' + "output.jsonl"
    sc = "result_new/" + model_name + '/' + "dsv3_score_psyeval3-plus.jsonl"
    print(f"Output file: {filename2}")
    # Initialize components
    duplicater = Duplicater(tokenizer_path="Qwen2.5-7B-Instruct", sensitive_words="")
    # Initialize LLM core with the current model
    llm = LLM_Core(
        duplicater.tokenizer,
        use_async=True,
        api_model=model_name,
        base_url=base_url,
        api_key='EMPTY'
    )

    llm2 = LLM_Core(
        duplicater.tokenizer,
        api_model="deepseek-v3-241226",#deepseek-v3-241226 doubao-1-5-pro-32k-250115 deepseek-r1-250120 
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key='XXXXXXXXXXXXXXXXXXXXXX'
    )
    
    # Initialize tools and controllers
    tools = Tools(filename, duplicater.tokenizer)
    tools.filename = filename
    tools.filename2 = filename2
    processer = Process_Controller(llm=llm, tools=tools)
    processer_GPT = Process_Controller(llm=llm2, tools=tools)
    
    # Read data
    input_data, total_lines, processed_count = processer.tools.Read_Document_PsyEval()
    
    # Set constants
    task_count0 = 64
    task_count1 = 64
    MAX_CONCURRENT_TASKS = 32
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
    
    # Initialize progress bars
    pbar_stage1 = tqdm(total=task_count0, desc=f"Stage1: ASYNC_SCENARIO_TO_REPORT ({model_name})", position=0)
    pbar_stage2 = tqdm(total=task_count1, desc=f"Stage2: REWARD_THERAPY_QUALITY ({model_name})", position=1)
    
    # Create event for progress bar full
    progress_bar_full_event = asyncio.Event()
    
    # Set task count and update totals
    pbar_stage1.total = task_count0
    pbar_stage2.total = task_count1
    
    outputs_stage1_all = []
    
    # Define stage1 process function within the model processing scope
    async def process_stage1(item, pbar1, progress_bar_full_event):
        async with semaphore:
            outputs = []
            try:
                async for output in duplicater.process_and_filter_async(
                    processer,
                    [item],
                    stage=ProcessStage.DIALOG_GENERATE
                ):  
                    if output is None:
                        continue
                    pbar1.update(1)
                    outputs.append(output)
                    tools.save_to_file([output], tools.filename2)
                    print("Information saved.")
                    # Check if progress bar is full
                    if pbar1.n >= pbar1.total:
                        print("Stage 2 progress bar is full, canceling all tasks and stopping code.")
                        progress_bar_full_event.set()
                        # Cancel all other tasks
                        for task in asyncio.all_tasks():
                            if task is not asyncio.current_task():
                                task.cancel()
                    return output
            except asyncio.CancelledError:
                print(f"Stage1 task canceled: {item}")
                raise

    # Define stage2 process function within the model processing scope
    # Define stage2 process function within the model processing scope
    async def process_stage2(output_item, pbar2, progress_bar_full_event):
        async with semaphore:
            outputs = []
            try:
                # Stage 2: REWARD_THERAPY_QUALITY_IMPEDANCE
                async for output in duplicater.process_and_filter_async(
                    processer_GPT,
                    [(output_item,None,False)],  # Pass as list
                    stage=ProcessStage.REWARD_THERAPY_QUALITY
                ):
                    pbar2.update(1)
                    outputs.append(output)
                    tools.save_to_file([output], sc)
                    print("Information saved.")
                    # Check if progress bar is full
                    if pbar2.n >= pbar2.total:
                        print("Stage 2 progress bar is full, canceling all tasks and stopping code.")
                        progress_bar_full_event.set()
                        # Cancel all other tasks
                        for task in asyncio.all_tasks():
                            if task is not asyncio.current_task():
                                task.cancel()
                    return output
            except asyncio.CancelledError:
                print(f"Stage1 task canceled: {output_item}")
                raise
    
    with tqdm(total=task_count0, desc=f"Instruction progress ({model_name})") as pbar:
        while processed_count < total_lines:
            input_data, total_lines, processed_count = processer.tools.Read_Document_PsyEval()
            total_lines = task_count0
            # Update progress bar
            pbar.n = processed_count
            doing = len(input_data) - processed_count
            print(f"Total lines: {total_lines}, Processed: {processed_count}, Doing: {doing}")
            pbar.refresh()
            # Update instruction progress bar
            pbar_stage1.n = processed_count
            pbar_stage1.refresh()
            random.seed(time.time())
            # Stage 1: Process all input data
            if doing > 0 and processed_count < task_count0:
                try:
                    stage1_tasks = [asyncio.create_task(process_stage1(item, pbar_stage1, progress_bar_full_event)) for item in input_data[:doing]]
                    stage1_results = await asyncio.gather(*stage1_tasks)
                    
                    for result in stage1_results:
                        if isinstance(result, list):
                            outputs_stage1_all.extend(result)
                        else:
                            outputs_stage1_all.append(result)
                    
                    outputs_stage1_all = [item for item in outputs_stage1_all if item is not None]
                    print(f"Total outputs from Stage1: {len(outputs_stage1_all)}")
                except asyncio.CancelledError:
                    print("Task canceled.")
                    break
            else:
                print("Reached total processing count.")
                break
    processer.tools.filename = filename2
    processer.tools.filename2 = sc
    input_data, total_lines, processed_count = processer.tools.Read_Document_01()
    with tqdm(total=task_count1, desc=f"Eval progress ({model_name})") as pbar:
        while processed_count < task_count1:
            # Prepare for stage 2
            processer.tools.filename = filename2
            processer.tools.filename2 = sc
            input_data, total_lines, processed_count = processer.tools.Read_Document_01()
            outputs_stage1_all = input_data
            pbar_stage2.n = processed_count
            pbar_stage2.refresh()
            
            # Stage 2: Process all Stage1 outputs
            if outputs_stage1_all:
                try:
                    stage2_tasks = [asyncio.create_task(process_stage2(output, pbar_stage2, progress_bar_full_event)) for output in outputs_stage1_all]
                    stage2_results = await asyncio.gather(*stage2_tasks)
                    
                    outputs_stage2_all = []
                    for result in stage2_results:
                        if not isinstance(result, list):
                            outputs_stage2_all.append(result)
                except asyncio.CancelledError:
                    print("Task canceled.")
                    break
            print("Information saved.")
        
        # Merge results
        processer.tools.filename = sc
        processer.tools.filename2 = ""
        input_data, total_lines, processed_count = processer.tools.Read_Document_01()
        outputs = input_data
        pbar.update(len(input_data))
        print("Processing completed.")
        
        # Calculate scores
        dimension_totals = {dim: 0.0 for dim in scoring_prompts}
        valid_output_count = 0
        
        for output in outputs:
            if all(dim in output for dim in scoring_prompts):
                valid_output_count += 1
                for dim in scoring_prompts:
                    a = scoring_weights[dim]
                    score = a*float(output[dim + "_score"])
                    dimension_totals[dim] += score
            else:
                print(f"Warning: Output missing dimension scores: {str(output)[:30]}")
        
        # Calculate average for each dimension
        dimension_averages = {dim: (dimension_totals[dim] / valid_output_count if valid_output_count > 0 else 0.0) for dim in scoring_prompts}
        print(dimension_averages)
        
        # Calculate total score
        total_score = sum(dimension_averages[dim] for dim in scoring_prompts)
        print(f"TOTAL_SCORE for {model_name} = {total_score}")
        
        # Process leaderboard
        update_leaderboard(model_name, total_score, dimension_averages)
        
        return total_score

def update_leaderboard(model_name, total_score, dimension_averages):
    # Initialize empty list to store all entries
    leaderboard_entries = []
    
    # Check if leaderboard.txt exists and read existing data
    if os.path.exists(LEADERBOARD_TXT):
        with open(LEADERBOARD_TXT, 'r', encoding='utf-8') as leaderboard_file:
            lines = leaderboard_file.readlines()
            # Skip header row
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                parts = line.split('\t')
                if len(parts) != len(scoring_prompts) + 2:
                    print(f"Warning: Cannot parse line: {line}")
                    continue
                model = parts[0]
                scores = parts[1:]
                try:
                    scores = [float(score) for score in scores]
                    entry = [model] + scores
                    leaderboard_entries.append(entry)
                except ValueError:
                    print(f"Warning: Cannot convert scores to float: {parts[1:]} for model: {model}")
                    continue
    
    # Remove existing entries with the same model name
    leaderboard_entries = [entry for entry in leaderboard_entries if entry[0] != model_name]
    
    # Add new entry with total score
    new_entry = [model_name, total_score] + [dimension_averages[dim] for dim in scoring_prompts]
    leaderboard_entries.append(new_entry)
    
    # Create DataFrame
    df = pd.DataFrame(leaderboard_entries, columns=["Model Name", "Total Score"] + [dim.replace('_', ' ').title() for dim in scoring_prompts])
    
    # Sort by total score in descending order
    df.sort_values(by="Total Score", ascending=False, inplace=True)
    
    # Save to leaderboard.txt
    df.to_csv(LEADERBOARD_TXT, sep='\t', index=False, float_format='%.2f')
    print(f"Model name and average scores saved and sorted to {LEADERBOARD_TXT}.")
    
    # Save to Excel
    df.to_excel(LEADERBOARD_XLSX, index=False)
    print(f"Leaderboard saved as Excel file: {LEADERBOARD_XLSX}")

async def main():
    # First, delete existing score files if requested
    #delete_score_files()
    
    # Process each model
    for model_name in models_to_process:
        total_score = await process_model(model_name, base_url)
        print(f"TOTAL_SCORE for {model_name} = {total_score}")
    
    # Display final leaderboard
    if os.path.exists(LEADERBOARD_XLSX):
        leaderboard_df = pd.read_excel(LEADERBOARD_XLSX)
        print("\nFinal Leaderboard:")
        print(leaderboard_df.to_string(index=False))
    
    print("\nAll models processed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
