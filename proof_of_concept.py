import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

def create_generator():
    model_id = "meta-llama/Llama-3.3-70B-Instruct"
    quantisation_config = BitsAndBytesConfig(load_in_4bit=True)

    quantised_model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.bfloat16, quantization_config=quantisation_config)
    tokeniser = AutoTokenizer.from_pretrained(model_id, torch_dtype=torch.bfloat16)

    return pipeline("text-generation", model=quantised_model, tokenizer=tokeniser)

generator = create_generator()

def update_generated_code(code_output):
    with open("generated_code.py", "w") as f:
        f.write(code_output.strip("```").strip("python"))

def execute_code():
    import importlib

    module_name = 'generated_code'

    # Import the module dynamically
    updated_lib = importlib.import_module(module_name)

    # Reload the module to get the latest version
    importlib.reload(updated_lib)

    updated_lib.run()

chat_history = []

def produce(txt):
    messages = [
    {"role": "system", "content": "You are a code generation model that will only output a Python function (without calling it) based on the task given in the input and nothing else! The code generated ABSOLUTELY NEEDS to be in a function called run()."},
    {"role": "system", "content": """You are given the following information about the machine environment that you are going to generate code for:
     - The working directory is /home/ubuntu/Documents/
     - There are photos scaterred across different folders and subfolders in the working directory
     - The operating system is Linux"""},
]

    messages.append({"role": "user", "content": txt})

    outputs = generator(messages, max_new_tokens=1000)

    code_output = outputs[0]["generated_text"][-1]["content"]

    update_generated_code(code_output)
    execute_code()

    chat_history.append([txt, "Task completed!"])

    return chat_history



with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="What do you want your assistant to do?")
        txt.submit(produce, txt, chatbot)

demo.launch()