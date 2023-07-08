import argparse

# Create the argument parser
parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('-n', '--model_name', type=str, help='Specify a model name', default='vilm/vietcuna-3b', choices=['vilm/vietcuna-3b', 'vilm/vietcuna-7b'])
parser.add_argument('--four-bit', action='store_true', help='Whether to use 4bit')

args = parser.parse_args()

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

if args.model_name == 'vietcuna-3b':
    model_name = 'vilm/vietcuna-3b'
elif args.model_name == 'vietcuna-7b':
    model_name = 'vilm/vietcuna-7b-alpha'
else:
    raise ValueError("Unsupported model_name. Please choose either 'vietcuna-3b' or 'vietcuna-7b'.")

print(f"Starting to load the model {model_name} into memory")

m = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=args.four_bit,
    torch_dtype=torch.bfloat16,
    device_map={"": 0}
)

tok = AutoTokenizer.from_pretrained(model_name)
tok.bos_token_id = 1

stop_token_ids = [2]

print(f"Successfully loaded the model {model_name} into memory")

import datetime
import os
from threading import Event, Thread
from uuid import uuid4

import gradio as gr
import requests

max_new_tokens = 512 if model_name == 'vilm/vietcuna-3b' else 1024
start_message = """Below is an instruction that describes a task.
Write a response that appropriately completes the request.\n\n"""

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def convert_history_to_text(history):
    text = start_message
    # text = start_message + "".join(
    #     [
    #         "".join(
    #             [
    #                 f"### Instruction: {item[0]}\n",
    #                 f"### Response: {item[1]}\n",
    #             ]
    #         )
    #         for item in history[:-1]
    #     ]
    # )
    text += "".join(
        [
            "".join(
                [
                    f"### Instruction: {history[-1][0]}\n",
                    f"### Response: {history[-1][1]}\n",
                ]
            )
        ]
    )
    return text


def log_conversation(conversation_id, history, messages, generate_kwargs):
    logging_url = os.getenv("LOGGING_URL", None)
    if logging_url is None:
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "history": history,
        "messages": messages,
        "generate_kwargs": generate_kwargs,
    }

    try:
        requests.post(logging_url, json=data)
    except requests.exceptions.RequestException as e:
        print(f"Error logging conversation: {e}")


def user(message, history):
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
    print(f"history: {history}")
    # Initialize a StopOnTokens object
    stop = StopOnTokens()

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = convert_history_to_text(history)

    # Tokenize the messages string
    input_ids = tok(messages, return_tensors="pt").input_ids
    input_ids = input_ids.to(m.device)
    streamer = TextIteratorStreamer(tok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
        stopping_criteria=StoppingCriteriaList([stop]),
    )

    stream_complete = Event()

    def generate_and_signal_complete():
        m.generate(**generate_kwargs)
        stream_complete.set()

    def log_after_stream_complete():
        stream_complete.wait()
        log_conversation(
            conversation_id,
            history,
            messages,
            {
                "top_k": top_k,
                "top_p": top_p,
                "temperature": temperature,
                "repetition_penalty": repetition_penalty,
            },
        )

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    t2 = Thread(target=log_after_stream_complete)
    t2.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        history[-1][1] = partial_text
        yield history


def get_uuid():
    return str(uuid4())


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    conversation_id = gr.State(get_uuid)
    gr.Markdown(
        """<h1><center>Vietcuna Demo</center></h1>
"""
    )
    chatbot = gr.Chatbot().style(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
            ).style(container=False)
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Gửi")
                stop = gr.Button("Dừng")
                clear = gr.Button("Xoá")
    with gr.Row():
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.7,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                with gr.Column():
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=0.9,
                            minimum=0.0,
                            maximum=1,
                            step=0.01,
                            interactive=True,
                            info=(
                                "Sample from the smallest possible set of tokens whose cumulative probability "
                                "exceeds top_p. Set to 1 to disable and sample from all tokens."
                            ),
                        )
                with gr.Column():
                    with gr.Row():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=0,
                            minimum=0.0,
                            maximum=200,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                        )
                with gr.Column():
                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            value=1.0,
                            minimum=1.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Penalize repetition — 1.0 to disable.",
                        )
    with gr.Row():
        gr.Markdown(
            "Disclaimer: The model can produce factually incorrect output, and should not be relied on to produce "
            "factually accurate information. The model was trained on various public datasets; while great efforts "
            "have been taken to clean the pretraining data, it is possible that this model could generate lewd, "
            "biased, or otherwise offensive outputs.",
            elem_classes=["disclaimer"],
        )
    with gr.Row():
        gr.Markdown(
            "[Privacy policy](https://https://github.com/vilm-ai/vietcuna)",
            elem_classes=["disclaimer"],
        )

    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=128, concurrency_count=2)
demo.launch()
