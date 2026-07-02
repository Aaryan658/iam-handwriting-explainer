import gradio as gr


def hello():
    return "hello, deploy check"


demo = gr.Interface(fn=hello, inputs=None, outputs="text")
demo.launch()
