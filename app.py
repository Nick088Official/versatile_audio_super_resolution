import gradio as gr
from audiosr import super_resolution, build_model

def inference(audio_file, model_name, guidance_scale, ddim_steps, seed):
    audiosr = build_model(model_name=model_name)
    
    waveform = super_resolution(
        audiosr,
        audio_file,
        seed,
        guidance_scale=guidance_scale,
        ddim_steps=ddim_steps
    )
    
    return (44100, waveform)

iface = gr.Interface(
    fn=inference, 
    inputs=[
        gr.Audio(type="filepath", label="Input Audio"),
        gr.Dropdown(["basic", "speech"], value="basic", label="Model"),
        gr.Slider(1, 10, value=3.5, step=0.1, label="Guidance Scale", info="Guidance scale (Large => better quality and relavancy to text; Small => better diversity)"),  
        gr.Slider(1, 100, value=50, step=1, label="DDIM Steps", info="The sampling step for DDIM"),
        gr.Number(value=42, precision=0, label="Seed", info="Changing this value (any integer number) will lead to a different generation result.")
    ],
    outputs=gr.Audio(type="numpy", label="Output Audio"),
    title="AudioSR",
    description="Audio Super Resolution with AudioSR"
)

iface.launch(share=False)
