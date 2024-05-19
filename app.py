import torch
import gradio as gr
from src.model import DRModel
from torchvision import transforms as T

CHECKPOINT_PATH = "artifacts/dr-model.ckpt"
model = DRModel.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu")
model.eval()

labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# Define the prediction function
def predict(input_img):
    input_img = transform(input_img).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(input_img)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in labels}
    return confidences


# Set up the Gradio app interface
dr_app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Diabetic Retinopathy Classification",
    description="Diabetic Retinopathy Classification Web Application",
    examples=[
        "data/sample/13_left.jpeg",
        "data/sample/16_right.jpeg",
        "data/sample/17_right.jpeg",
        "data/sample/10_right.jpeg",
    ],
)

# Run the Gradio app
if __name__ == "__main__":
    dr_app.launch(share=True)
