# application of vlm in remote sensing image captioning 

This project demonstrates how to fine-tune a transformer-based image captioning model and also the evaluation of the output of this model using different metrics . The model is trained on a custom image-caption dataset and incorporates techniques to freeze and adapt specific layers, optimizing performance and resource usage.

## 🚀 Features

- Implements LoRA to fine-tune decoder layers.
- Efficient training by freezing encoder layers (except the last 4).
- Custom dataset loading and preprocessing.
- Generates image captions with pretrained vision-language models.
- Visualizes training loss over time.

## 📂 Project Structure

```
main.ipynb           # Core notebook for model setup, training, and testing
README.md            # Project overview and usage
```

## 🧰 Requirements

- Python 3.8+
- PyTorch
- Transformers
- PEFT (Parameter-Efficient Fine-Tuning)
- Datasets
- Matplotlib

Install dependencies:

```bash
pip install torch torchvision transformers peft datasets matplotlib
```

## 🧪 Model Architecture

- **Backbone**: Pretrained vision encoder (vit)
- **Decoder**: GPT-2 with LoRA applied to `c_attn` and `c_proj` layers
- **LoRA Settings**:
  - `r=8`
  - `alpha=32`
  - `dropout=0.1`

## 🧠 Training Strategy

- Freeze all encoder layers except the last 4.
- Apply LoRA to decoder layers only.
- Use Hugging Face `Trainer` API.
- Track loss during training.

## 📊 Results

- LoRA significantly reduces the number of trainable parameters.
- The model successfully learns to generate meaningful captions from image inputs.
- Loss curve visualized at the end of training.



## 📈 Visualization

Training loss is plotted to show convergence behavior.

## 🛠️ Acknowledgements

- Hugging Face Transformers
- PEFT library by Hugging Face
- Datasets API



