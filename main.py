from fastai.vision.all import *
from fastai.vision.widgets import *
import requests
import io

# URL of the raw .pkl file on GitHub
url = "https://github.com/KyleAndruczk/cpsc171-hw1-fin/raw/main/bears_model.pkl"
response = requests.get(url)
response.raise_for_status()
model_data = io.BytesIO(response.content)

learn_inf = load_learner(model_data, cpu=True)

btn_upload = widgets.FileUpload()
btn_classify = widgets.Button(description='Classify')
out_pl = widgets.Output()
lbl_pred = widgets.Label()

def on_classify(b):
    if not btn_upload.data:
        lbl_pred.value = 'Please upload an image first'
        return
    
    lbl_pred.value = ''
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'


btn_classify.on_click(on_classify)

display(widgets.VBox([
    widgets.Label('Select your bear!'), 
    btn_upload,
    btn_classify,
    out_pl, 
    lbl_pred
]))