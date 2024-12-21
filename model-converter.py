from optimum.onnxruntime import ORTModelForQuestionAnswering
from transformers import AutoTokenizer

model_checkpoint = "csarron/mobilebert-uncased-squad-v2"
save_directory = "/Users/prakashr/Documents/codebase/tmp/mobilebert/"

# Load a model from transformers and export it to ONNX
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

#for classification
#ort_model = ORTModelForSequenceClassification.from_pretrained(model_checkpoint, export=True)

#for QA
ort_model = ORTModelForQuestionAnswering.from_pretrained(model_checkpoint, export=True)


# Save the ONNX model and tokenizer
ort_model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)