package com.r7b7.tc;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class DistillBertONNXInference {
    public static void main(String[] args) throws OrtException, IOException {

        String modelPath = "path/to/model.onnx";

        try (OrtEnvironment env = OrtEnvironment.getEnvironment();
                OrtSession session = env.createSession(modelPath, new OrtSession.SessionOptions())) {
            System.out.println("ONNX model loaded successfully!");

            HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer
                    .newInstance("ruvalappil/distilbert-base-uncased-finetuned-sst-2-english-ONNX");

            String text = "Hello, how are you?";
            Encoding encoding = tokenizer.encode(text);
            long[][] inputIds = { encoding.getIds() }; 
            long[][] attentionMask = { encoding.getAttentionMask() };

            OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, inputIds);
            OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, attentionMask);

            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", inputIdsTensor);
            inputs.put("attention_mask", attentionMaskTensor);

            OrtSession.Result results = session.run(inputs);

            float[][] logits = (float[][]) results.get(0).getValue(); // Output logits
            System.out.println("Model output (logits): " + Arrays.toString(logits[0]));

            int predictedClass = logits[0][0] > logits[0][1] ? 0 : 1;
            System.out.println("Predicted class: " + predictedClass);

            inputIdsTensor.close();
            attentionMaskTensor.close();
        }
    }

}
