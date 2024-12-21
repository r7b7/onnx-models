package com.r7b7.qa;

import java.nio.LongBuffer;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

public class MobilebertONNXInference {
    public static void main(String[] args) throws Exception {
        // Load the ONNX model
        String modelPath = "/path/to/model.onnx";
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, options);

        // Input question and context - Sample
        String question = "What is the capital of France?";
        String context = "Paris is the capital and largest city of France.";

        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer
                .newInstance(Paths.get("/path/to/tokenizer.json"));
        Encoding encoding = tokenizer.encode(question, context);

        OnnxTensor inputIdsTensor = OnnxTensor.createTensor(env, toLongBuffer(encoding.getIds()),
                new long[] { 1, encoding.getIds().length });
        OnnxTensor attentionMaskTensor = OnnxTensor.createTensor(env, toLongBuffer(encoding.getAttentionMask()),
                new long[] { 1, encoding.getAttentionMask().length });
        OnnxTensor tokenTypeIdsTensor = OnnxTensor.createTensor(env, toLongBuffer(encoding.getTypeIds()),
                new long[] { 1, encoding.getTypeIds().length });

        Map<String, OnnxTensor> inputs = new HashMap<>();
        inputs.put("input_ids", inputIdsTensor);
        inputs.put("attention_mask", attentionMaskTensor);
        inputs.put("token_type_ids", tokenTypeIdsTensor);

        OrtSession.Result results = session.run(inputs);
        float[][] startLogits = (float[][]) results.get(0).getValue();
        float[][] endLogits = (float[][]) results.get(1).getValue();

        int startIndex = argMax(startLogits[0]);
        int endIndex = argMax(endLogits[0]);

        // Decode and append to the final answer
        List<String> tokens = List.of(encoding.getTokens());
        StringBuilder answer = new StringBuilder();
        for (int i = startIndex; i <= endIndex; i++) {
            String token = tokens.get(i);
            if (token.startsWith("##")) {
                answer.append(token.substring(2));
            } else if (answer.length() > 0) {
                answer.append(" ").append(token);
            } else {
                answer.append(token);
            }
        }
        System.out.println("Answer: " + answer.toString());

        // Clean up
        session.close();
        inputIdsTensor.close();
        attentionMaskTensor.close();
        tokenTypeIdsTensor.close();
    }

    private static int argMax(float[] array) {
        int maxIndex = 0;
        float maxValue = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < array.length; i++) {
            if (array[i] > maxValue) {
                maxValue = array[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private static LongBuffer toLongBuffer(long[] list) {
        LongBuffer buffer = LongBuffer.allocate(list.length);
        for (long i : list)
            buffer.put(i);
        buffer.rewind();
        return buffer;
    }

}
