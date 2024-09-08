package me.eliknowles;

import me.eliknowles.neuronlib.NeuralNetwork;

public class Main {
    public static void main(String[] args) {
        int inputSize = 3;
        int hiddenLayers = 2;
        int[] hiddenLayerSizes = {4, 3};
        int outputSize = 1;

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputSize, hiddenLayers, hiddenLayerSizes, outputSize);

        // Training data
        double[] input = {1.0, 0.5, -1.5};
        double[] desiredOutput = {1.0}; // Expected output

        // Train the network
        for (int i = 0; i < 1000; i++) {  // Training for 1000 epochs
            neuralNetwork.train(input, desiredOutput);
        }

        // Forward pass after training
        double[] output = neuralNetwork.forward(input);

        System.out.println("Output: ");
        for (double i : output) {
            System.out.println(i);
        }
    }
}