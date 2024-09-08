package me.eliknowles;

import java.util.Random;

public class NeuralNetwork {

    private int inputSize;
    private int outputSize;
    private int hiddenLayers;
    private int[] hiddenLayerSizes;

    private double[][][] weights;
    private double[][] biases;
    private double learningRate = 0.01; // Learning rate for gradient descent

    // Constructor to initialize the network
    public NeuralNetwork(int inputSize, int hiddenLayers, int[] hiddenLayerSizes, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenLayers = hiddenLayers;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.outputSize = outputSize;

        // Initialize weights and biases
        initializeWeightsAndBiases();
    }

    // Initialize weights and biases randomly
    private void initializeWeightsAndBiases() {
        weights = new double[hiddenLayers + 1][][]; // Including input to first hidden layer and hidden to output
        biases = new double[hiddenLayers + 1][];

        Random rand = new Random();

        // Input layer to first hidden layer
        weights[0] = new double[hiddenLayerSizes[0]][inputSize];
        biases[0] = new double[hiddenLayerSizes[0]];
        for (int i = 0; i < hiddenLayerSizes[0]; i++) {
            for (int j = 0; j < inputSize; j++) {
                weights[0][i][j] = rand.nextDouble() - 0.5; // Random initialization
            }
            biases[0][i] = rand.nextDouble() - 0.5;
        }

        // Hidden layers
        for (int l = 1; l < hiddenLayers; l++) {
            weights[l] = new double[hiddenLayerSizes[l]][hiddenLayerSizes[l - 1]];
            biases[l] = new double[hiddenLayerSizes[l]];
            for (int i = 0; i < hiddenLayerSizes[l]; i++) {
                for (int j = 0; j < hiddenLayerSizes[l - 1]; j++) {
                    weights[l][i][j] = rand.nextDouble() - 0.5;
                }
                biases[l][i] = rand.nextDouble() - 0.5;
            }
        }

        // Last hidden layer to output layer
        weights[hiddenLayers] = new double[outputSize][hiddenLayerSizes[hiddenLayers - 1]];
        biases[hiddenLayers] = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < hiddenLayerSizes[hiddenLayers - 1]; j++) {
                weights[hiddenLayers][i][j] = rand.nextDouble() - 0.5;
            }
            biases[hiddenLayers][i] = rand.nextDouble() - 0.5;
        }
    }

    // Forward propagation
    public double[] forward(double[] input) {
        double[] currentLayerOutput = input;

        // Propagate through hidden layers
        for (int l = 0; l < hiddenLayers; l++) {
            currentLayerOutput = activate(matrixVectorMultiply(weights[l], currentLayerOutput), biases[l]);
        }

        // Propagate through output layer
        return activate(matrixVectorMultiply(weights[hiddenLayers], currentLayerOutput), biases[hiddenLayers]);
    }

    // Training function using backpropagation
    public void train(double[] input, double[] desiredOutput) {
        // Perform a forward pass
        double[][] layerOutputs = new double[hiddenLayers + 2][];
        layerOutputs[0] = input;

        for (int l = 0; l < hiddenLayers; l++) {
            layerOutputs[l + 1] = activate(matrixVectorMultiply(weights[l], layerOutputs[l]), biases[l]);
        }

        // Output layer
        layerOutputs[hiddenLayers + 1] = activate(matrixVectorMultiply(weights[hiddenLayers], layerOutputs[hiddenLayers]), biases[hiddenLayers]);

        // Compute output error (output - desiredOutput)
        double[] outputLayerError = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            outputLayerError[i] = layerOutputs[hiddenLayers + 1][i] - desiredOutput[i];
        }

        // Backpropagate error through output layer
        double[][] layerDeltas = new double[hiddenLayers + 1][];
        layerDeltas[hiddenLayers] = outputLayerError;

        for (int l = hiddenLayers - 1; l >= 0; l--) {
            layerDeltas[l] = new double[hiddenLayerSizes[l]];
            for (int i = 0; i < hiddenLayerSizes[l]; i++) {
                double deltaSum = 0;
                for (int j = 0; j < layerDeltas[l + 1].length; j++) {
                    deltaSum += layerDeltas[l + 1][j] * weights[l + 1][j][i];
                }
                layerDeltas[l][i] = deltaSum * derivativeReLU(layerOutputs[l + 1][i]);
            }
        }

        // Update weights and biases using gradient descent
        for (int l = hiddenLayers; l >= 0; l--) {
            for (int i = 0; i < weights[l].length; i++) {
                for (int j = 0; j < weights[l][i].length; j++) {
                    weights[l][i][j] -= learningRate * layerDeltas[l][i] * layerOutputs[l][j];
                }
                biases[l][i] -= learningRate * layerDeltas[l][i];
            }
        }
    }

    // Matrix-vector multiplication
    private double[] matrixVectorMultiply(double[][] matrix, double[] vector) {
        double[] result = new double[matrix.length];
        for (int i = 0; i < matrix.length; i++) {
            result[i] = 0;
            for (int j = 0; j < vector.length; j++) {
                result[i] += matrix[i][j] * vector[j];
            }
        }
        return result;
    }

    // Activation function (ReLU in this case)
    private double[] activate(double[] z, double[] bias) {
        double[] activated = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            activated[i] = Math.max(0, z[i] + bias[i]); // ReLU activation
        }
        return activated;
    }

    // Derivative of ReLU for backpropagation
    private double derivativeReLU(double z) {
        return z > 0 ? 1 : 0;
    }
}