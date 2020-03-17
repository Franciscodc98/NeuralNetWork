package menoita.ai.neuralNetwork;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.ejml.simple.SimpleMatrix;

//import lombok.Data;

//@Data
public class NeuralNetwork {


	private List<Neuron[]> neuronsLayer = new ArrayList<>();
	private List<SimpleMatrix> connectionsLayer = new ArrayList<>();

	private ActivationStrategy activation;

	private float learningRate = 0.01f;

	private Random random = new Random();

	//training data status
	private List<Double> costs = new ArrayList<>(); 	



	public NeuralNetwork(ActivationStrategy activation,float learningRate,int inputs , int[] hiddenLayers, int outputs) {
		this.activation = activation;
		this.learningRate = learningRate;
		initializeNeurons(inputs,hiddenLayers,outputs);
		initializeWeight();
	}





	/**
	 * Initialise the neuron layers.
	 * Neurons in hidden and output layer have bias randomly generated with values
	 * between -0.5 and 0.5
	 * 
	 * @param inputsNumber number of inputs
	 * @param hiddenLayers number of hidden layers with number of neurons for layer
	 * @param outputs number of outputs
	 */
	private void initializeNeurons(int inputsNumber, int[] hiddenLayers, int outputsNumber) {
		//create input neurons
		Neuron[] inputs = new Neuron[inputsNumber];

		for (int i = 0; i <inputs.length; i++)
			inputs[i] = new Neuron();

		neuronsLayer.add(inputs);

		//create hidden layer neurons
		for (int i = 0; i < hiddenLayers.length; i++) {
			if(hiddenLayers[i] > 0) {
				Neuron[] hiddens = new Neuron[hiddenLayers[i]];
				//generate layer bias with value between -0.5 and 0.5
				float bias = random.nextFloat()-0.5f;
				for (int j = 0; j < hiddens.length; j++) 
					//Create a hidden neuron with a layer bias
					hiddens[j] = new Neuron(bias);

				neuronsLayer.add(hiddens);
			}else 
				System.err.println("Invalid hidden Layer number detected "+" position: "+i+" value:"+hiddenLayers[i]);

		}

		//create output layer neurons

		Neuron[] outputs = new Neuron[outputsNumber];

		for (int i = 0; i <outputs.length; i++)
			outputs[i] = new Neuron(random.nextFloat()-0.5f);

		neuronsLayer.add(outputs);
	}




	/**
	 * Initialise weight matrix with random weights
	 */
	private void initializeWeight() {
		for (int i = 0; i < neuronsLayer.size()-1; i++) {
			System.out.println("Layer "+i);//TODO set to normal distribution
			SimpleMatrix m = SimpleMatrix.random_DDRM(neuronsLayer.get(i+1).length, neuronsLayer.get(i).length,-Math.pow(neuronsLayer.get(i).length, -0.5),Math.pow(neuronsLayer.get(i).length, -0.5), random);
			getConnectionsLayer().add(m);
			m.print();
		}
		System.out.println("---------------------------------------------------//---------------------------------------------------------");
	}




	/**
	 * 
	 * @param inputs
	 * @return
	 */
	public float[] predict(float[] inputs) {
		if(inputs.length != neuronsLayer.get(0).length)
			throw new IllegalArgumentException("Number of inputs unexpected, was expecting: "+neuronsLayer.get(0).length+" inputs but got "+inputs.length);

		//Set input neurons
		for (int i = 0; i < inputs.length; i++) 
			neuronsLayer.get(0)[i].setOutput(inputs[i]);

		//Set hidden and output value
		for (int i = 0; i < getConnectionsLayer().size(); i++) {

			//Gets a neuron matrix
			SimpleMatrix neuronMatrix = new SimpleMatrix(Neuron.toOutPutMatrix(neuronsLayer.get(i)));
			SimpleMatrix valueMatrix = getConnectionsLayer().get(i).mult(neuronMatrix);


			//Set's neuron output using matrix multiplication and activation strategy
			for (int j = 0; j < neuronsLayer.get(i+1).length; j++)
				neuronsLayer.get(i+1)[j].setOutput(activation.activateFunction((float)(valueMatrix.get(j, 0) + neuronsLayer.get(i+1)[j].getBias())));

		}

		Neuron[] outputNeurons = neuronsLayer.get(neuronsLayer.size()-1);
		float[] output = new float[outputNeurons.length];
		for (int i = 0; i < outputNeurons.length; i++) 
			output[i] = outputNeurons[i].getOutput();

		return output;
	}



	/**
	 * 
	 * @param inputs
	 * @param targets
	 */
	public void train(float[] inputs,float[] targets) {
		if(targets.length != neuronsLayer.get(neuronsLayer.size()-1).length)
			throw new IllegalArgumentException("Number of targets elements unexpected, was expecting: "+neuronsLayer.get(neuronsLayer.size()-1).length+" inputs but got "+targets.length);

		//get neural network prediction
		float[] outputs = predict(inputs);
		Neuron[] outputNeurons = neuronsLayer.get(neuronsLayer.size()-1);

		double cost = 0;

		//calculate errors
		for (int i = 0; i < outputs.length; i++) {
			outputNeurons[i].setError(targets[i] - outputs[i]);
			//calculate cost
			cost += Math.pow(targets[i] - outputs[i], 2);
		}
		//cost is not used at the moment but in the future will be useful to calculate Neural network efficiency
		costs.add(cost);

		//backpropagation 		DeltaW = LR x ErrorMatrix x Gradient * TransposedInputs
		for (int i = getConnectionsLayer().size()-1; i >= 0; i--) {
			
			//error matrix
			SimpleMatrix errorMatrix = new SimpleMatrix(Neuron.toErrorMatrix(neuronsLayer.get(i+1)));
			//calculate derivative matrix that is activation function derivative of output
			SimpleMatrix derivativeMatrix = new SimpleMatrix(Neuron.toDerivativeMatrix(neuronsLayer.get(i+1),activation));
			//gradient matrix is the matrix [ek . Sk(1-Sk) ] --> error * derivative value of output
			SimpleMatrix gradientMatrix = derivativeMatrix.elementMult(errorMatrix).scale(learningRate);
			//Transposed outputs of previous layer or saying with other words this layer inputs transposed
			SimpleMatrix transposedInputMatrix = new SimpleMatrix(Neuron.toOutPutMatrix(neuronsLayer.get(i))).transpose();
			//calculating the delta
			SimpleMatrix delta = gradientMatrix.mult(transposedInputMatrix);
			
			//adjust the weight
			getConnectionsLayer().set(i,getConnectionsLayer().get(i).plus(delta));
			
			//calculate hidden errors = weight transposed * errorMatrix
			SimpleMatrix nextLayerErrors = getConnectionsLayer().get(i).transpose().mult(errorMatrix);
			//setting the new errors
			for (int j = 0; j < neuronsLayer.get(i).length && i !=0; j++)
				neuronsLayer.get(i)[j].setError((float)nextLayerErrors.get(j, 0));
		}
	}


	/**
	 * @return the connectionsLayer
	 */
	public List<SimpleMatrix> getConnectionsLayer() {
		return connectionsLayer;
	}





	public static void main(String[] args) {
		NeuralNetwork nn = new NeuralNetwork(ActivationStrategy.ReLu, 0.01f, 2,new int[]{5,10}, 1);
		//training loop
		for (int i = 0; i < 90000; i++) {
			Random r = new Random();
			
			switch (r.nextInt(4)) {
			case 0: {
				nn.train(new float[] {1f,0f}, new float[] {1f});
				break;
			}case 1: {
				nn.train(new float[] {0f,1f}, new float[] {1f});
				break;
			}case 2: {
				nn.train(new float[] {0f,0f}, new float[] {0f});
				break;
			}case 3: {
				nn.train(new float[] {1f,1f}, new float[] {0f});
				break;
			}
			default:
				throw new IllegalArgumentException("Unexpected value");
			}
		}
		//test
		nn.predict(new float[] {1f,0f});
		System.out.println("*-------XOR TEST-------*");
		System.out.println("|     |  F  |   T  |");
		System.out.println("|  F  |  0  |   1  |");
		System.out.println("|  T  |  1  |   0  |");
		System.out.println("------------------------");
		System.out.println("For false and false : "+Arrays.toString(nn.predict(new float[] {0f,0f})));
		System.out.println("For true and true   : "+Arrays.toString(nn.predict(new float[] {1f,1f})));
		System.out.println("For true and false  : "+Arrays.toString(nn.predict(new float[] {1f,0f})));
		System.out.println("For false and true  : "+Arrays.toString(nn.predict(new float[] {0f,1f})));
		System.out.println("\n\n---------------------------------------------------//---------------------------------------------------------");
		for (int i = 0; i < nn.getConnectionsLayer().size(); i++) {
			System.out.println("Layer "+i);
			nn.getConnectionsLayer().get(i).print();
		}
		System.out.println("---------------------------------------------------//---------------------------------------------------------");
	}
}
