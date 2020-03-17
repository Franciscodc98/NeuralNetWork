package menoita.ai.neuralNetwork;

//import lombok.Data;
//import lombok.NoArgsConstructor;

//@Data
//@NoArgsConstructor
public class Neuron {
	
	private float output;
	private float bias;
	private float error;
	
	

	/**
	 * @param bias
	 */
	public Neuron(float bias) {
		this.bias = bias;
	}
	
	public Neuron() {}
	
	public static float[][] toOutPutMatrix(Neuron[] neurons){
		float[][] matrix = new float[neurons.length][1];
		for (int i = 0; i < neurons.length; i++) 
			matrix[i][0] = neurons[i].getOutput();
		return matrix;
	}


	public static float[][] toErrorMatrix(Neuron[] neurons) {
		float[][] matrix = new float[neurons.length][1];
		for (int i = 0; i < neurons.length; i++) 
			matrix[i][0] = neurons[i].getError();
		return matrix;
	}

	

	/**
	 * Generate a float matrix that maps neurons output values to activation derivative strategy
	 * @param neurons
	 * @param activation
	 * @return
	 */
	public static float[][] toDerivativeMatrix(Neuron[] neurons, ActivationStrategy activation) {
		float[][] matrix = new float[neurons.length][1];
		for (int i = 0; i < neurons.length; i++) 
			matrix[i][0] = activation.derivate(neurons[i].getOutput());
		return matrix;
	}

	/**
	 * @return the output
	 */
	public float getOutput() {
		return output;
	}

	/**
	 * @return the bias
	 */
	public float getBias() {
		return bias;
	}

	/**
	 * @return the error
	 */
	public float getError() {
		return error;
	}

	/**
	 * @param output the output to set
	 */
	public void setOutput(float output) {
		this.output = output;
	}

	/**
	 * @param bias the bias to set
	 */
	public void setBias(float bias) {
		this.bias = bias;
	}

	/**
	 * @param error the error to set
	 */
	public void setError(float error) {
		this.error = error;
	}
}
