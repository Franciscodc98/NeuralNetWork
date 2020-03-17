package menoita.ai.neuralNetwork;

public enum ActivationStrategy {

	ReLu, SoftPlus, Sigmoid;



	public float derivate(float value) {
		switch (this) {
		case ReLu: 
			return value <=0 ? 0 : 1;	// u(x) function
		case SoftPlus: 
			return (float) (Math.pow(Math.E, value) / ( 1 + Math.pow(Math.E, value) )); // e^x / ( 1+ e^x )
		case Sigmoid: 
			return sigmoid(value)*(1-sigmoid(value));
		default:
			throw new IllegalArgumentException("Unexpected value: " + this);
		}
	}

	
	
	public float activateFunction(float value) {
		switch (this) {
		case ReLu: 
			return Math.max(0, value);
		case SoftPlus: 
			return (float) (Math.log(1+ Math.pow(Math.E, value))); // ln ( 1+ e^x )
		case Sigmoid: 
			return sigmoid(value);
		default:
			throw new IllegalArgumentException("Unexpected value: " + this);
		}
	}
	


	private float sigmoid(float value) {
		return (float) (1 / ( 1 + Math.pow(Math.E, -value)));	// 1 / ( 1 + e^-x)
	}
}
