package edu.stanford.nlp.optimization;

import java.text.DecimalFormat;
import java.text.NumberFormat;

import edu.stanford.nlp.util.Pair;

/**
 * <p>
 * Stochastic Gradient Descent Minimizer
 *
 *
 * The basic way to use the minimizer is with a null constructor, then
 * the simple minimize method:
 * <p/>
 * <p><code>Minimizer smd = new SGDMinimizer();</code>
 * <br><code>DiffFunction df = new SomeDiffFunction(); //Note that it must be a incidence of AbstractStochasticCachingDiffFunction</code>
 * <br><code>double tol = 1e-4;</code>
 * <br><code>double[] initial = getInitialGuess();</code>
 * <br><code>int maxIterations = someSafeNumber;
 * <br><code>double[] minimum = qnm.minimize(df,tol,initial,maxIterations);</code>
 * <p/>
 * Constructing with a null constructor will use the default values of
 * <p>
 * <br><code>batchSize = 15;</code>
 * <br><code>initialGain = 0.1;</code>
 * <p/>
 * <p/>
 *
 * @author <a href="mailto:akleeman@stanford.edu">Alex Kleeman</a>
 * @version 1.0
 * @since 1.0
 */
public class SGDMinimizer extends StochasticMinimizer {
  

  @Override
  public void shutUp() {
    this.quiet = true;
  }

  public void setBatchSize(int batchSize) {
    bSize = batchSize;
  }


  private static NumberFormat nf = new DecimalFormat("0.000E0");

  public SGDMinimizer() {
  }

  public SGDMinimizer(double SGDGain, int batchSize){
    this(SGDGain,batchSize,50);
  }
  
  public SGDMinimizer(double SGDGain, int batchSize, int passes){
    this(SGDGain,batchSize,passes,Long.MAX_VALUE,false);
  }

  public SGDMinimizer(double SGDGain, int batchSize, int passes, boolean outputToFile){
    this(SGDGain, batchSize, passes, Long.MAX_VALUE ,outputToFile );
  }
  
  public SGDMinimizer(double SGDGain, int batchSize, int passes, long maxTime){
    this(SGDGain,batchSize,passes,maxTime,false);
  }
  
  public SGDMinimizer(double SGDGain, int batchSize, int passes, long maxTime, boolean outputToFile){
    StochasticMinimizer.bSize = batchSize;
    StochasticMinimizer.gain = SGDGain;
    this.numPasses = passes;
    this.outputIterationsToFile = outputToFile;
    this.maxTime = maxTime;
  }

  
  @Override
  protected String getName(){
    int g = (int) gain*1000;
      return "SGD" + bSize + "_g" + g;
  }
 
  
  public Pair <Integer,Double> tune(Function function, double[] initial,long msPerTest,double gainLow,double gainHigh){
    this.quiet = true;
    StochasticMinimizer.gain = tuneGain(function, initial, msPerTest, gainLow,gainHigh);
    StochasticMinimizer.bSize = tuneBatch(function,initial,msPerTest,1);
    
    return new Pair<Integer,Double>(StochasticMinimizer.bSize , StochasticMinimizer.gain);
  }
  
  @Override
  public Pair<Integer,Double> tune(Function function,double[] initial, long msPerTest){
    return this.tune(function, initial, msPerTest, 1e-7,1.0);

  }
  
  @Override
  protected void takeStep(AbstractStochasticCachingDiffFunction dfunction){
    for(int i = 0; i < x.length; i++){    
      newX[i] = x[i] - gain*gainSchedule(k,5*numBatches)*grad[i];
    }
  }
  





  public static void main(String[] args) {
    // optimizes test function using doubles and floats
    // test function is (0.5 sum(x_i^2 * var_i)) ^ PI
    // where var is a vector of random nonnegative numbers
    // dimensionality is variable.
    final int dim = 500000;
    final double maxVar = 5;
    final double[] var = new double[dim];
    double[] init = new double[dim];

    for (int i = 0; i < dim; i++) {
      init[i] = ((i + 1) / (double) dim - 0.5);//init[i] = (Math.random() - 0.5);
      var[i] = maxVar * (i + 1) / dim;
    }

    final double[] grads = new double[dim];

    final DiffFunction f = new DiffFunction() {
      public double[] derivativeAt(double[] x) {
        double val = Math.PI * valuePow(x, Math.PI - 1);
        for (int i = 0; i < dim; i++) {
          grads[i] = x[i] * var[i] * val;
        }
        return grads;
      }

      public double valueAt(double[] x) {
        return 1.0 + valuePow(x, Math.PI);
      }

      private double valuePow(double[] x, double pow) {
        double val = 0.0;
        for (int i = 0; i < dim; i++) {
          val += x[i] * x[i] * var[i];
        }
        return Math.pow(val * 0.5, pow);
      }

      public int domainDimension() {
        return dim;
      }
    };



    SGDMinimizer min = new SGDMinimizer();

    min.minimize(f, 1.0E-4, init);

  }

}
