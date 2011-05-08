package edu.stanford.nlp.optimization;

import java.io.Serializable;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.List;

import edu.stanford.nlp.sequences.SeqClassifierFlags;

/**
 * <p>
 * Stochastic Gradient Descent To Quasi Newton Minimizer
 *
 *  An experimental minimizer which takes a stochastic function (one implementing AbstractStochasticCachingDiffFunction)
 * and executes SGD for the first couple passes,  During the final iterations a series of approximate hessian vector
 * products are built up...  These are then passed to the QNminimizer so that it can start right up without the typical
 * delay.
 *
 * @author <a href="mailto:akleeman@stanford.edu">Alex Kleeman</a>
 * @version 1.0
 * @since 1.0
 */
public class SGDToQNMinimizer implements Minimizer<DiffFunction>,Serializable  {

  private int k;
  private int bSize = 15;
  private boolean quiet = false;

  public boolean outputIterationsToFile = false;
  public int outputFrequency = 10;
  public double gain = 0.1;
  private List<double[]> gradList = null;
  private List<double[]> yList = null;
  private List<double[]> sList = null;
  private List<double[]> tmpYList = null;
  private List<double[]> tmpSList = null;
  private int memory = 5;
  public int SGDPasses = -1;
  public int QNPasses = -1;
  private int hessSampleSize = 50;
  private int QNMem = 10;
  private boolean toTest = false;

  public void shutUp() {
    this.quiet = true;
  }

  public void setBatchSize(int batchSize) {
    bSize = batchSize;
  }


  private static NumberFormat nf = new DecimalFormat("0.000E0");


  public SGDToQNMinimizer(SeqClassifierFlags flags){
    this.bSize = flags.stochasticBatchSize;
    this.gain = flags.initialGain;
    this.SGDPasses = flags.SGDPasses;
    this.QNPasses = flags.QNPasses;
    this.QNMem = flags.QNsize;
    this.outputIterationsToFile = flags.outputIterationsToFile;
    this.toTest = flags.testObjFunction;
    this.hessSampleSize = flags.SGD2QNhessSamples; 
  }

  public SGDToQNMinimizer(double SGDGain, int batchSize, int sgdPasses, int qnPasses, int hessSamples, int QNMem){
    this( SGDGain, batchSize, sgdPasses, qnPasses, hessSamples, QNMem,false);
  }
  
  public SGDToQNMinimizer(double SGDGain, int batchSize, int sgdPasses, int qnPasses, int hessSamples, int QNMem,boolean outputToFile){
    this.bSize = batchSize;
    this.gain = SGDGain;
    this.SGDPasses = sgdPasses;
    this.QNPasses = qnPasses;
    this.QNMem = QNMem;
    this.outputIterationsToFile = outputToFile;
    this.hessSampleSize = hessSamples;
  }

  public SGDToQNMinimizer(double SGDGain, int batchSize, int SGDPasses, int QNPasses){
    this(SGDGain,batchSize,SGDPasses,QNPasses,50,10);
  }

  public void setQNMem(int mem){
    QNMem = mem;
  }

  public void setHessSampleSize(int size){
    hessSampleSize = size;
  }

  
  protected String getName(){
    int g = (int) (gain * 1000);
    return "SGD2QN" + bSize + "_g" + g;
  }

  public double[] minimize(DiffFunction function, double functionTolerance, double[] initial) {
    return minimize(function,functionTolerance,initial,-1);
  }
  
  public double[] minimize(DiffFunction function, double functionTolerance, double[] initial, int maxIterations) {
    sayln("SGDToQNMinimizer called on function of " + function.domainDimension() + " variables;");

    // check for stochastic derivatives
    if (!(function instanceof AbstractStochasticCachingDiffFunction)) {
      throw new UnsupportedOperationException();
    }
    AbstractStochasticCachingDiffFunction dfunction = (AbstractStochasticCachingDiffFunction) function;

    dfunction.method = StochasticCalculateMethods.GradientOnly;
    
    ScaledSGDMinimizer sgd = new ScaledSGDMinimizer(this.gain,this.bSize,this.SGDPasses,1,this.outputIterationsToFile);
    QNMinimizer qn = new QNMinimizer(this.QNMem,true);
    
    double[] x = sgd.minimize(dfunction, functionTolerance, initial, this.SGDPasses);
    
    QNMinimizer.QNInfo qnInfo = qn.new QNInfo(sgd.sList , sgd.yList);
    qnInfo.d = sgd.diag;    
    
    qn.minimize(dfunction, functionTolerance, x, this.QNPasses, qnInfo);

    System.err.println("");
    System.err.println("Minimization complete.");
    System.err.println("");
    System.err.println("Exiting for Debug");
    return x;
  }


  private void sayln(String s) {
    if (!quiet) {
      System.err.println(s);
    }
  }

  private void say(String s) {
    if (!quiet) {
      System.err.print(s);
    }
  }





}
