package edu.stanford.nlp.optimization;

import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.util.Timing;

import java.text.NumberFormat;
import java.text.DecimalFormat;
import java.io.PrintWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.List;
import java.util.ArrayList;
import java.util.Random;
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
public abstract class StochasticMinimizer implements Minimizer {


  public boolean outputIterationsToFile = false;
  public int outputFrequency = 1000;
  public static double gain = 0.1;
  
  protected double[] x, newX, grad, newGrad,v;
  protected int numBatches;
  protected int k;
  protected static int bSize = 15;
  protected boolean quiet = false; 
  protected List<double[]> gradList = null;
  protected int memory = 10;
  protected int numPasses = -1;
  protected Random gen = new Random(1);
  protected PrintWriter file = null;
  protected PrintWriter infoFile = null;
  protected long maxTime = Long.MAX_VALUE;


  public void shutUp() {
    this.quiet = true;
  }

  private static NumberFormat nf = new DecimalFormat("0.000E0");

  
  abstract protected String getName();

  abstract protected void takeStep(AbstractStochasticCachingDiffFunction dfunction);
  
  
    /*
    This is the scaling factor for the gains to ensure convergence
  */
  protected double gainSchedule(int it, double tau){
    return (tau / (tau + it));
  }
  
  /*
   * This is used to smooth the gradients, providing a more robust calculation which
   * generally leads to a better routine.
   */
  
  protected double[] smooth(List<double[]> toSmooth){
    double[] smoothed = new double[toSmooth.get(0).length];

    for(double[] thisArray:toSmooth){
      ArrayMath.pairwiseAddInPlace(smoothed,thisArray);
    }

    ArrayMath.multiplyInPlace(smoothed,1/((double) toSmooth.size() ));
    return smoothed;
  }

  
  public static class InvalidElementException extends Throwable {
    public InvalidElementException(String s) {
      super(s);
    }
  }

  private void ensureFinite(double[] vect) throws InvalidElementException{
    ensureFinite(vect,"");
  }

  private void ensureFinite(double[] vect, String name) throws InvalidElementException{

    for(int i=0;i<vect.length;i++){
      if(Double.isNaN(vect[i])){
        throw new InvalidElementException("NAN found in " + name + " element " + i);
      }else if(Double.isInfinite(vect[i])){
        throw new InvalidElementException("Infinity found in " + name + " element " + i);
      }
    } 

  }


  private void initFiles() {

    if(outputIterationsToFile){

      String fileName = getName() + ".output";
      String infoName = getName() + ".info";

      try{
        file = new PrintWriter(new FileOutputStream(fileName),true);
        infoFile = new PrintWriter(new FileOutputStream(infoName),true);
      }
      catch (IOException e){
        System.err.println("Caught IOException outputing data to file: " + e.getMessage());
        System.exit(1);
      }
    };
    
  }

  
  public abstract Pair<Integer,Double> tune(Function function,double[] initial, long msPerTest);
  
  public double tuneDouble(edu.stanford.nlp.optimization.Function function, double[] initial, long msPerTest,PropertySetter<Double> ps,double lower,double upper){
    return this.tuneDouble(function, initial, msPerTest, ps, lower, upper, 1e-3*Math.abs(upper-lower));
  }
  
public double tuneDouble(edu.stanford.nlp.optimization.Function function, double[] initial, long msPerTest,PropertySetter<Double> ps,double lower,double upper,double TOL){
    
    double[] xtest = new double[initial.length];
    double opt = 0.0;
    double min = Double.POSITIVE_INFINITY;
    double result;
    this.maxTime = msPerTest;
    double prev = Double.POSITIVE_INFINITY;
    // check for stochastic derivatives
    if (!(function instanceof AbstractStochasticCachingDiffFunction)) {
      throw new UnsupportedOperationException();
    }
    AbstractStochasticCachingDiffFunction dfunction = (AbstractStochasticCachingDiffFunction) function;
    
    List<Pair<Double,Double>> res = new ArrayList<Pair<Double,Double>>();
    Pair<Double,Double> best = new Pair<Double,Double>(lower,Double.POSITIVE_INFINITY); //this is set to lower because the first it will always use the lower first, so it has to be best
    Pair<Double,Double> low = new Pair<Double,Double>(lower,Double.POSITIVE_INFINITY);
    Pair<Double,Double> high = new Pair<Double,Double>(upper,Double.POSITIVE_INFINITY);
    Pair<Double,Double> cur = new Pair<Double,Double>();
    Pair<Double,Double> tmp = new Pair<Double,Double>();
    
    List<Double> queue = new ArrayList<Double>();
    queue.add(lower);
    queue.add(upper);
    //queue.add(0.5* (lower + upper));
    
    int it = 1;
    boolean  toContinue = true;
    this.numPasses = 10000;
    
    do{
        System.arraycopy(initial, 0, xtest, 0, initial.length);
        if(queue.size() != 0){
          cur.first = queue.remove(0);
        }else{
          cur.first = 0.5*( low.first() + high.first() );
        }
        
        ps.set(cur.first() ); 
        
        System.err.println("");
        System.err.println("About to test with batchsize:  " + StochasticMinimizer.bSize + "  gain: "  + StochasticMinimizer.gain + " and  " + ps.toString() + " set to  " + cur.first());
        
        xtest = this.minimize(function, 1e-100, xtest);
        
        if(Double.isNaN( xtest[0] ) ){
          cur.second = Double.POSITIVE_INFINITY;
        } else {
          cur.second = dfunction.valueAt(xtest);
        }
        
        if( cur.second() < best.second() ){
          
          copyPair(best,tmp);
          copyPair(cur,best);
          
          if(tmp.first() > best.first()){
            copyPair(tmp,high); // The old best is now the upper bound
          }else{
            copyPair(tmp,low);  // The old best is now the lower bound
          }
          queue.add( 0.5 * ( cur.first() + high.first()  ) ); // check in the right interval next
        } else if ( cur.first() < best.first() ){
          copyPair(cur,low);
        } else if( cur.first() > best.first()){
          copyPair(cur,high);
        }
        
        if( Math.abs( low.first() - high.first() ) < TOL   ) {
          toContinue = false;
        }
        
        res.add(new Pair<Double,Double>(cur.first(),cur.second()));
        
       it += 1;
        System.err.println("");
        System.err.println("Final value is: " + nf.format(cur.second()));
        System.err.println("Optimal so far using " + ps.toString() + " is: "+ best.first() );
    } while(toContinue);
     
    
    //output the results to screen.
    System.err.println("-------------");
    System.err.println(" RESULTS          ");
    System.err.println(ps.getClass().toString());
    System.err.println("-------------");
    System.err.println("  val    ,    function after " + msPerTest + " ms");
    for(int i=0;i<res.size();i++ ){
      System.err.println(res.get(i).first() + "    ,    " + res.get(i).second() );
     }
    System.err.println("");
    System.err.println("");
    
    return best.first();
    
  }
  
  private void copyPair(Pair<Double,Double> from, Pair<Double,Double> to){
    to.first = (double) from.first();
    to.second = (double) from.second();
  }
  

  private class setGain implements PropertySetter<Double>{
    StochasticMinimizer parent = null;

    public setGain(StochasticMinimizer min){parent = min;}
    
    public void set(Double in){
      StochasticMinimizer.gain = in ;
    }
  }
  
  
  public double tuneGain(Function function, double[] initial, long msPerTest, double lower, double upper){
    
    return tuneDouble(function,initial,msPerTest,new setGain(this),lower,upper);
    
  }
  
  
  
  public int  tuneBatch(Function function, double[] initial, long msPerTest, int bStart){
    
    double[] xtest = new double[initial.length];
    int bOpt = 0;
    double min = Double.POSITIVE_INFINITY;
    double result;
    this.maxTime = msPerTest;
    double prev = Double.POSITIVE_INFINITY;
    
    // check for stochastic derivatives
    if (!(function instanceof AbstractStochasticCachingDiffFunction)) {
      throw new UnsupportedOperationException();
    }
    AbstractStochasticCachingDiffFunction dfunction = (AbstractStochasticCachingDiffFunction) function;
    
    int bMax = dfunction.dataDimension();
    int b = bStart;
    int it = 1;
    boolean  toContinue = true;
    
    do{
        System.arraycopy(initial, 0, xtest, 0, initial.length);
        System.err.println("");
        System.err.println("Testing with batchsize:  " + b );
        StochasticMinimizer.bSize = b;
        this.quiet = true;
        this.minimize(function, 1e-100, xtest);
        result = dfunction.valueAt(xtest);
        
        if(it == 1){
          b *=2;
        }
        
        if( result < min ){
          min = result;
          bOpt = StochasticMinimizer.bSize;
          b *= 2;
          prev = result;
        }else if(result < prev){
          b*=2;
          prev = result;
        }else if(result > prev){
          toContinue = false;
        }
        
       
        System.err.println("");
        System.err.println("Final value is: " + nf.format(result));
        System.err.println("Optimal so far is:  batchsize: " + bOpt );
    } while(toContinue);
        
    
    return bOpt;
    
  }
  
  public Pair<Integer,Double> tune(Function function, double[] initial, long msPerTest,List<Integer> batchSizes, List<Double> gains){

    double[] xtest = new double[initial.length];
    int bOpt = 0;
    double gOpt = 0.0;
    double min = Double.POSITIVE_INFINITY;
    
    double[][] results = new double[batchSizes.size()][gains.size()];
    
    this.maxTime = msPerTest;
    
    for( int b=0;b<batchSizes.size();b++){
      for(int g=0;g<gains.size();g++){
        System.arraycopy(initial, 0, xtest, 0, initial.length);
        StochasticMinimizer.bSize = batchSizes.get(b);
        StochasticMinimizer.gain = gains.get(g);
        System.err.println("");
        System.err.println("Testing with batchsize: " + StochasticMinimizer.bSize + "    gain:  " + nf.format(StochasticMinimizer.gain) );
        this.quiet = true;
        this.minimize(function, 1e-100, xtest);
        results[b][g] = function.valueAt(xtest);
        
        if( results[b][g] < min ){
          min = results[b][g];
          bOpt = StochasticMinimizer.bSize;
          gOpt = StochasticMinimizer.gain;
        }
       
        System.err.println("");
        System.err.println("Final value is: " + nf.format(results[b][g]));
        System.err.println("Optimal so far is:  batchsize: " + bOpt + "   gain:  " + nf.format(gOpt) );
        
      }
    }
    
    return new Pair<Integer,Double>(bOpt,gOpt);
    
  }
  
  
  //This can be filled if an extending class needs to initialize things.
  protected void init(AbstractStochasticCachingDiffFunction func){
  }
  
  
  
  public double[] minimize(Function function, double functionTolerance, double[] initial) {
    return minimize(function, functionTolerance, initial, -1);
  }


  public double[] minimize(Function function, double functionTolerance, double[] initial, int maxIterations) {

    // check for stochastic derivatives
    if (!(function instanceof AbstractStochasticCachingDiffFunction)) {
      throw new UnsupportedOperationException();
    }
    AbstractStochasticCachingDiffFunction dfunction = (AbstractStochasticCachingDiffFunction) function;

    dfunction.method = StochasticCalculateMethods.GradientOnly;

    if(false){
      StochasticDiffFunctionTester sdft = new StochasticDiffFunctionTester(dfunction);
      ArrayMath.add(initial, gen.nextDouble() ); // to make sure that priors are working.
      sdft.testSumOfBatches(initial, 1e-4);
      System.exit(1);
    }
    
    x = initial;
    grad = new double[x.length];
    newX = new double[x.length];
    gradList = new ArrayList<double[]>();
    numBatches =  dfunction.dataDimension()/ bSize;
    outputFrequency = (int) Math.ceil( ((double) numBatches) /( (double) outputFrequency) )  ;

    init(dfunction);
    initFiles();
    
    boolean have_max = (maxIterations > 0 || numPasses > 0);

    if (!have_max){
      throw new UnsupportedOperationException("No maximum number of iterations has been secified.");
    }else{
      maxIterations = Math.max(maxIterations, numPasses)*numBatches;
    }

    sayln("       Batchsize of: " + bSize);
    sayln("       Data dimension of: " + dfunction.dataDimension() );
    sayln("       Batches per pass through data:  " + numBatches );
    sayln("       Max iterations is = " + maxIterations);

    if (outputIterationsToFile) {
      infoFile.println(function.domainDimension() + "; DomainDimension " );
      infoFile.println(bSize + "; batchSize ");
      infoFile.println(maxIterations + "; maxIterations");
      infoFile.println(numBatches + "; numBatches ");
      infoFile.println(outputFrequency  + "; outputFrequency");
    }

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //            Loop
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Timing total = new Timing();
    Timing current = new Timing();
    total.start();
    current.start();
    for (k = 0; k<maxIterations ; k++)  {
      try{
        
        say("Iter: " + k + " ");

        if(k > 0 && gradList.size() >= memory){
          newGrad = gradList.remove(0);
        }else{
          newGrad = new double[grad.length];
        }
        
        dfunction.hasNewVals = true;
        System.arraycopy(dfunction.derivativeAt(x,v,bSize),0,newGrad,0,newGrad.length);
        ensureFinite(newGrad,"newGrad");
        gradList.add(newGrad);
        grad = smooth(gradList);

        //Get the next X
        takeStep(dfunction);

        ensureFinite(newX,"newX");

        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // THIS IS FOR DEBUG ONLY
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if(outputIterationsToFile && (k%outputFrequency == 0) && k!=0 ) {
          double curVal = dfunction.valueAt(x);
          say(" TrueValue{ " + curVal + " } ");
          file.println(k + " , " + curVal + " , " + total.report() );
        }
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // END OF DEBUG STUFF
        //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        
        if (k >= maxIterations) {
          sayln("Stochastic Optimization complete.  Stopped after max iterations");
          x = newX;
          break;
        }

        if (total.report() >= maxTime){
          sayln("Stochastic Optimization complete.  Stopped after max time");
          x = newX;
          break;
        }

        System.arraycopy(newX, 0, x, 0, x.length);

        say("[" + ( total.report() )/1000.0 + " s " );
        say("{" + (current.restart()/1000.0) + " s}] ");
        say(" "+dfunction.lastValue());

        if (quiet) {
          System.err.print(".");
        }else{
          sayln("");
        }

      }catch(InvalidElementException e){
        System.err.println(e.toString());
        for(int i=0;i<x.length;i++){ x[i]=Double.NaN; }
        break;
      }

    }

    if(outputIterationsToFile){
      infoFile.println(k + "; Iterations");
      infoFile.println(( total.report() )/1000.0 + "; Completion Time");
      infoFile.println(dfunction.valueAt(x) + "; Finalvalue");
      
      infoFile.close();
      file.close();
      System.err.println("Output Files Closed");
      //System.exit(1);
    }

    say("Completed in: " + ( total.report() )/1000.0 + " s");
    
    return x;
  }


  public interface PropertySetter <T1> {
    public void set(T1 in);
  }
  
  protected void sayln(String s) {
    if (!quiet) {
      System.err.println(s);
    }
  }

  protected void say(String s) {
    if (!quiet) {
      System.err.print(s);
    }
  }



}
