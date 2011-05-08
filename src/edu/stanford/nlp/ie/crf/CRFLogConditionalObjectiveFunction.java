
package edu.stanford.nlp.ie.crf;

import edu.stanford.nlp.math.ArrayMath;
import edu.stanford.nlp.optimization.AbstractStochasticCachingDiffFunction;
import edu.stanford.nlp.util.Index;

import java.util.Arrays;


/**
 * @author Jenny Finkel
 */

public class CRFLogConditionalObjectiveFunction extends AbstractStochasticCachingDiffFunction {

  public static final int NO_PRIOR = 0;
  public static final int QUADRATIC_PRIOR = 1;
  /* Use a Huber robust regression penalty (L1 except very near 0) not L2 */
  public static final int HUBER_PRIOR = 2;
  public static final int QUARTIC_PRIOR = 3;

  protected int prior;
  protected double sigma;
  protected double epsilon;

  Index[] labelIndices;
  Index classIndex;
  Index featureIndex;
  double[][] Ehat; // empirical counts of all the features [feature][class]
  int window;
  int numClasses;
  int[] map;
  int[][][][] data;
  int[][] labels;
  int domainDimension = -1;

  String crfType = "maxent";
  String backgroundSymbol;

  public static boolean VERBOSE = false;

  CRFLogConditionalObjectiveFunction(int[][][][] data, int[][] labels, Index featureIndex, int window, Index classIndex, Index[] labelIndices, int[] map, String backgroundSymbol) {
    this(data, labels, featureIndex, window, classIndex, labelIndices, map, QUADRATIC_PRIOR, backgroundSymbol);
  }

  CRFLogConditionalObjectiveFunction(int[][][][] data, int[][] labels, Index featureIndex, int window, Index classIndex, Index[] labelIndices, int[] map, String backgroundSymbol, double sigma) {
    this(data, labels, featureIndex, window, classIndex, labelIndices, map, QUADRATIC_PRIOR, backgroundSymbol, sigma);
  }

  CRFLogConditionalObjectiveFunction(int[][][][] data, int[][] labels, Index featureIndex, int window, Index classIndex, Index[] labelIndices, int[] map, int prior, String backgroundSymbol) {
    this(data, labels, featureIndex, window, classIndex, labelIndices, map, prior, backgroundSymbol, 1.0);
  }

  CRFLogConditionalObjectiveFunction(int[][][][] data, int[][] labels, Index featureIndex, int window, Index classIndex, Index[] labelIndices, int[] map, int prior, String backgroundSymbol, double sigma) {
    this.featureIndex = featureIndex;
    this.window = window;
    this.classIndex = classIndex;
    this.numClasses = classIndex.size();
    this.labelIndices = labelIndices;
    this.map = map;
    this.data = data;
    this.labels = labels;
    this.prior = prior;
    this.backgroundSymbol = backgroundSymbol;
    this.sigma = sigma;
    empiricalCounts(data, labels);
  }

  @Override
  public int domainDimension() {
    if (domainDimension < 0) {
      domainDimension = 0;
      for (int i = 0; i < map.length; i++) {
        domainDimension += labelIndices[map[i]].size();
      }
    }
    return domainDimension;
  }

  /**
   * Takes a double array of weights which and creates a 2D array where:
   * 
   * the first element is the mapped index of featuresIndex
   * the second element is the index of the of the element
   *
   * @param weights
   * @return a 2D weight array
   */
  public double[][] to2D(double[] weights) {
    double[][] newWeights = new double[map.length][];
    int index = 0;
    for (int i = 0; i < map.length; i++) {
      newWeights[i] = new double[labelIndices[map[i]].size()];
      System.arraycopy(weights, index, newWeights[i], 0, labelIndices[map[i]].size());
      index += labelIndices[map[i]].size();
    }
    return newWeights;
  }

  public double[] to1D(double[][] weights) {
    double[] newWeights = new double[domainDimension()];
    int index = 0;
    for (int i = 0; i < weights.length; i++) {
      System.arraycopy(weights[i], 0, newWeights, index, weights[i].length);
      index += weights[i].length;
    }
    return newWeights;
  }

  public double[][] empty2D() {
    double[][] d = new double[map.length][];
    // int index = 0;
    for (int i = 0; i < map.length; i++) {
      d[i] = new double[labelIndices[map[i]].size()];
      // cdm july 2005: below array initialization isn't necessary: JLS (3rd ed.) 4.12.5 
      // Arrays.fill(d[i], 0.0);
      // index += labelIndices[map[i]].size();
    }
    return d;
  }

  private void empiricalCounts(int[][][][] data, int[][] labels) {
    Ehat = empty2D();

    for (int m = 0; m < data.length; m++) {
      int[][][] docData = data[m];
      int[] docLabels = labels[m];
      int[] windowLabels = new int[window];
      Arrays.fill(windowLabels, classIndex.indexOf(backgroundSymbol));
      if (docLabels.length>docData.length) { // only true for self-training
        // fill the windowLabel array with the extra docLabels
        System.arraycopy(docLabels, 0, windowLabels, 0, windowLabels.length);
        // shift the docLabels array left
        int[] newDocLabels = new int[docData.length];
        System.arraycopy(docLabels, docLabels.length-newDocLabels.length, newDocLabels, 0, newDocLabels.length);
        docLabels = newDocLabels;
      }
      for (int i = 0; i < docData.length; i++) {
        System.arraycopy(windowLabels, 1, windowLabels, 0, window - 1);
        windowLabels[window - 1] = docLabels[i];
        for (int j = 0; j < docData[i].length; j++) {
          int[] cliqueLabel = new int[j + 1];
          System.arraycopy(windowLabels, window - 1 - j, cliqueLabel, 0, j + 1);
          CRFLabel crfLabel = new CRFLabel(cliqueLabel);
          int labelIndex = labelIndices[j].indexOf(crfLabel);
          //System.err.println(crfLabel + " " + labelIndex);
          for (int k = 0; k < docData[i][j].length; k++) {
            Ehat[docData[i][j][k]][labelIndex]++;
          }
        }
      }
    }
  }

  /**
   * Calculates both value and partial derivatives at the point x, and save them internally.
   * @param x
   */
  @Override
  public void calculate(double[] x) {

    double prob = 0; // the log prob of the sequence given the model, which is the negation of value at this point
    double[][] weights = to2D(x);

    // the expectations over counts
    // first index is feature index, second index is of possible labeling
    double[][] E = empty2D();

    // iterate over all the documents
    for (int m = 0; m < data.length; m++) {
      int[][][] docData = data[m];
      int[] docLabels = labels[m];

      // make a clique tree for this document
      CRFCliqueTree cliqueTree = CRFCliqueTree.getCalibratedCliqueTree(weights, docData, labelIndices, numClasses, classIndex, backgroundSymbol);

      // compute the log probability of the document given the model with the parameters x
      int[] given = new int[window - 1];
      Arrays.fill(given, classIndex.indexOf(backgroundSymbol));
      if (docLabels.length>docData.length) { // only true for self-training
        // fill the given array with the extra docLabels
        System.arraycopy(docLabels, 0, given, 0, given.length);
        // shift the docLabels array left
        int[] newDocLabels = new int[docData.length];
        System.arraycopy(docLabels, docLabels.length-newDocLabels.length, newDocLabels, 0, newDocLabels.length);
        docLabels = newDocLabels;
      }
      // iterate over the positions in this document
      for (int i = 0; i < docData.length; i++) {
        int label = docLabels[i];
        double p = cliqueTree.condLogProbGivenPrevious(i, label, given);
        if (VERBOSE) {
          System.err.println("P(" + label + "|" + ArrayMath.toString(given) + ")=" + p);
        }
        prob += p;
        System.arraycopy(given, 1, given, 0, given.length - 1);
        given[given.length - 1] = label;
      }

      // compute the expected counts for this document, which we will need to compute the derivative
      // iterate over the positions in this document
      for (int i = 0; i < data[m].length; i++) {
        // for each possible clique at this position
        for (int j = 0; j < data[m][i].length; j++) {
          Index labelIndex = labelIndices[j];
          // for each possible labeling for that clique
          for (int k = 0; k < labelIndex.size(); k++) {
            int[] label = ((CRFLabel) labelIndex.get(k)).getLabel();
            double p = cliqueTree.prob(i, label); // probability of these labels occurring in this clique with these features
            for (int n = 0; n < data[m][i][j].length; n++) {
              E[data[m][i][j][n]][k] += p;
            }
          }
        }
      }
    }

    if (Double.isNaN(prob)) { // shouldn't be the case
      throw new RuntimeException("Got NaN for prob in CRFLogConditionalObjectiveFunction.calculate()");
    }

    value = -prob;

    // compute the partial derivative for each feature by comparing expected counts to empirical counts
    int index = 0;
    for (int i = 0; i < E.length; i++) {
      for (int j = 0; j < E[i].length; j++) {
        derivative[index++] = (E[i][j] - Ehat[i][j]);
        if (VERBOSE) {
          System.err.println("deriv(" + i + "," + j + ") = " + E[i][j] + " - " + Ehat[i][j] + " = " + derivative[index - 1]);
        }
      }
    }


    // incorporate priors
    if (prior == QUADRATIC_PRIOR) {
      double sigmaSq = sigma * sigma;
      for (int i = 0; i < x.length; i++) {
        double k = 1.0;
        double w = x[i];
        value += k * w * w / 2.0 / sigmaSq;
        derivative[i] += k * w / sigmaSq;
      }
    } else if (prior == HUBER_PRIOR) {
      double sigmaSq = sigma * sigma;
      for (int i = 0; i < x.length; i++) {
        double w = x[i];
        double wabs = Math.abs(w);
        if (wabs < epsilon) {
          value += w * w / 2.0 / epsilon / sigmaSq;
          derivative[i] += w / epsilon / sigmaSq;
        } else {
          value += (wabs - epsilon / 2) / sigmaSq;
          derivative[i] += ((w < 0.0) ? -1.0 : 1.0) / sigmaSq;
        }
      }
    } else if (prior == QUARTIC_PRIOR) {
      double sigmaQu = sigma * sigma * sigma * sigma;
      for (int i = 0; i < x.length; i++) {
        double k = 1.0;
        double w = x[i];
        value += k * w * w * w * w / 2.0 / sigmaQu;
        derivative[i] += k * w / sigmaQu;
      }
    }


  }

  @Override
  public void calculateStochastic(double[] x, double [] v, int[] batch){
    calculateStochasticGradientOnly(x,batch);
  }

  @Override
  public int dataDimension(){
    return data.length;
  }




  
  public void calculateStochasticGradientOnly(double[] x, int[] batch) {

    double prob = 0; // the log prob of the sequence given the model, which is the negation of value at this point
    double[][] weights = to2D(x);

    double batchScale = ((double) batch.length)/((double) this.dataDimension());
    
    // the expectations over counts
    // first index is feature index, second index is of possible labeling
    double[][] E = empty2D();

    // iterate over all the documents
    for (int m = 0; m < batch.length; m++) {
      int ind = batch[m];
      int[][][] docData = data[ind];
      int[] docLabels = labels[ind];

      // make a clique tree for this document
      CRFCliqueTree cliqueTree = CRFCliqueTree.getCalibratedCliqueTree(weights, docData, labelIndices, numClasses, classIndex, backgroundSymbol);

      // compute the log probability of the document given the model with the parameters x
      int[] given = new int[window - 1];
      Arrays.fill(given, classIndex.indexOf(backgroundSymbol));
      if (docLabels.length>docData.length) { // only true for self-training
        // fill the given array with the extra docLabels
        System.arraycopy(docLabels, 0, given, 0, given.length);
        // shift the docLabels array left
        int[] newDocLabels = new int[docData.length];
        System.arraycopy(docLabels, docLabels.length-newDocLabels.length, newDocLabels, 0, newDocLabels.length);
        docLabels = newDocLabels;
      }
      // iterate over the positions in this document
      for (int i = 0; i < docData.length; i++) {
        int label = docLabels[i];
        double p = cliqueTree.condLogProbGivenPrevious(i, label, given);
        if (VERBOSE) {
          System.err.println("P(" + label + "|" + ArrayMath.toString(given) + ")=" + p);
        }
        prob += p;
        System.arraycopy(given, 1, given, 0, given.length - 1);
        given[given.length - 1] = label;
      }

      // compute the expected counts for this document, which we will need to compute the derivative
      // iterate over the positions in this document
      for (int i = 0; i < data[ind].length; i++) {
        // for each possible clique at this position
        for (int j = 0; j < data[ind][i].length; j++) {
          Index labelIndex = labelIndices[j];
          // for each possible labeling for that clique
          for (int k = 0; k < labelIndex.size(); k++) {
            int[] label = ((CRFLabel) labelIndex.get(k)).getLabel();
            double p = cliqueTree.prob(i, label); // probability of these labels occurring in this clique with these features
            for (int n = 0; n < data[ind][i][j].length; n++) {
              E[data[ind][i][j][n]][k] += p;
            }
          }
        }
      }
    }

    if (Double.isNaN(prob)) { // shouldn't be the case
      throw new RuntimeException("Got NaN for prob in CRFLogConditionalObjectiveFunction.calculate()");
    }

    value = -prob;

    // compute the partial derivative for each feature by comparing expected counts to empirical counts
    int index = 0;
    for (int i = 0; i < E.length; i++) {
      for (int j = 0; j < E[i].length; j++) {
        derivative[index++] = (E[i][j] - batchScale*Ehat[i][j]);
        if (VERBOSE) {
          System.err.println("deriv(" + i + "," + j + ") = " + E[i][j] + " - " + Ehat[i][j] + " = " + derivative[index - 1]);
        }
      }
    }


    // incorporate priors
    if (prior == QUADRATIC_PRIOR) {
      double sigmaSq = sigma * sigma;
      for (int i = 0; i < x.length; i++) {
        double k = 1.0;
        double w = x[i];
        value += batchScale*k * w * w / 2.0 / sigmaSq;
        derivative[i] += batchScale*k * w / sigmaSq;
      }
    } else if (prior == HUBER_PRIOR) {
      double sigmaSq = sigma * sigma;
      for (int i = 0; i < x.length; i++) {
        double w = x[i];
        double wabs = Math.abs(w);
        if (wabs < epsilon) {
          value += batchScale*w * w / 2.0 / epsilon / sigmaSq;
          derivative[i] += batchScale*w / epsilon / sigmaSq;
        } else {
          value += batchScale*(wabs - epsilon / 2) / sigmaSq;
          derivative[i] += batchScale*((w < 0.0) ? -1.0 : 1.0) / sigmaSq;
        }
      }
    } else if (prior == QUARTIC_PRIOR) {
      double sigmaQu = sigma * sigma * sigma * sigma;
      for (int i = 0; i < x.length; i++) {
        double k = 1.0;
        double w = x[i];
        value += batchScale*k * w * w * w * w / 2.0 / sigmaQu;
        derivative[i] += batchScale*k * w / sigmaQu;
      }
    }


  }
//   public void calculateWeird1(double[] x) {

//     double[][] weights = to2D(x);
//     double[][] E = empty2D();

//     value = 0.0;
//     Arrays.fill(derivative, 0.0);
//     double[][] sums = new double[labelIndices.length][];
//     double[][] probs = new double[labelIndices.length][];
//     double[][] counts = new double[labelIndices.length][];

//     for (int i = 0; i < sums.length; i++) {
//       int size = labelIndices[i].size();
//       sums[i] = new double[size];
//       probs[i] = new double[size];
//       counts[i] = new double[size];
//       Arrays.fill(counts[i], 0.0);
//     }

//     for (int d = 0; d < data.length; d++) {
//       int[] llabels = labels[d];
//       for (int e = 0; e < data[d].length; e++) {
//         int[][] ddata = this.data[d][e];

//         for (int cl = 0; cl < ddata.length; cl++) {
//           int[] features = ddata[cl];
//           // activation
//           Arrays.fill(sums[cl], 0.0);
//           int numClasses = labelIndices[cl].size();
//           for (int c = 0; c < numClasses; c++) {
//             for (int f = 0; f < features.length; f++) {
//               sums[cl][c] += weights[features[f]][c];
//             }
//           }
//         }


//         for (int cl = 0; cl < ddata.length; cl++) {

//           int[] label = new int[cl + 1];
//           //Arrays.fill(label, classIndex.indexOf("O"));
//           Arrays.fill(label, classIndex.indexOf(backgroundSymbol));
//           int index1 = label.length - 1;
//           for (int pos = e; pos >= 0 && index1 >= 0; pos--) {
//             //System.err.println(index1+" "+pos);
//             label[index1--] = llabels[pos];
//           }
//           CRFLabel crfLabel = new CRFLabel(label);
//           int labelIndex = labelIndices[cl].indexOf(crfLabel);

//           double total = SloppyMath.logSum(sums[cl]);
//           //                     int[] features = ddata[cl];
//           int numClasses = labelIndices[cl].size();
//           for (int c = 0; c < numClasses; c++) {
//             probs[cl][c] = Math.exp(sums[cl][c] - total);
//           }
//           //                     for (int f=0; f<features.length; f++) {
//           //                         for (int c=0; c<numClasses; c++) {
//           //                             //probs[cl][c] = Math.exp(sums[cl][c]-total);
//           //                             derivative[index] += probs[cl][c];
//           //                             if (c == labelIndex) {
//           //                                 derivative[index]--;
//           //                             }
//           //                             index++;
//           //                         }                       
//           //                     }
                    

//           value -= sums[cl][labelIndex] - total;

//           //                     // observed
//           //                     for (int f=0; f<features.length; f++) {
//           //                         //int i = indexOf(features[f], labels[d]);
//           //                         derivative[index+labelIndex] -= 1.0;
//           //                     }

//         }
                
//         // go through each clique...
//         for (int j = 0; j < data[d][e].length; j++) {
//           Index labelIndex = labelIndices[j];
                    
//           // ...and each possible labeling for that clique
//           for (int k = 0; k < labelIndex.size(); k++) {
//             int[] label = ((CRFLabel) labelIndex.get(k)).getLabel();
                        
//             // double p = Math.pow(Math.E, factorTables[i].logProbEnd(label));
//             double p = probs[j][k];
//             for (int n = 0; n < data[d][e][j].length; n++) {
//               E[data[d][e][j][n]][k] += p;
//             }
//           }
//         }
//       }

//     }
   

//     // compute the partial derivative for each feature
//     int index = 0;
//     for (int i = 0; i < E.length; i++) {
//       for (int j = 0; j < E[i].length; j++) {
//         derivative[index++] = (E[i][j] - Ehat[i][j]);
//       }
//     }

//     // observed
//     //       int index = 0;
//     //       for (int i = 0; i < Ehat.length; i++) {
//     //           for (int j = 0; j < Ehat[i].length; j++) {
//     //               derivative[index++] -= Ehat[i][j];
//     //           }
//     //       }

//     // priors
//     if (prior == QUADRATIC_PRIOR) {
//       double sigmaSq = sigma * sigma;
//       for (int i = 0; i < x.length; i++) {
//         double k = 1.0;
//         double w = x[i];
//         value += k * w * w / 2.0 / sigmaSq;
//         derivative[i] += k * w / sigmaSq;
//       }
//     } else if (prior == HUBER_PRIOR) {
//       double sigmaSq = sigma * sigma;
//       for (int i = 0; i < x.length; i++) {
//         double w = x[i];
//         double wabs = Math.abs(w);
//         if (wabs < epsilon) {
//           value += w * w / 2.0 / epsilon / sigmaSq;
//           derivative[i] += w / epsilon / sigmaSq;
//         } else {
//           value += (wabs - epsilon / 2) / sigmaSq;
//           derivative[i] += ((w < 0.0) ? -1.0 : 1.0) / sigmaSq;
//         }
//       }
//     } else if (prior == QUARTIC_PRIOR) {
//       double sigmaQu = sigma * sigma * sigma * sigma;
//       for (int i = 0; i < x.length; i++) {
//         double k = 1.0;
//         double w = x[i];
//         value += k * w * w * w * w / 2.0 / sigmaQu;
//         derivative[i] += k * w / sigmaQu;
//       }
//     }
//   }

//   public void calculateWeird(double[] x) {

//     double[][] weights = to2D(x);
//     double[][] E = empty2D();

//     value = 0.0;
//     Arrays.fill(derivative, 0.0);

//     int size = labelIndices[labelIndices.length - 1].size();

//     double[] sums = new double[size];
//     double[] probs = new double[size];

//     Index labelIndex = labelIndices[labelIndices.length - 1];

//     for (int d = 0; d < data.length; d++) {
//       int[] llabels = labels[d];

//       int[] label = new int[window];
//       //Arrays.fill(label, classIndex.indexOf("O"));
//       Arrays.fill(label, classIndex.indexOf(backgroundSymbol));

//       for (int e = 0; e < data[d].length; e++) {

//         Arrays.fill(sums, 0.0);

//         System.arraycopy(label, 1, label, 0, window - 1);
//         label[window - 1] = llabels[e];
//         CRFLabel crfLabel = new CRFLabel(label);
//         int maxCliqueLabelIndex = labelIndex.indexOf(crfLabel);

//         int[][] ddata = this.data[d][e];

//         //Iterator labelIter = labelIndices[labelIndices.length-1].iterator();
//         //while (labelIter.hasNext()) {

//         for (int i = 0; i < labelIndex.size(); i++) {
//           CRFLabel c = (CRFLabel) labelIndex.get(i);

//           for (int cl = 0; cl < ddata.length; cl++) {

//             CRFLabel cliqueLabel = c.getSmallerLabel(cl + 1);
//             int clIndex = labelIndices[cl].indexOf(cliqueLabel);

//             int[] features = ddata[cl];
//             for (int f = 0; f < features.length; f++) {
//               sums[i] += weights[features[f]][clIndex];
//             }
//           }
//         }

//         double total = SloppyMath.logSum(sums);
//         for (int i = 0; i < probs.length; i++) {
//           probs[i] = Math.exp(sums[i] - total);
//         }
//         value -= sums[maxCliqueLabelIndex] - total;

//         for (int i = 0; i < labelIndex.size(); i++) {
//           CRFLabel c = (CRFLabel) labelIndex.get(i);

//           for (int cl = 0; cl < ddata.length; cl++) {

//             CRFLabel cliqueLabel = c.getSmallerLabel(cl + 1);
//             int clIndex = labelIndices[cl].indexOf(cliqueLabel);
//             int[] features = ddata[cl];

//             for (int f = 0; f < features.length; f++) {
//               E[features[f]][clIndex] += probs[i];
//               if (i == maxCliqueLabelIndex) {
//                 E[features[f]][clIndex] -= 1.0;
//               }
//               //sums[i] += weights[features[f]][cl];
//             }
//           }
//         }
//       }
//     }
  

//     // compute the partial derivative for each feature
//     int index = 0;
//     for (int i = 0; i < E.length; i++) {
//       for (int j = 0; j < E[i].length; j++) {
//         //derivative[index++] = (E[i][j] - Ehat[i][j]);
//         derivative[index++] = E[i][j];
//       }
//     }

//     // priors
//     if (prior == QUADRATIC_PRIOR) {
//       double sigmaSq = sigma * sigma;
//       for (int i = 0; i < x.length; i++) {
//         double k = 1.0;
//         double w = x[i];
//         value += k * w * w / 2.0 / sigmaSq;
//         derivative[i] += k * w / sigmaSq;
//       }
//     } else if (prior == HUBER_PRIOR) {
//       double sigmaSq = sigma * sigma;
//       for (int i = 0; i < x.length; i++) {
//         double w = x[i];
//         double wabs = Math.abs(w);
//         if (wabs < epsilon) {
//           value += w * w / 2.0 / epsilon / sigmaSq;
//           derivative[i] += w / epsilon / sigmaSq;
//         } else {
//           value += (wabs - epsilon / 2) / sigmaSq;
//           derivative[i] += ((w < 0.0) ? -1.0 : 1.0) / sigmaSq;
//         }
//       }
//     } else if (prior == QUARTIC_PRIOR) {
//       double sigmaQu = sigma * sigma * sigma * sigma;
//       for (int i = 0; i < x.length; i++) {
//         double k = 1.0;
//         double w = x[i];
//         value += k * w * w * w * w / 2.0 / sigmaQu;
//         derivative[i] += k * w / sigmaQu;
//       }
//     }
//   }
}
