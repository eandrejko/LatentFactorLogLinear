package org.apache.mahout.classifier;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

/**
 * Created by IntelliJ IDEA. User: tdunning Date: Oct 14, 2010 Time: 12:07:10 PM To change this
 * template use File | Settings | File Templates.
 */
public interface VectorClassifier {
  /**
   * Returns the number of categories for the target variable.  A vector classifier
   * will encode it's output using a zero-based 1 of numCategories encoding.
   * @return The number of categories.
   */
  int numCategories();

  /**
   * Classify a vector returning a vector of numCategories-1 scores.  It is assumed that
   * the score for the missing category is one minus the sum of the scores that are returned.
   *
   * Note that the missing score is the 0-th score.
   * @param instance  A feature vector to be classified.
   * @return  A vector of probabilities in 1 of n-1 encoding.
   */
  Vector classify(Vector instance);

  Vector classifyNoLink(Vector features);

  /**
   * Classifies a vector in the special case of a binary classifier where
   * <code>classify(Vector)</code> would return a vector with only one element.  As such,
   * using this method can void the allocation of a vector.
   * @param instance   The feature vector to be classified.
   * @return The score for category 1.
   *
   * @see #classify(org.apache.mahout.math.Vector)
   */
  double classifyScalar(Vector instance);

  Vector classifyFull(Vector instance);

  Vector classifyFull(Vector r, Vector instance);

  Matrix classify(Matrix data);

  Matrix classifyFull(Matrix data);

  Vector classifyScalar(Matrix data);
}
