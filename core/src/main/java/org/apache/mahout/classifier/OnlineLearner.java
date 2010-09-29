package org.apache.mahout.classifier;

import org.apache.mahout.math.Vector;

/**
 * The simplest interface for online learning algorithms.
 */
public interface OnlineLearner {
  /**
   * Updates the model using a particular target variable value and a feature vector.
   * <p/>
   * There may an assumption that if multiple passes through the training data are necessary, then
   * the training examples will be presented in the same order.  This is because the order of
   * training examples may be used to assign records to different data splits for evaluation by
   * cross-validation.  Without the order invariance, records might be assigned to training and test
   * splits and error estimates could be seriously affected.
   * <p/>
   * If re-ordering is necessary, then using the alternative API which allows a tracking key to be
   * added to the training example can be used.
   *
   * @param actual   The value of the target variable.  This value should be in the half-open
   *                 interval [0..n) where n is the number of target categories.
   * @param instance The feature vector for this example.
   */
  void train(int actual, Vector instance);

  /**
   * Updates the model using a particular target variable value and a feature vector.
   * <p/>
   * There may an assumption that if multiple passes through the training data are necessary that
   * the tracking key for a record will be the same for each pass and that there will be a
   * relatively large number of distinct tracking keys and that the low-order bits of the tracking
   * keys will not correlate with any of the input variables.  This tracking key is used to assign
   * training examples to different test/training splits.
   * <p/>
   * Examples of useful tracking keys include id-numbers for the training records derived from
   * a database id for the base table from the which the record is derived, or the offset of
   * the original data record in a data file.
   *
   * @param trackingKey The tracking key for this training example.
   * @param groupKey     An optional value that allows examples to be grouped in the computation of
   * the update to the model.
   * @param actual   The value of the target variable.  This value should be in the half-open
   *                 interval [0..n) where n is the number of target categories.
   * @param instance The feature vector for this example.
   */
  void train(long trackingKey, String groupKey, int actual, Vector instance);

  /**
   * Updates the model using a particular target variable value and a feature vector.
   * <p/>
   * There may an assumption that if multiple passes through the training data are necessary that
   * the tracking key for a record will be the same for each pass and that there will be a
   * relatively large number of distinct tracking keys and that the low-order bits of the tracking
   * keys will not correlate with any of the input variables.  This tracking key is used to assign
   * training examples to different test/training splits.
   * <p/>
   * Examples of useful tracking keys include id-numbers for the training records derived from
   * a database id for the base table from the which the record is derived, or the offset of
   * the original data record in a data file.
   *
   * @param trackingKey The tracking key for this training example.
   * @param actual   The value of the target variable.  This value should be in the half-open
   *                 interval [0..n) where n is the number of target categories.
   * @param instance The feature vector for this example.
   */
  void train(long trackingKey, int actual, Vector instance);

  /**
   * Prepares the classifier for classification and deallocates any temporary data structures.
   *
   * An online classifier should be able to accept more training after being closed, but
   * closing the classifier may make classification more efficient.
   */
  void close();
}
