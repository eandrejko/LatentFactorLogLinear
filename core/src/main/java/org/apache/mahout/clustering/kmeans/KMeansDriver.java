/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.clustering.kmeans;

import java.io.IOException;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.mahout.clustering.WeightedVectorWritable;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class KMeansDriver extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(KMeansDriver.class);

  public KMeansDriver() {
  }

  public static void main(String[] args) throws Exception {
    new KMeansDriver().run(args);
  }

  /**
   * Run the job using supplied arguments on a new driver instance (convenience)
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param numReduceTasks
   *          the number of reducers
   * @param runClustering 
   *          true if points are to be clustered after iterations are completed
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public static void runJob(Path input,
                            Path clustersIn,
                            Path output,
                            String measureClass,
                            double convergenceDelta,
                            int maxIterations,
                            int numReduceTasks,
                            boolean runClustering) throws IOException, InterruptedException, ClassNotFoundException {
    new KMeansDriver().job(input, clustersIn, output, measureClass, convergenceDelta, maxIterations, numReduceTasks, runClustering);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption(DefaultOptionCreator.distanceMeasureOption().create());
    addOption(DefaultOptionCreator.clustersInOption()
        .withDescription("The input centroids, as Vectors.  Must be a SequenceFile of Writable, Cluster/Canopy.  "
            + "If k is also specified, then a random set of vectors will be selected" + " and written out to this path first")
        .create());
    addOption(DefaultOptionCreator.numClustersOption()
        .withDescription("The k in k-Means.  If specified, then a random selection of k Vectors will be chosen"
            + " as the Centroid and written to the clusters input path.").create());
    addOption(DefaultOptionCreator.convergenceOption().create());
    addOption(DefaultOptionCreator.maxIterationsOption().create());
    addOption(DefaultOptionCreator.overwriteOption().create());
    addOption(DefaultOptionCreator.numReducersOption().create());
    addOption(DefaultOptionCreator.clusteringOption().create());

    if (parseArguments(args) == null) {
      return -1;
    }

    Path input = getInputPath();
    Path clusters = new Path(getOption(DefaultOptionCreator.CLUSTERS_IN_OPTION));
    Path output = getOutputPath();
    String measureClass = getOption(DefaultOptionCreator.DISTANCE_MEASURE_OPTION);
    if (measureClass == null) {
      measureClass = SquaredEuclideanDistanceMeasure.class.getName();
    }
    double convergenceDelta = Double.parseDouble(getOption(DefaultOptionCreator.CONVERGENCE_DELTA_OPTION));
    int numReduceTasks = Integer.parseInt(getOption(DefaultOptionCreator.MAX_REDUCERS_OPTION));
    int maxIterations = Integer.parseInt(getOption(DefaultOptionCreator.MAX_ITERATIONS_OPTION));
    if (hasOption(DefaultOptionCreator.OVERWRITE_OPTION)) {
      HadoopUtil.overwriteOutput(output);
    }
    if (hasOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)) {
      clusters = RandomSeedGenerator.buildRandom(input, clusters, Integer
          .parseInt(getOption(DefaultOptionCreator.NUM_CLUSTERS_OPTION)));
    }
    boolean runClustering = hasOption(DefaultOptionCreator.CLUSTERING_OPTION);
    job(input, clusters, output, measureClass, convergenceDelta, maxIterations, numReduceTasks, runClustering);
    return 0;
  }

  /**
   * Iterate over the input vectors to produce clusters and, if requested, use the
   * results of the final iteration to cluster the input vectors.
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param numReduceTasks
   *          the number of reducers
   * @param runClustering 
   *          true if points are to be clustered after iterations are completed
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public void job(Path input,
                   Path clustersIn,
                   Path output,
                   String measureClass,
                   double convergenceDelta,
                   int maxIterations,
                   int numReduceTasks,
                   boolean runClustering) throws IOException, InterruptedException, ClassNotFoundException {
    // iterate until the clusters converge
    String delta = Double.toString(convergenceDelta);
    if (log.isInfoEnabled()) {
      log.info("Input: {} Clusters In: {} Out: {} Distance: {}", new Object[] { input, clustersIn, output, measureClass });
      log.info("convergence: {} max Iterations: {} num Reduce Tasks: {} Input Vectors: {}", new Object[] { convergenceDelta,
          maxIterations, numReduceTasks, VectorWritable.class.getName() });
    }
    Path clustersOut = buildClusters(input, clustersIn, output, measureClass, maxIterations, numReduceTasks, delta);
    if (runClustering) {
      log.info("Clustering data");
      clusterData(input, clustersOut, new Path(output, Cluster.CLUSTERED_POINTS_DIR), measureClass, delta);
    }
  }

  /**
   * Iterate over the input vectors to produce cluster directories for each iteration
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for initial & computed clusters
   * @param output
   *          the directory pathname for output points
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param maxIterations
   *          the maximum number of iterations
   * @param numReduceTasks
   *          the number of reducers
   * @return the Path of the final clusters directory
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
  public Path buildClusters(Path input,
                             Path clustersIn,
                             Path output,
                             String measureClass,
                             int maxIterations,
                             int numReduceTasks,
                             String delta) throws IOException, InterruptedException, ClassNotFoundException {
    boolean converged = false;
    int iteration = 1;
    while (!converged && (iteration <= maxIterations)) {
      log.info("Iteration {}", iteration);
      // point the output to a new directory per iteration
      Path clustersOut = new Path(output, Cluster.CLUSTERS_DIR + iteration);
      converged = runIteration(input, clustersIn, clustersOut, measureClass, delta, numReduceTasks);
      // now point the input to the old output directory
      clustersIn = clustersOut;
      iteration++;
    }
    return clustersIn;
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for input clusters
   * @param clustersOut
   *          the directory pathname for output clusters
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @param numReduceTasks
   *          the number of reducer tasks
   * @return true if the iteration successfully runs
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  private boolean runIteration(Path input,
                               Path clustersIn,
                               Path clustersOut,
                               String measureClass,
                               String convergenceDelta,
                               int numReduceTasks) throws IOException, InterruptedException, ClassNotFoundException {
    Configuration conf = new Configuration();
    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    Job job = new Job(conf);

    job.setMapOutputKeyClass(Text.class);
    job.setMapOutputValueClass(KMeansInfo.class);
    job.setOutputKeyClass(Text.class);
    job.setOutputValueClass(Cluster.class);

    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setMapperClass(KMeansMapper.class);
    job.setCombinerClass(KMeansCombiner.class);
    job.setReducerClass(KMeansReducer.class);
    job.setNumReduceTasks(numReduceTasks);

    FileInputFormat.addInputPath(job, input);
    FileOutputFormat.setOutputPath(job, clustersOut);

    job.setJarByClass(KMeansDriver.class);
    HadoopUtil.overwriteOutput(clustersOut);
    job.waitForCompletion(true);
    FileSystem fs = FileSystem.get(clustersOut.toUri(), conf);

    return isConverged(clustersOut, conf, fs);
  }

  /**
   * Return if all of the Clusters in the parts in the filePath have converged or not
   * 
   * @param filePath
   *          the file path to the single file containing the clusters
   * @param conf
   *          the JobConf
   * @param fs
   *          the FileSystem
   * @return true if all Clusters are converged
   * @throws IOException
   *           if there was an IO error
   */
  private boolean isConverged(Path filePath, Configuration conf, FileSystem fs) throws IOException {
    FileStatus[] parts = fs.listStatus(filePath);
    for (FileStatus part : parts) {
      String name = part.getPath().getName();
      if (name.startsWith("part") && !name.endsWith(".crc")) {
        SequenceFile.Reader reader = new SequenceFile.Reader(fs, part.getPath(), conf);
        try {
          Writable key = (Writable) reader.getKeyClass().newInstance();
          Cluster value = new Cluster();
          while (reader.next(key, value)) {
            if (!value.isConverged()) {
              return false;
            }
          }
        } catch (InstantiationException e) { // shouldn't happen
          log.error("Exception", e);
          throw new IllegalStateException(e);
        } catch (IllegalAccessException e) {
          log.error("Exception", e);
          throw new IllegalStateException(e);
        } finally {
          reader.close();
        }
      }
    }
    return true;
  }

  /**
   * Run the job using supplied arguments
   * 
   * @param input
   *          the directory pathname for input points
   * @param clustersIn
   *          the directory pathname for input clusters
   * @param output
   *          the directory pathname for output points
   * @param measureClass
   *          the classname of the DistanceMeasure
   * @param convergenceDelta
   *          the convergence delta value
   * @throws ClassNotFoundException 
   * @throws InterruptedException 
   */
  public void clusterData(Path input, Path clustersIn, Path output, String measureClass, String convergenceDelta)
      throws IOException, InterruptedException, ClassNotFoundException {
    if (log.isInfoEnabled()) {
      log.info("Running Clustering");
      log.info("Input: {} Clusters In: {} Out: {} Distance: {}", new Object[] { input, clustersIn, output, measureClass });
      log.info("convergence: {} Input Vectors: {}", convergenceDelta, VectorWritable.class.getName());
    }
    Configuration conf = new Configuration();
    conf.set(KMeansConfigKeys.CLUSTER_PATH_KEY, clustersIn.toString());
    conf.set(KMeansConfigKeys.DISTANCE_MEASURE_KEY, measureClass);
    conf.set(KMeansConfigKeys.CLUSTER_CONVERGENCE_KEY, convergenceDelta);

    Job job = new Job(conf);
    job.setInputFormatClass(SequenceFileInputFormat.class);
    job.setOutputFormatClass(SequenceFileOutputFormat.class);
    job.setOutputKeyClass(IntWritable.class);
    job.setOutputValueClass(WeightedVectorWritable.class);

    FileInputFormat.setInputPaths(job, input);
    HadoopUtil.overwriteOutput(output);
    FileOutputFormat.setOutputPath(job, output);

    job.setMapperClass(KMeansClusterMapper.class);
    job.setNumReduceTasks(0);
    job.setJarByClass(KMeansDriver.class);

    job.waitForCompletion(true);
  }
}
