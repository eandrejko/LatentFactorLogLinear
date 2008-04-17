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
package org.apache.mahout.clustering.meanshift;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapred.JobConf;
import org.apache.hadoop.mapred.MapReduceBase;
import org.apache.hadoop.mapred.OutputCollector;
import org.apache.hadoop.mapred.Reducer;
import org.apache.hadoop.mapred.Reporter;
import org.apache.mahout.matrix.CardinalityException;

public class MeanShiftCanopyReducer extends MapReduceBase implements
    Reducer<Text, WritableComparable, Text, WritableComparable> {

  List<MeanShiftCanopy> canopies = new ArrayList<MeanShiftCanopy>();

  OutputCollector<Text, WritableComparable> collector;

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.hadoop.mapred.Reducer#reduce(org.apache.hadoop.io.WritableComparable,
   *      java.util.Iterator, org.apache.hadoop.mapred.OutputCollector,
   *      org.apache.hadoop.mapred.Reporter)
   */
  public void reduce(Text key, Iterator<WritableComparable> values,
      OutputCollector<Text, WritableComparable> output, Reporter reporter)
      throws IOException {
    collector = output;
    try {
      while (values.hasNext()) {
        Text value = (Text) values.next();
        MeanShiftCanopy canopy = MeanShiftCanopy.decodeCanopy(value.toString());
        MeanShiftCanopy.mergeCanopy(canopy, canopies);
      }

      for (MeanShiftCanopy canopy : canopies) {
        canopy.shiftToMean();
        collector.collect(new Text(canopy.getIdentifier()), new Text(
            MeanShiftCanopy.formatCanopy(canopy)));
      }
    } catch (CardinalityException e) {
      throw new RuntimeException(e);
    }
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.hadoop.mapred.MapReduceBase#configure(org.apache.hadoop.mapred.JobConf)
   */
  @Override
  public void configure(JobConf job) {
    super.configure(job);
    MeanShiftCanopy.configure(job);
  }

}