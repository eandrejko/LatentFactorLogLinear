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

package org.apache.mahout.utils.parameters;

import org.apache.hadoop.mapred.JobConf;

import java.util.Collection;
import java.util.Collections;

public abstract class AbstractParameter<T> implements Parameter<T> {

  protected T value;
  protected final String prefix;
  protected final String name;
  protected final String description;
  protected final Class<T> type;
  protected final String defaultValue;


  public void configure(JobConf jobConf) {
    // nothing to do    
  }

  public void createParameters(String prefix, JobConf jobConf) {
  }

  public String getStringValue() {
    if (value == null) {
      return null;
    }
    return value.toString();
  }


  @SuppressWarnings("unchecked")
  public Collection<Parameter> getParameters() {
    return Collections.EMPTY_LIST;
  }

  protected AbstractParameter(Class<T> type, String prefix, String name, JobConf jobConf, T defaultValue, String description) {
    this.type = type;
    this.name = name;
    this.description = description;

    this.value = defaultValue;
    this.defaultValue = getStringValue();

    this.prefix = prefix;
    String jobConfValue = jobConf.get(prefix + name);
    if (jobConfValue != null) {
      setStringValue(jobConfValue);
    }

  }

  public String prefix() {
    return prefix;
  }

  public String name() {
    return name;
  }

  public String description() {
    return description;
  }

  public Class<T> type() {
    return type;
  }


  public String defaultValue() {
    return defaultValue;
  }

  public T get() {
    return value;
  }

  public void set(T value) {
    this.value = value;
  }


  public String toString() {
    if (value != null) {
      return value.toString();
    } else {
      return super.toString();
    }
  }

}