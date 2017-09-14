# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


import os
from sets import Set
import shutil
import timeit

import model
import task

from google.protobuf import text_format
import tensorflow as tf
from tensorflow.python.platform import test

tf.logging.set_verbosity(tf.logging.ERROR)

TEST_CSV = os.path.join(os.getcwd(), 'wide-deep/test.csv')
MODEL_DIR = 'tflowlevel_model'

class BaseTest(test.TestCase):
  def test_csv_serving_input_fn_keys(self):
    serving_input = model.csv_serving_input_fn(default_batch_size=10)
    keys = serving_input[0].keys()

    target_keys = ['hours_per_week', 'workclass', 'relationship', 'gender',
                   'age', 'marital_status', 'race', 'capital_gain',
                   'native_country', 'capital_loss', 'education',
                   'education_num', 'occupation']

    self.assertEquals(len(target_keys), len(keys))
    self.assertEquals(len(Set(keys) - Set(target_keys)), 0)

    # Check the shape of the input tensors
    for key in target_keys:
      shape = serving_input[0].get(key).shape
      self.assertEquals(shape, [10])

  def test_parse_csv(self):
    input_csv = ','.join(str(x) for x in range(15))
    input_csv_tensor = tf.constant([input_csv])
    input_dict = model.parse_csv(input_csv_tensor)
    self.assertEquals(len(input_dict.keys()), 14)

    keys_target = [('hours_per_week', 12), ('workclass', '1'), ('relationship', '7'),
                   ('gender', '9'), ('age', 0), ('marital_status', '5'),
                   ('race', '8'), ('capital_gain', 10), ('income_bracket', '14'),
                   ('native_country', '13'), ('capital_loss', 11),
                   ('education', '3'), ('education_num', 4), ('occupation', '6')]

    sess = tf.InteractiveSession()

    for key,target in keys_target:
      self.assertEquals(sess.run(input_dict[key])[0], target)

  def test_input_fn(self):
    features, labels = model.input_fn([TEST_CSV])

    self.assertEquals(type(labels), tf.Tensor)
    self.assertEquals(len(features.keys()), 13)

  def test_input_fn_queues(self):
    features, labels = model.input_fn([TEST_CSV], batch_size=6)

    with tf.Session() as session:
      coord = tf.train.Coordinator(clean_stop_exception_types=(
          tf.errors.CancelledError, tf.errors.OutOfRangeError))

      # Important to start all queue runners so that data is available
      # for reading
      tf.train.start_queue_runners(coord=coord, sess=session)

      workclass = session.run(features.get('workclass'))
      native_country = session.run(features.get('native_country'))

      self.assertEquals(len(workclass), 6)
      self.assertEquals(len(native_country), 6)

      self.assertEquals(' Private' in workclass, True)
      self.assertEquals(' United-States' in native_country, True)

  def test_model_fn_metrics(self):
    features, labels = model.input_fn([TEST_CSV])

    metric_dict = model.model_fn(model.EVAL, features, labels)
    self.assertEquals('auroc' in metric_dict, True)
    self.assertEquals('accuracy' in metric_dict, True)

  def test_model_fn_train(self):
    features, labels = model.input_fn([TEST_CSV])

    train_op, global_step = model.model_fn(model.TRAIN, features, labels)

    self.assertEquals(type(train_op), tf.Operation)
    self.assertEquals(type(global_step), tf.Variable)

  def test_model_dispatch(self):
    try:
      shutil.rmtree(MODEL_DIR)
    except:
      pass

    arguments = {
            'eval_steps': 100,
            'eval_batch_size': 40,
            'scale_factor': 0.7,
            'num_layers': 4,
            'eval_files': [TEST_CSV],
            'train_files': [TEST_CSV],
            'num_epochs': None,
            'first_layer_size': 100,
            'train_batch_size': 40,
            'train_steps': 1000,
            'job_dir': MODEL_DIR,
            'eval_frequency': 50,
            'learning_rate': 0.003,
            'export_format': 'EXAMPLE'
            }
    timer = timeit.timeit(lambda: task.dispatch(**arguments), number=1)

    files = os.listdir(MODEL_DIR)

    self.assertEquals(len(files), 11)
    self.assertEquals('checkpoint' in files, True)
    self.assertEquals('eval' in files, True)
    self.assertGreater(
            os.stat(os.path.join(MODEL_DIR, 'graph.pbtxt')).st_size,
            200)

    gdef = tf.GraphDef()
    text_format.Merge(open(os.path.join(MODEL_DIR, 'graph.pbtxt')).read(), gdef)
    self.assertLess(timer, 50, msg='Model training too long {:.2f}s'.format(timer))

    node_input = [
        'DecodeCSV/record_defaults_0',
        'DecodeCSV/record_defaults_1',
        'DecodeCSV/record_defaults_2',
        'DecodeCSV/record_defaults_3',
        'DecodeCSV/record_defaults_4',
        'DecodeCSV/record_defaults_5',
        'DecodeCSV/record_defaults_6',
        'DecodeCSV/record_defaults_7',
        'DecodeCSV/record_defaults_8',
        'DecodeCSV/record_defaults_9',
        'DecodeCSV/record_defaults_10',
        'DecodeCSV/record_defaults_11',
        'DecodeCSV/record_defaults_12',
        'DecodeCSV/record_defaults_13',
        'DecodeCSV/record_defaults_14']

    node_types = [
        tf.int32, tf.string, tf.int32, tf.string,
        tf.int32, tf.string, tf.string, tf.string,
        tf.string, tf.string, tf.int32, tf.int32,
        tf.int32, tf.string, tf.string]

    for node in gdef.node:
      name = node.name
      if name == 'DecodeCSV':
        for inp in node_input:
          self.assertEquals(inp in node.input, True)

        for ntype in node.attr.get('OUT_TYPE').list.type:
          self.assertEquals(tf.DType(ntype) in node_types, True)
if __name__ == '__main__':
  test.main()
