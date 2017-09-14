# Copyright 2017 Google Inc. All Rights Reserved. Licensed under the Apache
# License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.


import os
import shutil
import timeit
import types
from google.protobuf import text_format

import model
import task
import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.training.python.training import hparam

tf.logging.set_verbosity(tf.logging.ERROR)

TEST_CSV = os.path.join(os.getcwd(), 'wide-deep/test.csv')
MODEL_DIR = 'estimator_model'

class BaseTest(test.TestCase):
  def test_model_label_column(self):
    self.assertEquals(model.LABEL_COLUMN, 'income_bracket')

  def test_model_labels(self):
    self.assertEquals(model.LABELS[0], ' <=50K')
    self.assertEquals(model.LABELS[1], ' >50K')

  def test_generate_input_fn_type(self):
    input_fn = model.input_fn([TEST_CSV], batch_size=6)
    self.assertEquals(type(input_fn), types.TupleType)

  def test_learn_runner_model(self):
    try:
      shutil.rmtree(MODEL_DIR)
    except:
      pass

    args1 = {
            'min_eval_frequency': 1,
            'eval_delay_secs': 10,
            'train_steps': 1000,
            'eval_steps': 100
            }

    args2 = {
            'eval_batch_size': 40,
            'embedding_size': 8,
            'scale_factor': 0.7,
            'num_layers': 4,
            'eval_files': [TEST_CSV],
            'train_files': [TEST_CSV],
            'num_epochs': None,
            'first_layer_size': 100,
            'train_batch_size': 40
            }

    args2.update(args1)


    timer = timeit.timeit(
        lambda : learn_runner.run(task.generate_experiment_fn(**args1),
            run_config=run_config.RunConfig(model_dir=MODEL_DIR),
            hparams=hparam.HParams(**args2)),
        number=1)


    files = os.listdir(MODEL_DIR)

    self.assertEquals(len(files), 10)
    self.assertEquals('checkpoint' in files, True)
    self.assertEquals('eval' in files, True)
    self.assertGreater(
            os.stat(os.path.join(MODEL_DIR, 'graph.pbtxt')).st_size,
            200)

    gdef = tf.GraphDef()
    text_format.Merge(open(os.path.join(MODEL_DIR, 'graph.pbtxt')).read(), gdef)

    self.assertLess(timer, 50, msg='Model training too long {:.2f}s'.format(timer))

    node_input = [
        'ExpandDims',
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
        for inp in node.input:
          self.assertEquals(inp in node_input, True)

        for ntype in node.attr.get('OUT_TYPE').list.type:
          self.assertEquals(tf.DType(ntype) in node_types, True)

if __name__ == '__main__':
  test.main()
