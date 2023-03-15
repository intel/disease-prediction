### Code changes to ClinicalBert
Minor changes to auto detect current directory and create to create output directory.

### Code changes to Bert
Code changes to modeling.py and optimization.py to adapt to Tensorflow 2.x

#### Changes to modeling.py
```
diff --git a/modeling.py b/modeling.py
index fed5259..da0bd2b 100644
--- a/modeling.py
+++ b/modeling.py
@@ -90,7 +90,7 @@ class BertConfig(object):
   @classmethod
   def from_json_file(cls, json_file):
     """Constructs a `BertConfig` from a json file of parameters."""
-    with tf.gfile.GFile(json_file, "r") as reader:
+    with tf.io.gfile.GFile(json_file, "r") as reader:
       text = reader.read()
     return cls.from_dict(json.loads(text))
 
@@ -168,8 +168,8 @@ class BertModel(object):
     if token_type_ids is None:
       token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)
 
-    with tf.variable_scope(scope, default_name="bert"):
-      with tf.variable_scope("embeddings"):
+    with tf.compat.v1.variable_scope(scope, default_name="bert"):
+      with tf.compat.v1.variable_scope("embeddings"):
         # Perform embedding lookup on the word ids.
         (self.embedding_output, self.embedding_table) = embedding_lookup(
             input_ids=input_ids,
@@ -193,7 +193,7 @@ class BertModel(object):
             max_position_embeddings=config.max_position_embeddings,
             dropout_prob=config.hidden_dropout_prob)
 
-      with tf.variable_scope("encoder"):
+      with tf.compat.v1.variable_scope("encoder"):
         # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
         # mask of shape [batch_size, seq_length, seq_length] which is used
         # for the attention scores.
@@ -221,11 +221,11 @@ class BertModel(object):
       # [batch_size, hidden_size]. This is necessary for segment-level
       # (or segment-pair-level) classification tasks where we need a fixed
       # dimensional representation of the segment.
-      with tf.variable_scope("pooler"):
+      with tf.compat.v1.variable_scope("pooler"):
         # We "pool" the model by simply taking the hidden state corresponding
         # to the first token. We assume that this has been pre-trained
         first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
-        self.pooled_output = tf.layers.dense(
+        self.pooled_output = tf.compat.v1.layers.dense(
             first_token_tensor,
             config.hidden_size,
             activation=tf.tanh,
@@ -359,11 +359,15 @@ def dropout(input_tensor, dropout_prob):
   return output
 
 
+#def layer_norm(input_tensor, name=None):
+#  """Run layer normalization on the last dimension of the tensor."""
+#  return tf.contrib.layers.layer_norm(
+#      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
+
 def layer_norm(input_tensor, name=None):
   """Run layer normalization on the last dimension of the tensor."""
-  return tf.contrib.layers.layer_norm(
-      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)
-
+  layer_norma = tf.keras.layers.LayerNormalization(axis = -1)
+  return layer_norma(input_tensor) 
 
 def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
   """Runs layer normalization followed by dropout."""
@@ -374,7 +378,7 @@ def layer_norm_and_dropout(input_tensor, dropout_prob, name=None):
 
 def create_initializer(initializer_range=0.02):
   """Creates a `truncated_normal_initializer` with the given range."""
-  return tf.truncated_normal_initializer(stddev=initializer_range)
+  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)
 
 
 def embedding_lookup(input_ids,
@@ -406,7 +410,7 @@ def embedding_lookup(input_ids,
   if input_ids.shape.ndims == 2:
     input_ids = tf.expand_dims(input_ids, axis=[-1])
 
-  embedding_table = tf.get_variable(
+  embedding_table = tf.compat.v1.get_variable(
       name=word_embedding_name,
       shape=[vocab_size, embedding_size],
       initializer=create_initializer(initializer_range))
@@ -473,7 +477,7 @@ def embedding_postprocessor(input_tensor,
     if token_type_ids is None:
       raise ValueError("`token_type_ids` must be specified if"
                        "`use_token_type` is True.")
-    token_type_table = tf.get_variable(
+    token_type_table = tf.compat.v1.get_variable(
         name=token_type_embedding_name,
         shape=[token_type_vocab_size, width],
         initializer=create_initializer(initializer_range))
@@ -487,9 +491,9 @@ def embedding_postprocessor(input_tensor,
     output += token_type_embeddings
 
   if use_position_embeddings:
-    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
+    assert_op = tf.compat.v1.assert_less_equal(seq_length, max_position_embeddings)
     with tf.control_dependencies([assert_op]):
-      full_position_embeddings = tf.get_variable(
+      full_position_embeddings = tf.compat.v1.get_variable(
           name=position_embedding_name,
           shape=[max_position_embeddings, width],
           initializer=create_initializer(initializer_range))
@@ -663,7 +667,7 @@ def attention_layer(from_tensor,
   to_tensor_2d = reshape_to_matrix(to_tensor)
 
   # `query_layer` = [B*F, N*H]
-  query_layer = tf.layers.dense(
+  query_layer = tf.compat.v1.layers.dense(
       from_tensor_2d,
       num_attention_heads * size_per_head,
       activation=query_act,
@@ -671,7 +675,7 @@ def attention_layer(from_tensor,
       kernel_initializer=create_initializer(initializer_range))
 
   # `key_layer` = [B*T, N*H]
-  key_layer = tf.layers.dense(
+  key_layer = tf.compat.v1.layers.dense(
       to_tensor_2d,
       num_attention_heads * size_per_head,
       activation=key_act,
@@ -679,7 +683,7 @@ def attention_layer(from_tensor,
       kernel_initializer=create_initializer(initializer_range))
 
   # `value_layer` = [B*T, N*H]
-  value_layer = tf.layers.dense(
+  value_layer = tf.compat.v1.layers.dense(
       to_tensor_2d,
       num_attention_heads * size_per_head,
       activation=value_act,
@@ -824,12 +828,12 @@ def transformer_model(input_tensor,
 
   all_layer_outputs = []
   for layer_idx in range(num_hidden_layers):
-    with tf.variable_scope("layer_%d" % layer_idx):
+    with tf.compat.v1.variable_scope("layer_%d" % layer_idx):
       layer_input = prev_output
 
-      with tf.variable_scope("attention"):
+      with tf.compat.v1.variable_scope("attention"):
         attention_heads = []
-        with tf.variable_scope("self"):
+        with tf.compat.v1.variable_scope("self"):
           attention_head = attention_layer(
               from_tensor=layer_input,
               to_tensor=layer_input,
@@ -854,8 +858,8 @@ def transformer_model(input_tensor,
 
         # Run a linear projection of `hidden_size` then add a residual
         # with `layer_input`.
-        with tf.variable_scope("output"):
-          attention_output = tf.layers.dense(
+        with tf.compat.v1.variable_scope("output"):
+          attention_output = tf.compat.v1.layers.dense(
               attention_output,
               hidden_size,
               kernel_initializer=create_initializer(initializer_range))
@@ -863,16 +867,16 @@ def transformer_model(input_tensor,
           attention_output = layer_norm(attention_output + layer_input)
 
       # The activation is only applied to the "intermediate" hidden layer.
-      with tf.variable_scope("intermediate"):
-        intermediate_output = tf.layers.dense(
+      with tf.compat.v1.variable_scope("intermediate"):
+        intermediate_output = tf.compat.v1.layers.dense(
             attention_output,
             intermediate_size,
             activation=intermediate_act_fn,
             kernel_initializer=create_initializer(initializer_range))
 
       # Down-project back to `hidden_size` then add the residual.
-      with tf.variable_scope("output"):
-        layer_output = tf.layers.dense(
+      with tf.compat.v1.variable_scope("output"):
+        layer_output = tf.compat.v1.layers.dense(
             intermediate_output,
             hidden_size,
             kernel_initializer=create_initializer(initializer_range))
```

#### Code changes to optimization.py
```
diff --git a/optimization.py b/optimization.py
index d33dabd..ee3ae6a 100644
--- a/optimization.py
+++ b/optimization.py
@@ -21,15 +21,13 @@ from __future__ import print_function
 import re
 import tensorflow as tf
 
-
 def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
   """Creates an optimizer training op."""
-  global_step = tf.train.get_or_create_global_step()
+  global_step = tf.compat.v1.train.get_or_create_global_step()
 
   learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
-
   # Implements linear decay of the learning rate.
-  learning_rate = tf.train.polynomial_decay(
+  learning_rate = tf.compat.v1.train.polynomial_decay(
       learning_rate,
       global_step,
       num_train_steps,
@@ -52,22 +50,23 @@ def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
     is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
     learning_rate = (
         (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
-
+    
   # It is recommended that you use this optimizer for fine tuning, since this
   # is how the model was trained (note that the Adam m/v variables are NOT
   # loaded from init_checkpoint.)
   optimizer = AdamWeightDecayOptimizer(
       learning_rate=learning_rate,
+      #learning_rate=.00005,
       weight_decay_rate=0.01,
       beta_1=0.9,
       beta_2=0.999,
       epsilon=1e-6,
       exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
-
+  
   if use_tpu:
     optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
 
-  tvars = tf.trainable_variables()
+  tvars = tf.compat.v1.trainable_variables()
   grads = tf.gradients(loss, tvars)
 
   # This is how the model was pre-trained.
@@ -84,7 +83,9 @@ def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
   return train_op
 
 
-class AdamWeightDecayOptimizer(tf.train.Optimizer):
+#class AdamWeightDecayOptimizer(tf.train.Optimizer):
+class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
+#class AdamWeightDecayOptimizer(tf.keras.optimizers.Optimizer):
   """A basic Adam optimizer that includes "correct" L2 weight decay."""
 
   def __init__(self,
@@ -98,6 +99,7 @@ class AdamWeightDecayOptimizer(tf.train.Optimizer):
     """Constructs a AdamWeightDecayOptimizer."""
     super(AdamWeightDecayOptimizer, self).__init__(False, name)
 
+    
     self.learning_rate = learning_rate
     self.weight_decay_rate = weight_decay_rate
     self.beta_1 = beta_1
@@ -114,13 +116,13 @@ class AdamWeightDecayOptimizer(tf.train.Optimizer):
 
       param_name = self._get_variable_name(param.name)
 
-      m = tf.get_variable(
+      m = tf.compat.v1.get_variable(
           name=param_name + "/adam_m",
           shape=param.shape.as_list(),
           dtype=tf.float32,
           trainable=False,
           initializer=tf.zeros_initializer())
-      v = tf.get_variable(
+      v = tf.compat.v1.get_variable(
           name=param_name + "/adam_v",
           shape=param.shape.as_list(),
           dtype=tf.float32,
```
