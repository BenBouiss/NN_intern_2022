	���{�.@���{�.@!���{�.@	���s�?���s�?!���s�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$���{�.@e��]���?A��[�6.@Y>�WXp�?*	��� ��U@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice,e�X�?!x
W3�9@),e�X�?1x
W3�9@:Preprocessing2F
Iterator::Model|E�^ӣ?!�~�G>F@)ˡE����?1	�E�7@:Preprocessing2U
Iterator::Model::ParallelMapV2KVE�ɨ�?!���I�4@)KVE�ɨ�?1���I�4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat}�H�F��?!����x�3@)�;���?1�L�o90@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�E|'f��?! a���K@)W'g(�xs?1���
��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��W9�i?!� e�L�@)��W9�i?1� e�L�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapE�@J�?!� Ot�4>@)���R�b?1�H���8@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�Σ����?!���T��;@)���pzW?1���_W�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.4% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9���s�?I�-�W��X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	e��]���?e��]���?!e��]���?      ��!       "      ��!       *      ��!       2	��[�6.@��[�6.@!��[�6.@:      ��!       B      ��!       J	>�WXp�?>�WXp�?!>�WXp�?R      ��!       Z	>�WXp�?>�WXp�?!>�WXp�?b      ��!       JCPU_ONLYY���s�?b q�-�W��X@