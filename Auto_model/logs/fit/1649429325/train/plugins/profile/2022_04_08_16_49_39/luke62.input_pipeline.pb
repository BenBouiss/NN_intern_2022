	s�ѓ.A@s�ѓ.A@!s�ѓ.A@	_ы�і�?_ы�і�?!_ы�і�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$s�ѓ.A@�g��?��?A1Bx�q A@Y~8H���?*	�t�vN@2U
Iterator::Model::ParallelMapV2�1�=B͐?!�����:@)�1�=B͐?1�����:@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��T����?!F�ʿ�x:@)��T����?1F�ʿ�x:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���U�?!>��47@)�����?11�	5�63@:Preprocessing2F
Iterator::Model@�R�?!�J<�KD@)�����
�?1�M�) Q+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�1��㇢?!V���Q�M@)�*5{�h?1�*���M@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorZ��/-�c?!Ag��;�@)Z��/-�c?1Ag��;�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapq�0'h��?!�a��$a?@)겘�|\[?1�ҋ��@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateC;�Y�ݑ?!z �d�<@)8�*5{�U?1��iiU@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9^ы�і�?I]�KZ�X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�g��?��?�g��?��?!�g��?��?      ��!       "      ��!       *      ��!       2	1Bx�q A@1Bx�q A@!1Bx�q A@:      ��!       B      ��!       J	~8H���?~8H���?!~8H���?R      ��!       Z	~8H���?~8H���?!~8H���?b      ��!       JCPU_ONLYY^ы�і�?b q]�KZ�X@