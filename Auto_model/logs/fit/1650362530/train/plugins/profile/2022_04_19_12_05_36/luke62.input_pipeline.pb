	�',�F3@�',�F3@!�',�F3@	�3���?�3���?!�3���?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�',�F3@�.�H��?AaR||B�2@YY�&�ʨ?*	��S�V@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice]��'��?!��<�;@)]��'��?1��<�;@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��9y�	�?!\�#�еA@)��R�?1pj��ڸ1@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�I����?!H�\�Ʋ1@)�I����?1H�\�Ʋ1@:Preprocessing2F
Iterator::Model2���?!X��NM<@)|�����?1�Ǵg�d1@:Preprocessing2U
Iterator::Model::ParallelMapV2�7L4H��?!�H��%@)�7L4H��?1�H��%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipo�o�>;�?!�}��Q@)���)o?1�]S�4@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�|�X���?!s�1���?@)(G�`�d?1?}�!�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�S�D�[�?!�ү��=@)�	�y�]?1I���X @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.3% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9 4���?I��=��X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�.�H��?�.�H��?!�.�H��?      ��!       "      ��!       *      ��!       2	aR||B�2@aR||B�2@!aR||B�2@:      ��!       B      ��!       J	Y�&�ʨ?Y�&�ʨ?!Y�&�ʨ?R      ��!       Z	Y�&�ʨ?Y�&�ʨ?!Y�&�ʨ?b      ��!       JCPU_ONLYY 4���?b q��=��X@