�	Ou��p�(@Ou��p�(@!Ou��p�(@	%�a��?%�a��?!%�a��?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Ou��p�(@�{�&�?A1�Z{��'@YeS��.�?*	
ףp=�Y@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�?�,ե?!^}��H�D@)�?�,ե?1^}��H�D@:Preprocessing2F
Iterator::Model��O��?! ��{�@@)�{���?1p{h"��2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat\ qW��?!�é�1@)7�Nx	N�?1�X���+@:Preprocessing2U
Iterator::Model::ParallelMapV2��@fgы?!!y��n*@)��@fgы?1!y��n*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip� ݗ3۱?! 2��P@)FCƣT�s?1o��J��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�b�J!p?!�M���@)�b�J!p?1�M���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�*��p�?!�7U�$�F@)z�ަ?�a?1���s�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate;S�Ʀ?!5-�ˣE@)Lo.2^?1��c��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 4.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9%�a��?I���z�X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�{�&�?�{�&�?!�{�&�?      ��!       "      ��!       *      ��!       2	1�Z{��'@1�Z{��'@!1�Z{��'@:      ��!       B      ��!       J	eS��.�?eS��.�?!eS��.�?R      ��!       Z	eS��.�?eS��.�?!eS��.�?b      ��!       JCPU_ONLYY%�a��?b q���z�X@Y      Y@q���D�>�?"�
both�Your program is POTENTIALLY input-bound because 4.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 