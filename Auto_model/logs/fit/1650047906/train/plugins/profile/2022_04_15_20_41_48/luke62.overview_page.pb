�	��)<@��)<@!��)<@	���k�w�?���k�w�?!���k�w�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$��)<@�����U�?A�/EH�;@YoF�W�Ǯ?*�v��oW@)      =2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��j�?!@�	��z:@)��j�?1@�	��z:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�S�D�[�?!�Vv;@)�����?1-zLH�5@:Preprocessing2U
Iterator::Model::ParallelMapV2�P�y�?!$Y���>3@)�P�y�?1$Y���>3@:Preprocessing2F
Iterator::Model�{+Ԡ?!�J�r?�A@)�L�x$^�?1�ytIF�/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipk�**�?!�Z�F�;P@)2�w�v?1E$�O�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensormɪ7u?!�)6�@)mɪ7u?1�)6�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����q�?!k\2���?@)r�Md�g?1By*HE�@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate �+�p��?!C-J��<@)Ӣ>�6a?1!�c�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9���k�w�?I"J4D�X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�����U�?�����U�?!�����U�?      ��!       "      ��!       *      ��!       2	�/EH�;@�/EH�;@!�/EH�;@:      ��!       B      ��!       J	oF�W�Ǯ?oF�W�Ǯ?!oF�W�Ǯ?R      ��!       Z	oF�W�Ǯ?oF�W�Ǯ?!oF�W�Ǯ?b      ��!       JCPU_ONLYY���k�w�?b q"J4D�X@Y      Y@q
&�z�Z�?"�
device�Your program is NOT input-bound because only 0.2% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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