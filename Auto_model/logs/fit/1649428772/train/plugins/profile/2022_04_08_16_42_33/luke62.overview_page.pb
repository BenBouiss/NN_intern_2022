�	�ꐛ�9@�ꐛ�9@!�ꐛ�9@	k���^:�?k���^:�?!k���^:�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�ꐛ�9@.S��i�?A���S9@Y��	���?*	�~j�tw`@2U
Iterator::Model::ParallelMapV2ԙ{H�ޟ?!�}jg�7@)ԙ{H�ޟ?1�}jg�7@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�f/۞?!�_T��6@)�f/۞?1�_T��6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�mē�̠?!_H���8@)�T��E	�?1��D�M3@:Preprocessing2F
Iterator::Model�rf�B�?!����D@)#LQ.�_�?1B�g�2@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����A~?!je�"n@)����A~?1je�"n@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip[�����?!r�'M@)l\���|?1�Nj�[@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�3��k�?!d�e>�<@)؀q��m?1P��,/@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��tw��?!��3��H9@)��QF\ j?1;��~}F@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 1.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9k���^:�?I_�B��X@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	.S��i�?.S��i�?!.S��i�?      ��!       "      ��!       *      ��!       2	���S9@���S9@!���S9@:      ��!       B      ��!       J	��	���?��	���?!��	���?R      ��!       Z	��	���?��	���?!��	���?b      ��!       JCPU_ONLYYk���^:�?b q_�B��X@Y      Y@qx{����?"�
device�Your program is NOT input-bound because only 1.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
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