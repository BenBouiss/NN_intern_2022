�	Y����%@Y����%@!Y����%@	n�4�G�?n�4�G�?!n�4�G�?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$Y����%@L��1�?AfN��Ķ$@YŒr�9>�?*	V-�-T@2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����ę?!��v��-?@)����ę?1��v��-?@:Preprocessing2U
Iterator::Model::ParallelMapV2i���?!��Z�zd>@)i���?1��Z�zd>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��U���?!�A�:%W0@)>+N��?12�̬�y+@:Preprocessing2F
Iterator::Model��y7�?!#�E��	F@)�IF��?1P�`�D^+@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip.��S�?!�q�p1�K@)�h��n?1��o�c�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorӢ>�6a?!K�v"��@)Ӣ>�6a?1K�v"��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��K�?!��1\2yA@)\;Qi[?1�,�Q�� @:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate/�혺+�?! 9w�o@@)�8�� nV?16w�F#�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 3.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9n�4�G�?I����X@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	L��1�?L��1�?!L��1�?      ��!       "      ��!       *      ��!       2	fN��Ķ$@fN��Ķ$@!fN��Ķ$@:      ��!       B      ��!       J	Œr�9>�?Œr�9>�?!Œr�9>�?R      ��!       Z	Œr�9>�?Œr�9>�?!Œr�9>�?b      ��!       JCPU_ONLYYn�4�G�?b q����X@Y      Y@q�"0Af"@"�
both�Your program is POTENTIALLY input-bound because 3.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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