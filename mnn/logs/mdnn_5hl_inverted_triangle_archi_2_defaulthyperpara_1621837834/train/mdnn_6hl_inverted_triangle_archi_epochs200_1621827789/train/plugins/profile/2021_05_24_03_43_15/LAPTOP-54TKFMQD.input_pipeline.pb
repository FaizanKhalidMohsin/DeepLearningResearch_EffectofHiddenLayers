	????9?@????9?@!????9?@	??
_%@??
_%@!??
_%@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????9?@?c?]K???A%u???YDio?????*	43333?@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate	?c?Z??!@XLR?Q@)_?L???1?4
DD@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice??y?)??!?(??(?;@)??y?)??1?(??(?;@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??(\????!???b??V@)??(????1F|<?Տ)@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatM?J???!?????@)???x?&??12?֖?/@:Preprocessing2U
Iterator::Model::ParallelMapV2??ݓ????!?J?t	?@)??ݓ????1?J?t	?@:Preprocessing2F
Iterator::ModelEGr????!r?"??@"@)??_?L??1???]pB@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?O??n??!?7sU??Q@)Ǻ?????1???d??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorǺ???v?!???d????)Ǻ???v?1???d????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 10.7% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.1 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??
_%@I?ꞾTV@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?c?]K????c?]K???!?c?]K???      ??!       "      ??!       *      ??!       2	%u???%u???!%u???:      ??!       B      ??!       J	Dio?????Dio?????!Dio?????R      ??!       Z	Dio?????Dio?????!Dio?????b      ??!       JCPU_ONLYY??
_%@b q?ꞾTV@