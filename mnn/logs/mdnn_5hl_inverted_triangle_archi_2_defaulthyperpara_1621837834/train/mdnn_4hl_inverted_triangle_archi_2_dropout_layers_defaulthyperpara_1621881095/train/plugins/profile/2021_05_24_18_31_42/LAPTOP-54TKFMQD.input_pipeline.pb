	M?O?@M?O?@!M?O?@	滶?#@滶?#@!滶?#@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$M?O?@*:??H??A??j+?@Y???QI???*	?????r@2U
Iterator::Model::ParallelMapV2??? ?r??!.za? MF@)??? ?r??1.za? MF@:Preprocessing2F
Iterator::ModelR???Q??!:t???|P@)?rh??|??1?????X5@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate????????!??i??Z1@)??A?f??1zi!?-@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???{????!X ?=??&@))\???(??1?Pt#1#@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip[Ӽ???!?7?|A@)??@??ǈ?1?o???@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice	?^)ˀ?!?s?M?@)	?^)ˀ?1?s?M?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??ZӼ?t?!J}???R??)??ZӼ?t?1J}???R??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapz6?>W??!???2@)_?Q?k?1?ST?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t12.6 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9滶?#@I?(	?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	*:??H??*:??H??!*:??H??      ??!       "      ??!       *      ??!       2	??j+?@??j+?@!??j+?@:      ??!       B      ??!       J	???QI??????QI???!???QI???R      ??!       Z	???QI??????QI???!???QI???b      ??!       JCPU_ONLYY滶?#@b q?(	?V@