	?? ?rh@?? ?rh@!?? ?rh@	n!?Ma.@n!?Ma.@!n!?Ma.@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?? ?rh@(??y??A?O??n?@Y??B?i???*	????̐?@2F
Iterator::Model?p=
ף??!y??#??U@)?=?U???1?????T@:Preprocessing2U
Iterator::Model::ParallelMapV2??ݓ????!?fd*p@)??ݓ????1?fd*p@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??Ɯ?!ܑ?\?@)g??j+???1??XR??@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?ZӼ???!!??s@)46<???1?D?n?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?3??7???!8?0???(@)???_vO??14u????@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceM?O???!??C&h3??)M?O???1??C&h3??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorU???N@s?!??b?P??)U???N@s?1??b?P??:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?q??????!???6?@)Ǻ???f?1???4?)??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 15.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t14.9 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9n!?Ma.@I??H?3U@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	(??y??(??y??!(??y??      ??!       "      ??!       *      ??!       2	?O??n?@?O??n?@!?O??n?@:      ??!       B      ??!       J	??B?i?????B?i???!??B?i???R      ??!       Z	??B?i?????B?i???!??B?i???b      ??!       JCPU_ONLYYn!?Ma.@b q??H?3U@