	+?َ#@+?َ#@!+?َ#@	??Q,?????Q,???!??Q,???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+?َ#@-!?lV??A:??H_!@YC?i?q???*	?????i?@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateo???T???!???p7V@)F%u???1?C?VMV@:Preprocessing2U
Iterator::Model::ParallelMapV2h??|?5??!8>??@)h??|?5??18>??@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?p=
ף??!?x?3V@)??e?c]??1bN}??@:Preprocessing2F
Iterator::Model	??g????!?[1??h@)U???N@??1|u\찆??:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip@?߾???!F??vYW@)???<,Ԋ?1/1%?m???:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicea2U0*???!?Qi"???)a2U0*???1?Qi"???:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?z?G???!?v??aV@)?q?????1???;g.??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*?s?!?Qi"???)a2U0*?s?1?Qi"???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Q,???I	?N???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	-!?lV??-!?lV??!-!?lV??      ??!       "      ??!       *      ??!       2	:??H_!@:??H_!@!:??H_!@:      ??!       B      ??!       J	C?i?q???C?i?q???!C?i?q???R      ??!       Z	C?i?q???C?i?q???!C?i?q???b      ??!       JCPU_ONLYY??Q,???b q	?N???X@