	'1?Z??'1?Z??!'1?Z??	??:?@??:?@!??:?@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$'1?Z???????AGr?????Y???B?i??*	??????k@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat o?ŏ??!jNq??>@)6?>W[???1 .Ԝ?;@:Preprocessing2U
Iterator::Model::ParallelMapV2??H.?!??!jNq?9@)??H.?!??1jNq?9@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???JY???!9	ą??5@)c?ZB>???1???w??&@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceM??St$??!ą??@\$@)M??St$??1ą??@\$@:Preprocessing2F
Iterator::Model??????!5'???A@)A??ǘ???1      $@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip_?L?J??!zel??P@)lxz?,C??1???g?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap`??"????!	ą??@;@) ?o_Ή?1??S??@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorS?!?uq{?!Oq??$@)S?!?uq{?1Oq??$@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 7.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2t10.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9??:?@I?6?S\2W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????!?????      ??!       "      ??!       *      ??!       2	Gr?????Gr?????!Gr?????:      ??!       B      ??!       J	???B?i?????B?i??!???B?i??R      ??!       Z	???B?i?????B?i??!???B?i??b      ??!       JCPU_ONLYY??:?@b q?6?S\2W@