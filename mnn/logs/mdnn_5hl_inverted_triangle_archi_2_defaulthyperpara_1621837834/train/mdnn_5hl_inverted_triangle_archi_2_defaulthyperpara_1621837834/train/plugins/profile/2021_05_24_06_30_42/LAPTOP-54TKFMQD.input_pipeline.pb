	]?C??k	@]?C??k	@!]?C??k	@	ī???@ī???@!ī???@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$]?C??k	@ ?o_???A?????@Y?_?L??*	43333Sc@2U
Iterator::Model::ParallelMapV2 ?o_Ω?!??3G?L@@) ?o_Ω?1??3G?L@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate??3????!?¾??=@)	??g????1???e)9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???S㥛?!Kl%??v1@)??&???1<?4?1?-@:Preprocessing2F
Iterator::Model?s????!?$???F@)????<,??1X??1|)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipo??ʡ??!=??wTK@)vq?-??1?g???p@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice???_vO~?!'?Kl%@)???_vO~?1'?Kl%@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?q????o?!h?Xd].@)?q????o?1h?Xd].@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa??+e??!X? ?
@@)F%u?k?1l???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 19.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9ī???@I?B?q#X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	 ?o_??? ?o_???! ?o_???      ??!       "      ??!       *      ??!       2	?????@?????@!?????@:      ??!       B      ??!       J	?_?L???_?L??!?_?L??R      ??!       Z	?_?L???_?L??!?_?L??b      ??!       JCPU_ONLYYī???@b q?B?q#X@