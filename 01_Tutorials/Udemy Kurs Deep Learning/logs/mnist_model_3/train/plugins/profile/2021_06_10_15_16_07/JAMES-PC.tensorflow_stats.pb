"?S
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1????Y??@9????Y??@A????Y??@I????Y??@a??6A??i??6A???Unknown?
BHostIDLE"IDLE1ffffff?@Affffff?@a'??x?q??i2???1???Unknown
tHost_FusedMatMul"sequential/activation/Relu(1     ??@9     ??@A     ??@I     ??@a?^??K??i???:?????Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1fffffƁ@9fffffƁ@AfffffƁ@IfffffƁ@a????|???i?K>-aV???Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1?????Az@9?????Az@A?????Az@I?????Az@aUHx??!??i?,4??????Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1?????ax@9?????ax@A?????ax@I?????ax@a? P?Y???iCm=e????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(133333?t@933333?t@A33333?t@I33333?t@a??䨀?i>y}?<???Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     0p@9     0p@A     0p@I     0p@a^7??Tz?i???v%p???Unknown
v
Host_FusedMatMul"sequential/activation_1/Relu(1     @o@9     @o@A     @o@I     @o@aM?G?&y?i?/??r????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(133333?l@933333?l@A33333?l@I33333?l@a?z???Lw?i{06?????Unknown
^HostGatherV2"GatherV2(1fffffvk@9fffffvk@Afffffvk@Ifffffvk@a?R1v?ik?hD@????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     ?j@9     ?j@A     ?j@I     ?j@a?????mu?i?~?(???Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1?????ya@9?????ya@A?????ya@I?????ya@aX`g?? l?iai<D???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(133333\@933333\@A33333\@I33333\@a(y)"c?f?iz????Z???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??????Y@9??????Y@A??????Y@I??????Y@a??[j?d?i??6?o???Unknown
?HostReluGrad",gradient_tape/sequential/activation/ReluGrad(1??????W@9??????W@A??????W@I??????W@a}MRDYAc?ia?#??????Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1?????,S@9?????,S@A?????,S@I?????,S@aM?aXa?^?i9 ?@b????Unknown
vHost_FusedMatMul"sequential/activation_2/Relu(1?????M@9?????M@A?????M@I?????M@a2K?kW?i???	????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1      I@9      I@A      I@I      I@a
?l??T?i`?t'????Unknown
?HostReluGrad".gradient_tape/sequential/activation_1/ReluGrad(1??????D@9??????D@A??????D@I??????D@a?G??I?P?ið??q????Unknown
`HostGatherV2"
GatherV2_1(1??????=@9??????=@A??????=@I??????=@ad?y??G?i? {?p????Unknown
tHostSoftmax"sequential/activation_3/Softmax(1??????<@9??????<@A??????<@I??????<@a????-G?i?6?;????Unknown
tHost_FusedMatMul"sequential/dense_3/BiasAdd(133333?:@933333?:@A33333?:@I33333?:@a???a}E?izC?????Unknown
}HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(13333339@93333339@A3333339@I3333339@a+0?
HD?i?"?E?????Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?????L4@9?????L4@A?????L4@I?????L4@a?"|IzV@?i??E??????Unknown
iHostWriteSummary"WriteSummary(1      /@9      /@A      /@I      /@a??S??8?iLcE?????Unknown?
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1      -@9      -@A      -@I      -@a+V???V7?i-"^$?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?????L0@9?????L0@Affffff+@Iffffff+@a?啐P6?i?4p΍????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1??????(@9??????(@A??????(@I??????(@a?N????3?i
??????Unknown
dHostDataset"Iterator::Model(1??????6@9??????6@A      (@I      (@a??P3?i?}+?v????Unknown
Z HostArgMax"ArgMax(1??????&@9??????&@A??????&@I??????&@a^?C?Z02?i?E???????Unknown
s!HostDataset"Iterator::Model::ParallelMapV2(1333333%@9333333%@A333333%@I333333%@a???`?1?ilc???????Unknown
g"HostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a#bi?3?,?i??7?????Unknown
[#HostAddV2"Adam/add(1      !@9      !@A      !@I      !@a?Ǒ"]+?is?	d????Unknown
?$HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1333333 @9333333 @A333333 @I333333 @a??K{*?i-ѧA????Unknown
?%HostReluGrad".gradient_tape/sequential/activation_2/ReluGrad(1??????@9??????@A??????@I??????@a?읨?n)?i[",?????Unknown
?&HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1??????@9??????@A??????@I??????@a?4???(?i?b?(????Unknown
?'HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff-@9ffffff-@Affffff@Iffffff@a?X	4jw(?i??@?????Unknown
?(HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a?gY?&?i???????Unknown
\)HostArgMax"ArgMax_1(1333333@9333333@A333333@I333333@a+0?
H$?i?@vb????Unknown
?*HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??P#?i?w??????Unknown
?+HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a??P#?i?1???????Unknown
l,HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @a2J???!?i q???????Unknown
V-HostSum"Sum_2(1??????@9??????@A??????@I??????@anr2Rb!?iG?? ?????Unknown
{.HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??V?? ?i?]v??????Unknown
Y/HostPow"Adam/Pow(1??????@9??????@A??????@I??????@a?҄??B?i??M??????Unknown
x0HostDataset"#Iterator::Model::ParallelMapV2::Zip(1????̌B@9????̌B@A      @I      @a#bi?3??i)%?k?????Unknown
?1HostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1??????@9??????@A??????@I??????@a??[`T?iX??????Unknown
?2HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?$z??i,)w?????Unknown
v3HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@aV??????i	?+?H????Unknown
?4HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@aV??????i?8?@????Unknown
[5HostPow"
Adam/Pow_1(1??????@9??????@A??????@I??????@a?N?????i0.???????Unknown
e6Host
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@a?N?????iz#??F????Unknown?
?7HostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@a?N?????i??G?????Unknown
V8HostCast"Cast(1ffffff@9ffffff@Affffff@Iffffff@aO&??%?i?? ?v????Unknown
X9HostCast"Cast_2(1??????@9??????@A??????@I??????@anr2Rb?iXX??????Unknown
?:HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9????????A??????@I????????aܵd?~??i~Ӧ?????Unknown
t;HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a\wL??iZ???????Unknown
`<HostDivNoNan"
div_no_nan(1ffffff@9ffffff@Affffff@Iffffff@a\wL??i67?ot????Unknown
?=HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a\wL??ii???????Unknown
b>HostDivNoNan"div_no_nan_1(1??????@9??????@A??????@I??????@a??[`T?i??~9\????Unknown
~?HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1?????? @9?????? @A?????? @I?????? @aw9@??
?i??ad?????Unknown
]@HostCast"Adam/Cast_1(1?????? @9?????? @A?????? @I?????? @aw9@??
?i??D?4????Unknown
XAHostCast"Cast_1(1?????? @9?????? @A?????? @I?????? @aw9@??
?i??'??????Unknown
XBHostEqual"Equal(1?????? @9?????? @A?????? @I?????? @aw9@??
?i??
?????Unknown
vCHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a?$z?	?ioP?s????Unknown
?DHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??a?wҧ??ib?y?????Unknown
?EHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?????L0@9?????L0@A????????I????????a6?at??i>???????Unknown
vFHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a??P?i??&k????Unknown
TGHostMul"Mul(1      ??9      ??A      ??I      ??a??P?ir?i?????Unknown
?HHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1ffffff??9ffffff??Affffff??Iffffff??aO&??%?iȐ? ????Unknown
?IHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??aO&??%?i(?H????Unknown
tJHostReadVariableOp"Adam/Cast/ReadVariableOp(1????????9????????A????????I????????aܵd?~? ?i?["??????Unknown
uKHostReadVariableOp"div_no_nan/ReadVariableOp(1????????9????????A????????I????????aܵd?~? ?iB???????Unknown
wLHostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????aܵd?~? ?i???????Unknown
?MHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??aЊ?????>i??sTO????Unknown
vNHostAssignAddVariableOp"AssignAddVariableOp_4(1????????9????????A????????I????????a??[`T?>i?4??????Unknown
?OHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?$z??>i??V?????Unknown
?PHostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?$z??>iE?y?????Unknown
vQHostAssignAddVariableOp"AssignAddVariableOp_3(1????????9????????A????????I????????a????-?>i!??\????Unknown
?RHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1????????9????????A????????I????????a????-?>i????K????Unknown
oSHostReadVariableOp"Adam/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??aO&??%?>i?S??o????Unknown
aTHostIdentity"Identity(1ffffff??9ffffff??Affffff??Iffffff??aO&??%?>i??Փ????Unknown?
?UHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??aO&??%?>i??h??????Unknown
vVHostAssignAddVariableOp"AssignAddVariableOp_1(1333333??9333333??A333333??I333333??aЊ?????>i?<??????Unknown
wWHostReadVariableOp"div_no_nan/ReadVariableOp_1(1????????9????????A????????I????????a6?at??>iG??e?????Unknown
yXHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1????????9????????A????????I????????a6?at??>i?????????Unknown*?R
uHostFlushSummaryWriter"FlushSummaryWriter(1????Y??@9????Y??@A????Y??@I????Y??@a?+?]x??i?+?]x???Unknown?
tHost_FusedMatMul"sequential/activation/Relu(1     ??@9     ??@A     ??@I     ??@a???ꕈ??i??m?L???Unknown
yHostMatMul"%gradient_tape/sequential/dense/MatMul(1fffffƁ@9fffffƁ@AfffffƁ@IfffffƁ@a???I??i?L7??????Unknown
{HostMatMul"'gradient_tape/sequential/dense_1/MatMul(1?????Az@9?????Az@A?????Az@I?????Az@a?Ig??%??i??\l9???Unknown
}HostMatMul")gradient_tape/sequential/dense_1/MatMul_1(1?????ax@9?????ax@A?????ax@I?????ax@a???Z??i?+?aԖ???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(133333?t@933333?t@A33333?t@I33333?t@aO??yӃ?i3?WH"????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1     0p@9     0p@A     0p@I     0p@a`?b??i o?w&$???Unknown
vHost_FusedMatMul"sequential/activation_1/Relu(1     @o@9     @o@A     @o@I     @o@a??H@?}?iDf?`???Unknown
?	HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(133333?l@933333?l@A33333?l@I33333?l@a???6O?{?i!:??w????Unknown
^
HostGatherV2"GatherV2(1fffffvk@9fffffvk@Afffffvk@Ifffffvk@a??[?Mz?i????????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1     ?j@9     ?j@A     ?j@I     ?j@a??6?<?y?i?^?<????Unknown
}HostMatMul")gradient_tape/sequential/dense_2/MatMul_1(1?????ya@9?????ya@A?????ya@I?????ya@a?3??p?i?ĵ?? ???Unknown
?HostSoftmaxCrossEntropyWithLogits":categorical_crossentropy/softmax_cross_entropy_with_logits(133333\@933333\@A33333\@I33333\@a???2??j?iZ??vp;???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1??????Y@9??????Y@A??????Y@I??????Y@a?M????h?i?!iET???Unknown
?HostReluGrad",gradient_tape/sequential/activation/ReluGrad(1??????W@9??????W@A??????W@I??????W@al?.8?f?iC-?I/k???Unknown
{HostMatMul"'gradient_tape/sequential/dense_2/MatMul(1?????,S@9?????,S@A?????,S@I?????,S@a????]b?i?KA،}???Unknown
vHost_FusedMatMul"sequential/activation_2/Relu(1?????M@9?????M@A?????M@I?????M@a2o???[?i???c|????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1      I@9      I@A      I@I      I@aܴ/??W?i?=Ju????Unknown
?HostReluGrad".gradient_tape/sequential/activation_1/ReluGrad(1??????D@9??????D@A??????D@I??????D@ab????S?iJl?R????Unknown
`HostGatherV2"
GatherV2_1(1??????=@9??????=@A??????=@I??????=@a4?_ ??L?i?a?ru????Unknown
tHostSoftmax"sequential/activation_3/Softmax(1??????<@9??????<@A??????<@I??????<@an ׇ?K?i*??Z????Unknown
tHost_FusedMatMul"sequential/dense_3/BiasAdd(133333?:@933333?:@A33333?:@I33333?:@af???I?i???|?????Unknown
}HostMatMul")gradient_tape/sequential/dense_3/MatMul_1(13333339@93333339@A3333339@I3333339@aK`<??"H?i?2Ȼ???Unknown
?HostBiasAddGrad"2gradient_tape/sequential/dense/BiasAdd/BiasAddGrad(1?????L4@9?????L4@A?????L4@I?????L4@a<???eqC?i?????????Unknown
iHostWriteSummary"WriteSummary(1      /@9      /@A      /@I      /@a?˫???=?i??}?Z????Unknown?
{HostMatMul"'gradient_tape/sequential/dense_3/MatMul(1      -@9      -@A      -@I      -@a{-???;?i?<?|?????Unknown
?HostDataset"5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(1?????L0@9?????L0@Affffff+@Iffffff+@a??C>:?i??E????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1??????(@9??????(@A??????(@I??????(@an	#2??7?i??y]????Unknown
dHostDataset"Iterator::Model(1??????6@9??????6@A      (@I      (@a?[?ݛ?6?i????????Unknown
ZHostArgMax"ArgMax(1??????&@9??????&@A??????&@I??????&@a???
W?5?i?
כ?????Unknown
sHostDataset"Iterator::Model::ParallelMapV2(1333333%@9333333%@A333333%@I333333%@a??>7N4?i??^1????Unknown
g HostStridedSlice"strided_slice(1      "@9      "@A      "@I      "@a?Dt?t=1?ia??Y????Unknown
[!HostAddV2"Adam/add(1      !@9      !@A      !@I      !@a??4?CH0?i?g2b????Unknown
?"HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1333333 @9333333 @A333333 @I333333 @a?{?8/?iF???R????Unknown
?#HostReluGrad".gradient_tape/sequential/activation_2/ReluGrad(1??????@9??????@A??????@I??????@a??}D.?ice??6????Unknown
?$HostMul"Lgradient_tape/categorical_crossentropy/softmax_cross_entropy_with_logits/mul(1??????@9??????@A??????@I??????@a[ ?)?-?iU?y?????Unknown
?%HostDataset"?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate(1ffffff-@9ffffff-@Affffff@Iffffff@a~Ʌ?-?i?????????Unknown
?&HostDataset"OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice(1ffffff@9ffffff@Affffff@Iffffff@a/-t3+?i#?$??????Unknown
\'HostArgMax"ArgMax_1(1333333@9333333@A333333@I333333@aK`<??"(?i鋒????Unknown
?(HostBiasAddGrad"4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?[?ݛ?&?i?jP??????Unknown
?)HostBiasAddGrad"4gradient_tape/sequential/dense_3/BiasAdd/BiasAddGrad(1      @9      @A      @I      @a?[?ݛ?&?i?I??????Unknown
l*HostIteratorGetNext"IteratorGetNext(1      @9      @A      @I      @af?q?9%?i??F????Unknown
V+HostSum"Sum_2(1??????@9??????@A??????@I??????@a?RX?%?$?i?ԑ????Unknown
{,HostSum"*categorical_crossentropy/weighted_loss/Sum(1      @9      @A      @I      @a??8?'#?iĥxQ?????Unknown
Y-HostPow"Adam/Pow(1??????@9??????@A??????@I??????@a???:?"?i3P<k?????Unknown
x.HostDataset"#Iterator::Model::ParallelMapV2::Zip(1????̌B@9????̌B@A      @I      @a?Dt?t=!?iw??B?????Unknown
?/HostReadVariableOp"(sequential/dense_3/MatMul/ReadVariableOp(1??????@9??????@A??????@I??????@a??Z<a? ?i&}??????Unknown
?0HostTile";gradient_tape/categorical_crossentropy/weighted_loss/Tile_1(1      @9      @A      @I      @a?$?'%??i??)?????Unknown
v1HostAssignAddVariableOp"AssignAddVariableOp_2(1??????@9??????@A??????@I??????@a?S+?Y?iA??????Unknown
?2HostBiasAddGrad"4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGrad(1??????@9??????@A??????@I??????@a?S+?Y?i?q???????Unknown
[3HostPow"
Adam/Pow_1(1??????@9??????@A??????@I??????@an	#2???i???~????Unknown
e4Host
LogicalAnd"
LogicalAnd(1??????@9??????@A??????@I??????@an	#2???i????<????Unknown?
?5HostReadVariableOp")sequential/dense_3/BiasAdd/ReadVariableOp(1??????@9??????@A??????@I??????@an	#2???i?$??????Unknown
V6HostCast"Cast(1ffffff@9ffffff@Affffff@Iffffff@aB ?5Mt?iQ?py?????Unknown
X7HostCast"Cast_2(1??????@9??????@A??????@I??????@a?RX?%??iܟ?K????Unknown
?8HostDataset"AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1??????@9????????A??????@I????????aФ%????iAE?Z?????Unknown
t9HostAssignAddVariableOp"AssignAddVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a???????i???Vx????Unknown
`:HostDivNoNan"
div_no_nan(1ffffff@9ffffff@Affffff@Iffffff@a???????iNS????Unknown
?;HostReadVariableOp")sequential/dense_1/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a???????i??aO?????Unknown
b<HostDivNoNan"div_no_nan_1(1??????@9??????@A??????@I??????@a??Z<a??i_?k*????Unknown
~=HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1?????? @9?????? @A?????? @I?????? @a2@(?9?i??:??????Unknown
]>HostCast"Adam/Cast_1(1?????? @9?????? @A?????? @I?????? @a2@(?9?i?7
?????Unknown
X?HostCast"Cast_1(1?????? @9?????? @A?????? @I?????? @a2@(?9?i%y?W?????Unknown
X@HostEqual"Equal(1?????? @9?????? @A?????? @I?????? @a2@(?9?ig??????Unknown
vAHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1       @9       @A       @I       @a?$?'%??iZ=??????Unknown
?BHostReadVariableOp"(sequential/dense_1/MatMul/ReadVariableOp(1333333??9333333??A333333??I333333??a??.9
?i "??????Unknown
?CHostDataset"/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap(1?????L0@9?????L0@A????????I????????a(?U????iW.??`????Unknown
vDHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1      ??9      ??A      ??I      ??a?[?ݛ??i?;??????Unknown
TEHostMul"Mul(1      ??9      ??A      ??I      ??a?[?ݛ??i???????Unknown
?FHostDivNoNan"Egradient_tape/categorical_crossentropy/weighted_loss/value/div_no_nan(1ffffff??9ffffff??Affffff??Iffffff??aB ?5Mt?i?ߨn????Unknown
?GHostReadVariableOp")sequential/dense_2/BiasAdd/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??aB ?5Mt?i1?z?????Unknown
tHHostReadVariableOp"Adam/Cast/ReadVariableOp(1????????9????????A????????I????????aФ%????i??*????Unknown
uIHostReadVariableOp"div_no_nan/ReadVariableOp(1????????9????????A????????I????????aФ%????i_3	?c????Unknown
wJHostReadVariableOp"div_no_nan_1/ReadVariableOp(1????????9????????A????????I????????aФ%????i?g??????Unknown
?KHostReadVariableOp"'sequential/dense/BiasAdd/ReadVariableOp(1333333??9333333??A333333??I333333??a]I???c?i????????Unknown
vLHostAssignAddVariableOp"AssignAddVariableOp_4(1????????9????????A????????I????????a??Z<a? ?ic?G?@????Unknown
?MHostCast"8categorical_crossentropy/weighted_loss/num_elements/Cast(1      ??9      ??A      ??I      ??a?$?'%??>i9<??}????Unknown
?NHostDivNoNan",categorical_crossentropy/weighted_loss/value(1      ??9      ??A      ??I      ??a?$?'%??>i???????Unknown
vOHostAssignAddVariableOp"AssignAddVariableOp_3(1????????9????????A????????I????????an ׇ??>iP:?I?????Unknown
?PHostReadVariableOp"&sequential/dense/MatMul/ReadVariableOp(1????????9????????A????????I????????an ׇ??>i???t)????Unknown
oQHostReadVariableOp"Adam/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??aB ?5Mt?>i?S?]T????Unknown
aRHostIdentity"Identity(1ffffff??9ffffff??Affffff??Iffffff??aB ?5Mt?>i??0F????Unknown?
?SHostReadVariableOp"(sequential/dense_2/MatMul/ReadVariableOp(1ffffff??9ffffff??Affffff??Iffffff??aB ?5Mt?>i?)?.?????Unknown
vTHostAssignAddVariableOp"AssignAddVariableOp_1(1333333??9333333??A333333??I333333??a]I???c?>iT?*??????Unknown
wUHostReadVariableOp"div_no_nan/ReadVariableOp_1(1????????9????????A????????I????????a(?U????>i?y{?????Unknown
yVHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1????????9????????A????????I????????a(?U????>i      ???Unknown2GPU