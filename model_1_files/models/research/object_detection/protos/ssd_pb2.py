# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: object_detection/protos/ssd.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from . import anchor_generator_pb2 as object__detection_dot_protos_dot_anchor__generator__pb2
from . import box_coder_pb2 as object__detection_dot_protos_dot_box__coder__pb2
from . import box_predictor_pb2 as object__detection_dot_protos_dot_box__predictor__pb2
from . import fpn_pb2 as object__detection_dot_protos_dot_fpn__pb2
from . import hyperparams_pb2 as object__detection_dot_protos_dot_hyperparams__pb2
from . import image_resizer_pb2 as object__detection_dot_protos_dot_image__resizer__pb2
from . import losses_pb2 as object__detection_dot_protos_dot_losses__pb2
from . import matcher_pb2 as object__detection_dot_protos_dot_matcher__pb2
from . import post_processing_pb2 as object__detection_dot_protos_dot_post__processing__pb2
from . import region_similarity_calculator_pb2 as object__detection_dot_protos_dot_region__similarity__calculator__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='object_detection/protos/ssd.proto',
  package='object_detection.protos',
  syntax='proto2',
  serialized_options=None,
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\n!object_detection/protos/ssd.proto\x12\x17object_detection.protos\x1a.object_detection/protos/anchor_generator.proto\x1a\'object_detection/protos/box_coder.proto\x1a+object_detection/protos/box_predictor.proto\x1a!object_detection/protos/fpn.proto\x1a)object_detection/protos/hyperparams.proto\x1a+object_detection/protos/image_resizer.proto\x1a$object_detection/protos/losses.proto\x1a%object_detection/protos/matcher.proto\x1a-object_detection/protos/post_processing.proto\x1a:object_detection/protos/region_similarity_calculator.proto\"\xdc\x0b\n\x03Ssd\x12\x13\n\x0bnum_classes\x18\x01 \x01(\x05\x12<\n\rimage_resizer\x18\x02 \x01(\x0b\x32%.object_detection.protos.ImageResizer\x12G\n\x11\x66\x65\x61ture_extractor\x18\x03 \x01(\x0b\x32,.object_detection.protos.SsdFeatureExtractor\x12\x34\n\tbox_coder\x18\x04 \x01(\x0b\x32!.object_detection.protos.BoxCoder\x12\x31\n\x07matcher\x18\x05 \x01(\x0b\x32 .object_detection.protos.Matcher\x12R\n\x15similarity_calculator\x18\x06 \x01(\x0b\x32\x33.object_detection.protos.RegionSimilarityCalculator\x12)\n\x1a\x65ncode_background_as_zeros\x18\x0c \x01(\x08:\x05\x66\x61lse\x12 \n\x15negative_class_weight\x18\r \x01(\x02:\x01\x31\x12<\n\rbox_predictor\x18\x07 \x01(\x0b\x32%.object_detection.protos.BoxPredictor\x12\x42\n\x10\x61nchor_generator\x18\x08 \x01(\x0b\x32(.object_detection.protos.AnchorGenerator\x12@\n\x0fpost_processing\x18\t \x01(\x0b\x32\'.object_detection.protos.PostProcessing\x12+\n\x1dnormalize_loss_by_num_matches\x18\n \x01(\x08:\x04true\x12-\n\x1enormalize_loc_loss_by_codesize\x18\x0e \x01(\x08:\x05\x66\x61lse\x12+\n\x04loss\x18\x0b \x01(\x0b\x32\x1d.object_detection.protos.Loss\x12\x1f\n\x10\x66reeze_batchnorm\x18\x10 \x01(\x08:\x05\x66\x61lse\x12\'\n\x18inplace_batchnorm_update\x18\x0f \x01(\x08:\x05\x66\x61lse\x12\"\n\x14\x61\x64\x64_background_class\x18\x15 \x01(\x08:\x04true\x12(\n\x19\x65xplicit_background_class\x18\x18 \x01(\x08:\x05\x66\x61lse\x12)\n\x1ause_confidences_as_targets\x18\x16 \x01(\x08:\x05\x66\x61lse\x12\"\n\x17implicit_example_weight\x18\x17 \x01(\x02:\x01\x31\x12\x33\n$return_raw_detections_during_predict\x18\x1a \x01(\x08:\x05\x66\x61lse\x12?\n\x10mask_head_config\x18\x19 \x01(\x0b\x32%.object_detection.protos.Ssd.MaskHead\x1a\x84\x03\n\x08MaskHead\x12\x17\n\x0bmask_height\x18\x01 \x01(\x05:\x02\x31\x35\x12\x16\n\nmask_width\x18\x02 \x01(\x05:\x02\x31\x35\x12&\n\x18masks_are_class_agnostic\x18\x03 \x01(\x08:\x04true\x12\'\n\x1amask_prediction_conv_depth\x18\x04 \x01(\x05:\x03\x32\x35\x36\x12*\n\x1fmask_prediction_num_conv_layers\x18\x05 \x01(\x05:\x01\x32\x12+\n\x1c\x63onvolve_then_upsample_masks\x18\x06 \x01(\x08:\x05\x66\x61lse\x12\x1b\n\x10mask_loss_weight\x18\x07 \x01(\x02:\x01\x35\x12!\n\x15mask_loss_sample_size\x18\x08 \x01(\x05:\x02\x31\x36\x12>\n\x10\x63onv_hyperparams\x18\t \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12\x1d\n\x11initial_crop_size\x18\n \x01(\x05:\x02\x31\x35\"\xeb\x04\n\x13SsdFeatureExtractor\x12\x0c\n\x04type\x18\x01 \x01(\t\x12\x1b\n\x10\x64\x65pth_multiplier\x18\x02 \x01(\x02:\x01\x31\x12\x15\n\tmin_depth\x18\x03 \x01(\x05:\x02\x31\x36\x12>\n\x10\x63onv_hyperparams\x18\x04 \x01(\x0b\x32$.object_detection.protos.Hyperparams\x12:\n+override_base_feature_extractor_hyperparams\x18\t \x01(\x08:\x05\x66\x61lse\x12\x1a\n\x0fpad_to_multiple\x18\x05 \x01(\x05:\x01\x31\x12#\n\x14use_explicit_padding\x18\x07 \x01(\x08:\x05\x66\x61lse\x12\x1c\n\ruse_depthwise\x18\x08 \x01(\x08:\x05\x66\x61lse\x12>\n\x03\x66pn\x18\n \x01(\x0b\x32/.object_detection.protos.FeaturePyramidNetworksH\x00\x12M\n\x05\x62ifpn\x18\x13 \x01(\x0b\x32<.object_detection.protos.BidirectionalFeaturePyramidNetworksH\x00\x12\x34\n%replace_preprocessor_with_placeholder\x18\x0b \x01(\x08:\x05\x66\x61lse\x12\x15\n\nnum_layers\x18\x0c \x01(\x05:\x01\x36\x12\x1e\n\x16spaghettinet_arch_name\x18\x14 \x01(\t\x12\x1c\n\ruse_hardswish\x18\x15 \x01(\x08:\x05\x66\x61lseB\x17\n\x15\x66\x65\x61ture_pyramid_oneofJ\x04\x08\x06\x10\x07'
  ,
  dependencies=[object__detection_dot_protos_dot_anchor__generator__pb2.DESCRIPTOR,object__detection_dot_protos_dot_box__coder__pb2.DESCRIPTOR,object__detection_dot_protos_dot_box__predictor__pb2.DESCRIPTOR,object__detection_dot_protos_dot_fpn__pb2.DESCRIPTOR,object__detection_dot_protos_dot_hyperparams__pb2.DESCRIPTOR,object__detection_dot_protos_dot_image__resizer__pb2.DESCRIPTOR,object__detection_dot_protos_dot_losses__pb2.DESCRIPTOR,object__detection_dot_protos_dot_matcher__pb2.DESCRIPTOR,object__detection_dot_protos_dot_post__processing__pb2.DESCRIPTOR,object__detection_dot_protos_dot_region__similarity__calculator__pb2.DESCRIPTOR,])




_SSD_MASKHEAD = _descriptor.Descriptor(
  name='MaskHead',
  full_name='object_detection.protos.Ssd.MaskHead',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='mask_height', full_name='object_detection.protos.Ssd.MaskHead.mask_height', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=15,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mask_width', full_name='object_detection.protos.Ssd.MaskHead.mask_width', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=15,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='masks_are_class_agnostic', full_name='object_detection.protos.Ssd.MaskHead.masks_are_class_agnostic', index=2,
      number=3, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mask_prediction_conv_depth', full_name='object_detection.protos.Ssd.MaskHead.mask_prediction_conv_depth', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=256,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mask_prediction_num_conv_layers', full_name='object_detection.protos.Ssd.MaskHead.mask_prediction_num_conv_layers', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=2,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='convolve_then_upsample_masks', full_name='object_detection.protos.Ssd.MaskHead.convolve_then_upsample_masks', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mask_loss_weight', full_name='object_detection.protos.Ssd.MaskHead.mask_loss_weight', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(5),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mask_loss_sample_size', full_name='object_detection.protos.Ssd.MaskHead.mask_loss_sample_size', index=7,
      number=8, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=16,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='conv_hyperparams', full_name='object_detection.protos.Ssd.MaskHead.conv_hyperparams', index=8,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='initial_crop_size', full_name='object_detection.protos.Ssd.MaskHead.initial_crop_size', index=9,
      number=10, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=15,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1616,
  serialized_end=2004,
)

_SSD = _descriptor.Descriptor(
  name='Ssd',
  full_name='object_detection.protos.Ssd',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_classes', full_name='object_detection.protos.Ssd.num_classes', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='image_resizer', full_name='object_detection.protos.Ssd.image_resizer', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='feature_extractor', full_name='object_detection.protos.Ssd.feature_extractor', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='box_coder', full_name='object_detection.protos.Ssd.box_coder', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='matcher', full_name='object_detection.protos.Ssd.matcher', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='similarity_calculator', full_name='object_detection.protos.Ssd.similarity_calculator', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='encode_background_as_zeros', full_name='object_detection.protos.Ssd.encode_background_as_zeros', index=6,
      number=12, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='negative_class_weight', full_name='object_detection.protos.Ssd.negative_class_weight', index=7,
      number=13, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='box_predictor', full_name='object_detection.protos.Ssd.box_predictor', index=8,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='anchor_generator', full_name='object_detection.protos.Ssd.anchor_generator', index=9,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='post_processing', full_name='object_detection.protos.Ssd.post_processing', index=10,
      number=9, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='normalize_loss_by_num_matches', full_name='object_detection.protos.Ssd.normalize_loss_by_num_matches', index=11,
      number=10, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='normalize_loc_loss_by_codesize', full_name='object_detection.protos.Ssd.normalize_loc_loss_by_codesize', index=12,
      number=14, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='loss', full_name='object_detection.protos.Ssd.loss', index=13,
      number=11, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='freeze_batchnorm', full_name='object_detection.protos.Ssd.freeze_batchnorm', index=14,
      number=16, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='inplace_batchnorm_update', full_name='object_detection.protos.Ssd.inplace_batchnorm_update', index=15,
      number=15, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='add_background_class', full_name='object_detection.protos.Ssd.add_background_class', index=16,
      number=21, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=True,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='explicit_background_class', full_name='object_detection.protos.Ssd.explicit_background_class', index=17,
      number=24, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='use_confidences_as_targets', full_name='object_detection.protos.Ssd.use_confidences_as_targets', index=18,
      number=22, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='implicit_example_weight', full_name='object_detection.protos.Ssd.implicit_example_weight', index=19,
      number=23, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='return_raw_detections_during_predict', full_name='object_detection.protos.Ssd.return_raw_detections_during_predict', index=20,
      number=26, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='mask_head_config', full_name='object_detection.protos.Ssd.mask_head_config', index=21,
      number=25, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[_SSD_MASKHEAD, ],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=504,
  serialized_end=2004,
)


_SSDFEATUREEXTRACTOR = _descriptor.Descriptor(
  name='SsdFeatureExtractor',
  full_name='object_detection.protos.SsdFeatureExtractor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='type', full_name='object_detection.protos.SsdFeatureExtractor.type', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='depth_multiplier', full_name='object_detection.protos.SsdFeatureExtractor.depth_multiplier', index=1,
      number=2, type=2, cpp_type=6, label=1,
      has_default_value=True, default_value=float(1),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='min_depth', full_name='object_detection.protos.SsdFeatureExtractor.min_depth', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=16,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='conv_hyperparams', full_name='object_detection.protos.SsdFeatureExtractor.conv_hyperparams', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='override_base_feature_extractor_hyperparams', full_name='object_detection.protos.SsdFeatureExtractor.override_base_feature_extractor_hyperparams', index=4,
      number=9, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='pad_to_multiple', full_name='object_detection.protos.SsdFeatureExtractor.pad_to_multiple', index=5,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=1,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='use_explicit_padding', full_name='object_detection.protos.SsdFeatureExtractor.use_explicit_padding', index=6,
      number=7, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='use_depthwise', full_name='object_detection.protos.SsdFeatureExtractor.use_depthwise', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='fpn', full_name='object_detection.protos.SsdFeatureExtractor.fpn', index=8,
      number=10, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='bifpn', full_name='object_detection.protos.SsdFeatureExtractor.bifpn', index=9,
      number=19, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='replace_preprocessor_with_placeholder', full_name='object_detection.protos.SsdFeatureExtractor.replace_preprocessor_with_placeholder', index=10,
      number=11, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='num_layers', full_name='object_detection.protos.SsdFeatureExtractor.num_layers', index=11,
      number=12, type=5, cpp_type=1, label=1,
      has_default_value=True, default_value=6,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='spaghettinet_arch_name', full_name='object_detection.protos.SsdFeatureExtractor.spaghettinet_arch_name', index=12,
      number=20, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='use_hardswish', full_name='object_detection.protos.SsdFeatureExtractor.use_hardswish', index=13,
      number=21, type=8, cpp_type=7, label=1,
      has_default_value=True, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='feature_pyramid_oneof', full_name='object_detection.protos.SsdFeatureExtractor.feature_pyramid_oneof',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=2007,
  serialized_end=2626,
)

_SSD_MASKHEAD.fields_by_name['conv_hyperparams'].message_type = object__detection_dot_protos_dot_hyperparams__pb2._HYPERPARAMS
_SSD_MASKHEAD.containing_type = _SSD
_SSD.fields_by_name['image_resizer'].message_type = object__detection_dot_protos_dot_image__resizer__pb2._IMAGERESIZER
_SSD.fields_by_name['feature_extractor'].message_type = _SSDFEATUREEXTRACTOR
_SSD.fields_by_name['box_coder'].message_type = object__detection_dot_protos_dot_box__coder__pb2._BOXCODER
_SSD.fields_by_name['matcher'].message_type = object__detection_dot_protos_dot_matcher__pb2._MATCHER
_SSD.fields_by_name['similarity_calculator'].message_type = object__detection_dot_protos_dot_region__similarity__calculator__pb2._REGIONSIMILARITYCALCULATOR
_SSD.fields_by_name['box_predictor'].message_type = object__detection_dot_protos_dot_box__predictor__pb2._BOXPREDICTOR
_SSD.fields_by_name['anchor_generator'].message_type = object__detection_dot_protos_dot_anchor__generator__pb2._ANCHORGENERATOR
_SSD.fields_by_name['post_processing'].message_type = object__detection_dot_protos_dot_post__processing__pb2._POSTPROCESSING
_SSD.fields_by_name['loss'].message_type = object__detection_dot_protos_dot_losses__pb2._LOSS
_SSD.fields_by_name['mask_head_config'].message_type = _SSD_MASKHEAD
_SSDFEATUREEXTRACTOR.fields_by_name['conv_hyperparams'].message_type = object__detection_dot_protos_dot_hyperparams__pb2._HYPERPARAMS
_SSDFEATUREEXTRACTOR.fields_by_name['fpn'].message_type = object__detection_dot_protos_dot_fpn__pb2._FEATUREPYRAMIDNETWORKS
_SSDFEATUREEXTRACTOR.fields_by_name['bifpn'].message_type = object__detection_dot_protos_dot_fpn__pb2._BIDIRECTIONALFEATUREPYRAMIDNETWORKS
_SSDFEATUREEXTRACTOR.oneofs_by_name['feature_pyramid_oneof'].fields.append(
  _SSDFEATUREEXTRACTOR.fields_by_name['fpn'])
_SSDFEATUREEXTRACTOR.fields_by_name['fpn'].containing_oneof = _SSDFEATUREEXTRACTOR.oneofs_by_name['feature_pyramid_oneof']
_SSDFEATUREEXTRACTOR.oneofs_by_name['feature_pyramid_oneof'].fields.append(
  _SSDFEATUREEXTRACTOR.fields_by_name['bifpn'])
_SSDFEATUREEXTRACTOR.fields_by_name['bifpn'].containing_oneof = _SSDFEATUREEXTRACTOR.oneofs_by_name['feature_pyramid_oneof']
DESCRIPTOR.message_types_by_name['Ssd'] = _SSD
DESCRIPTOR.message_types_by_name['SsdFeatureExtractor'] = _SSDFEATUREEXTRACTOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

Ssd = _reflection.GeneratedProtocolMessageType('Ssd', (_message.Message,), {

  'MaskHead' : _reflection.GeneratedProtocolMessageType('MaskHead', (_message.Message,), {
    'DESCRIPTOR' : _SSD_MASKHEAD,
    '__module__' : 'object_detection.protos.ssd_pb2'
    # @@protoc_insertion_point(class_scope:object_detection.protos.Ssd.MaskHead)
    })
  ,
  'DESCRIPTOR' : _SSD,
  '__module__' : 'object_detection.protos.ssd_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.Ssd)
  })
_sym_db.RegisterMessage(Ssd)
_sym_db.RegisterMessage(Ssd.MaskHead)

SsdFeatureExtractor = _reflection.GeneratedProtocolMessageType('SsdFeatureExtractor', (_message.Message,), {
  'DESCRIPTOR' : _SSDFEATUREEXTRACTOR,
  '__module__' : 'object_detection.protos.ssd_pb2'
  # @@protoc_insertion_point(class_scope:object_detection.protos.SsdFeatureExtractor)
  })
_sym_db.RegisterMessage(SsdFeatureExtractor)


# @@protoc_insertion_point(module_scope)
